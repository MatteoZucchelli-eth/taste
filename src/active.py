"""Active pair selection for the Phase B and Phase C critics (CONTEXT "Phase C decisions", step 4).

Simulation: each of the 18 Phase B raters answers "which of these two do you prefer?" from their own ratings
(pool = their training images in that fold; tied pairs cannot be asked). All strategies start from the same 5
random pairs, then ask in batches and refit after each batch, up to 250 pairs. Evaluation on the same held-out
test images as Phase B/C (headline: pairs with score gap >= 10).

Strategies (same for both models):
  random       uniformly random non-tied pair (= the Phase B/C protocol, but nested)
  uncertainty  pair with predicted P(prefer) closest to 0.5
  infogain     D-optimal / expected information about the person's parameters under a Laplace posterior:
               score = p(1-p) d^T H^-1 d, d = feature difference of the pair, H = prior precision + sum over asked
               pairs of p(1-p) d d^T; batches chosen greedily with rank-1 updates (Bayesian experimental design,
               cf. ED-PBRL, Schacht et al. 2025; Sadigh 2017).
Models:
  B  Phase B: s = beta * consensus + w . f (769 params), lambda by inner CV over images (at n in CV_AT; otherwise
     the last chosen lambda). Consensus critic retrained per fold exactly as in Phase B. Selection prior: w ~ N(0,
     1/(2 n lambda)) (the fitted objective), lambda=inf -> 1e-3 for selection only.
  C  Phase C (frozen CLIP, K*=8): s = z . U(f) + w . f; z MAP under N(mu, Sigma/tau); tau, lambda_w from the
     Phase C validation choice for the budget bin. Selection on z (9 params). Shared models from
     checkpoints/phase_c/shared_K8_fold*.pt.

Usage: python src/active.py --models C B
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phase_c_lowrank as C  # noqa: E402
from phase_b_critics import fit_personal, fit_with_cv, load_emb, load_ratings, train_consensus  # noqa: E402

DEV = "cuda"
SCHED = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 250]
CV_AT = {10, 25, 50, 100, 250}
STRATS = ["random", "uncertainty", "infogain"]
N_CAND = 3000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- pair pool + selection
def candidates(y, asked, rng, n=N_CAND):
    m = len(y)
    i, j = rng.integers(0, m, 3 * n), rng.integers(0, m, 3 * n)
    a, b = np.minimum(i, j), np.maximum(i, j)
    keep = (a != b) & (y[a] != y[b])
    a, b = a[keep], b[keep]
    key = a * m + b
    _, u = np.unique(key, return_index=True)
    u = np.sort(u)
    a, b, key = a[u], b[u], key[u]
    keep = ~np.isin(key, np.fromiter(asked, dtype=np.int64, count=len(asked)) if asked else np.empty(0, np.int64))
    return a[keep][:n], b[keep][:n]


def greedy_info(D, pq, Hinv, k):
    """D: n x d candidate feature differences, pq: p(1-p), Hinv: d x d. Greedy D-optimal batch of size k."""
    DH = D @ Hinv
    picks = []
    for _ in range(min(k, len(D))):
        sc = pq * (DH * D).sum(1)
        if picks:
            sc[torch.tensor(picks, device=D.device)] = -1
        b = int(torch.argmax(sc))
        picks.append(b)
        v = DH[b]  # = Hinv d_b
        denom = 1 + pq[b] * (v @ D[b])
        DH = DH - pq[b] * torch.outer(D @ v, v) / denom
    return picks


def select(strategy, k, rng, a, b, s, D=None, Hinv=None):
    if strategy == "random":
        return list(range(min(k, len(a))))  # candidates are already a random sample
    p = torch.sigmoid(torch.as_tensor(s[a] - s[b], device=DEV, dtype=torch.float64))
    if strategy == "uncertainty":
        return torch.argsort((p - 0.5).abs())[:k].tolist()
    return greedy_info(D, p * (1 - p), Hinv, k)


# ---------------------------------------------------------------- per-run state
class Run:
    def __init__(self, pid, f, seed, strategy, p_tr, y_tr, p_te, y_te, init):
        self.pid, self.f, self.seed, self.strategy = pid, f, seed, strategy
        self.p_tr, self.y_tr, self.p_te, self.y_te = p_tr, y_tr, p_te, y_te
        self.rng = np.random.default_rng([seed, f, pid, STRATS.index(strategy)])
        self.I, self.J = list(init[0]), list(init[1])
        self.asked = set((np.array(self.I) * len(y_tr) + np.array(self.J)).tolist())
        self.lam = 1e-3

    def pairs(self):
        I, J = np.array(self.I), np.array(self.J)
        return I, J, (self.y_tr[I] > self.y_tr[J]).astype(float)

    def add(self, a, b, picks):
        for k in picks:
            self.I.append(int(a[k]))
            self.J.append(int(b[k]))
            self.asked.add(int(a[k]) * len(self.y_tr) + int(b[k]))


def make_runs(ctx, f, pos, seeds):
    S, ids, rd, fold_of, *_ = ctx
    runs = []
    for pid in ids:
        g = rd[pid]
        tr, te = fold_of[pid] != f, fold_of[pid] == f
        p_tr, y_tr = pos[g.loc[tr, "image_id"]].to_numpy(), g.loc[tr, "rating"].to_numpy(float)
        p_te, y_te = pos[g.loc[te, "image_id"]].to_numpy(), g.loc[te, "rating"].to_numpy(float)
        for s in range(seeds):
            init = candidates(y_tr, set(), np.random.default_rng([s, f, pid]), n=SCHED[0])
            for st in STRATS:
                runs.append(Run(pid, f, s, st, p_tr, y_tr, p_te, y_te, init))
    return runs


def record(rows, model, r, n, s_te_full):
    m = C.metrics(r.y_te, s_te_full)
    rows.append(dict(model=model, participant_id=r.pid, fold=r.f, seed=r.seed, strategy=r.strategy, n_pairs=n,
                     acc=m["acc"], acc_g10=m["acc_g10"]))


# ---------------------------------------------------------------- model C
def cfg_for(cfg, f, n):
    c = cfg[cfg.fold == f].set_index("budget")
    b = min([x for x in [25, 50, 100, 250, 500, 1000, 2500] if x >= n] or [2500])
    return float(c.loc[b, "full_tau"]), float(c.loc[b, "full_lam_w"])


def run_C(a, ctx, pos, Eall):
    cfg = pd.read_csv(a.phase_c / "chosen_cfg.csv")
    rows = []
    for f in a.folds_run:
        ck = torch.load(Path("checkpoints/phase_c") / f"shared_K8_fold{f}.pt")
        U, Z = ck["U"].to(DEV), ck["Z"].to(DEV)
        mu, Sinv = C.population_prior(Z)
        runs = make_runs(ctx, f, pos, a.seeds)
        for step, n in enumerate(SCHED):
            tau, lam = cfg_for(cfg, f, n)
            probs = []
            for r in runs:
                I, J, T = r.pairs()
                probs.append((r.p_tr[I], r.p_tr[J], T))
            Zb, Wb = C.fit_users(U, Eall, mu, Sinv, probs, tau, lam)
            Sc = (U @ Zb.T + Eall @ Wb.T)
            Sc_np = Sc.cpu().numpy()
            for k, r in enumerate(runs):
                record(rows, "C", r, n, Sc_np[r.p_te, k])
            if step + 1 == len(SCHED):
                break
            k_new = SCHED[step + 1] - n
            for k, r in enumerate(runs):
                ca, cb = candidates(r.y_tr, r.asked, r.rng)
                s_loc = Sc_np[r.p_tr, k]
                D = Hinv = None
                if r.strategy == "infogain":
                    Ut = U[r.p_tr]
                    I, J, _ = r.pairs()
                    Da = Ut[I] - Ut[J]
                    pa = torch.sigmoid(Sc[r.p_tr[I], k] - Sc[r.p_tr[J], k])
                    H = tau * Sinv + (Da * (pa * (1 - pa))[:, None]).T @ Da
                    Hinv = torch.linalg.inv(H)
                    D = Ut[ca] - Ut[cb]
                r.add(ca, cb, select(r.strategy, k_new, r.rng, ca, cb, s_loc, D, Hinv))
        log(f"C fold {f} done")
        pd.DataFrame(rows).to_csv(a.out / "curve_C.csv", index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model B
def run_B(a, ctx, d, pos, E):
    S, ids, rd, fold_of, *_ = ctx
    choice = json.loads((a.phase_b / "consensus_select.json").read_text())
    Eall = torch.tensor(E, device=DEV, dtype=torch.float64)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    rows = []
    for f in a.folds_run:
        test_imgs = set()
        for pid in ids:
            test_imgs |= set(rd[pid].loc[fold_of[pid] == f, "image_id"])
        d_cons = d[~d.participant_id.isin(ids) & ~d.image_id.isin(test_imgs)]
        sc_all, _, _, _ = train_consensus(E, d_cons, pos, choice["head"], choice["steps"], a.seed + f)
        sc_t = torch.tensor(sc_all, device=DEV, dtype=torch.float64)
        Phi = torch.cat([sc_t[:, None], Eall], 1)  # features for selection: [consensus, f]
        runs = make_runs(ctx, f, pos, a.seeds)
        log(f"B fold {f}: consensus retrained; {len(runs)} runs")
        for r in runs:
            sc_tr, E_tr = sc_t[r.p_tr], Eall[r.p_tr]
            for step, n in enumerate(SCHED):
                I, J, T = r.pairs()
                pi, pj = torch.tensor(I, device=DEV), torch.tensor(J, device=DEV)
                t = torch.tensor(T, device=DEV, dtype=torch.float64)
                if n in CV_AT:
                    th, r.lam = fit_with_cv(sc_tr, E_tr, pi, pj, t, gen)
                else:
                    th = fit_personal(sc_tr, E_tr, pi, pj, t, r.lam)
                s_all = (th[0] * sc_t + Eall @ th[1:])
                record(rows, "B", r, n, s_all[r.p_te].cpu().numpy())
                if step + 1 == len(SCHED):
                    break
                k_new = SCHED[step + 1] - n
                ca, cb = candidates(r.y_tr, r.asked, r.rng)
                s_loc = s_all[r.p_tr].cpu().numpy()
                D = Hinv = None
                if r.strategy == "infogain":
                    Pt = Phi[r.p_tr]
                    Da = Pt[I] - Pt[J]
                    pa = torch.sigmoid(torch.as_tensor(s_loc[I] - s_loc[J], device=DEV))
                    lam_sel = r.lam if np.isfinite(r.lam) else 1e-3
                    prior = torch.full((Phi.shape[1],), 2 * n * lam_sel, device=DEV, dtype=torch.float64)
                    prior[0] = 1e-3  # beta is not penalised
                    H = torch.diag(prior) + (Da * (pa * (1 - pa))[:, None]).T @ Da
                    Hinv = torch.linalg.inv(H)
                    D = Pt[ca] - Pt[cb]
                r.add(ca, cb, select(r.strategy, k_new, r.rng, ca, cb, s_loc, D, Hinv))
        log(f"B fold {f} done")
        pd.DataFrame(rows).to_csv(a.out / "curve_B.csv", index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def report(a, S):
    parts = [pd.read_csv(p) for p in [a.out / "curve_C.csv", a.out / "curve_B.csv"] if p.exists()]
    R = pd.concat(parts)
    R["group"] = R["participant_id"].map(S.set_index("participant_id")["group"])
    per = R.groupby(["model", "group", "strategy", "n_pairs", "participant_id"])[["acc", "acc_g10"]].mean().reset_index()
    per.to_csv(a.out / "per_rater.csv", index=False)
    med = per.groupby(["model", "group", "strategy", "n_pairs"])["acc_g10"].median().unstack("strategy")
    log("median acc (gap>=10) over raters:\n" + med.round(4).to_string())
    # pairs needed to reach the random strategy's accuracy at 250 pairs (per model/group, median curve)
    out = {}
    for (mdl, grp), g in med.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        target = g.loc[250, "random"]
        out[f"{mdl}_{grp}"] = {st: (int(g.index[np.argmax(g[st].to_numpy() >= target)])
                                    if (g[st] >= target).any() else None) for st in STRATS}
        out[f"{mdl}_{grp}"]["random_acc_at_250"] = float(target)
    wide = per.pivot_table(index=["model", "group", "n_pairs", "participant_id"], columns="strategy", values="acc_g10")
    for st in ["uncertainty", "infogain"]:
        wide[f"d_{st}"] = 100 * (wide[st] - wide["random"])
    dd = wide.groupby(level=[0, 1, 2])[["d_uncertainty", "d_infogain"]].agg(["median", lambda s: int((s > 0).sum())])
    log("gain over random (pts, gap>=10): median / #raters better\n" + dd.round(2).to_string())
    (a.out / "summary.json").write_text(json.dumps(dict(pairs_to_match_random_250=out, sched=SCHED,
                                                        seeds=a.seeds, n_cand=N_CAND,
                                                        runtime_s=round(time.time() - T0)), indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sty = {"random": ":", "uncertainty": "--", "infogain": "-"}
    col = {"B": "0.45", "C": "C3"}
    for k, grp in enumerate(["selected", "control"]):
        for (mdl, st), g in per[per.group == grp].groupby(["model", "strategy"]):
            m = g.groupby("n_pairs")["acc_g10"].median()
            ax[k].plot(m.index, m.values, sty[st], color=col[mdl], marker="o", ms=3, label=f"Phase {mdl} · {st}")
        ax[k].set(xscale="log", xlabel="pairs asked", ylabel="pairwise acc, gap ≥ 10 (median over raters)",
                  title=f"{grp} raters")
        ax[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "active.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["C", "B"])
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--emb", type=Path, default=Path("embeddings"))
    ap.add_argument("--phase-b", type=Path, default=Path("outputs/phase_b"))
    ap.add_argument("--phase-c", type=Path, default=Path("outputs/phase_c"))
    ap.add_argument("--out", type=Path, default=Path("outputs/active"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds-run", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-val-imgs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    _, d = load_ratings(a.data)
    image_ids = np.sort(d["image_id"].unique())
    E, pos = load_emb(a.emb, "clip", image_ids)
    ctx = C.setup(a, d)
    if not a.report_only:
        Eall = torch.tensor(E, device=DEV, dtype=torch.float64)
        if "C" in a.models:
            run_C(a, ctx, pos, Eall)
            report(a, ctx[0])
        if "B" in a.models:
            run_B(a, ctx, d, pos, E)
    report(a, ctx[0])
    log("done")


if __name__ == "__main__":
    main()
