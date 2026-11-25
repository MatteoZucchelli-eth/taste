"""Phase B for text: pairwise critics on frozen text embeddings, as src/phase_b_critics.py does for images. Data: LiteraryTaste.

LiteraryTaste: 20 sets x 3 raters x 100 pairs of ~140-word snippets; each text in exactly one pair; "not sure" dropped.
  embed      frozen BAAI/bge-large-en-v1.5 (CLS pooling, L2-normalised) on all 4,000 texts.
  consensus  per evaluated set s: linear Bradley-Terry critic on embedding differences, trained on the answers of the OTHER 19
             sets only (never sees set s or its texts); L2 chosen by CV grouped by set.
  personal   per rater A: 5 folds over the set's 100 pairs (shared by the set's 3 raters). score_A = beta*consensus + w.emb (Ren 2017,
             as Phase B), BT on A's train answers, L2 on w by inner CV over pairs (incl. inf = consensus only). Pair budgets.
  criteria   (1) personal vs consensus choice accuracy on A's held-out pairs (bootstrap CI per rater; median over raters with CI);
             (2) A's critic vs the other 59 critics on A's held-out pairs (each critic's beta, w on A's consensus), and vs A's two
             set-mates (same pairs). Best-of-n is not applicable (choices only). Reference: the other 2 raters' votes.

Usage: python src/phase_b_text.py [--data data/literarytaste --out outputs/phase_b_text]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import phase_b_critics as pb  # noqa: E402
from phase_a_text import load  # noqa: E402

DEV = pb.DEV
BUDGETS = [25, 50, None]
CONS_LAMS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


@torch.no_grad()
def embed(texts, model_name, batch=32):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(DEV).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        out.append(F.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def fit_logreg(D, t, lam):
    """L2 logistic regression without intercept on difference vectors D (symmetric BT)."""
    w = torch.zeros(D.shape[1], device=DEV, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=500, tolerance_grad=1e-8, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(D @ w, t) + lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def train_consensus(Et, A, sets_train, rng):
    """A: answers with columns i1, i2 (text rows), y, set. Returns w and chosen lambda (CV grouped by set)."""
    tr = A[A.set.isin(sets_train)]
    groups = np.array_split(rng.permutation(sets_train), 4)
    D = Et[tr.i1.to_numpy()] - Et[tr.i2.to_numpy()]
    t = torch.tensor(tr.y.to_numpy(), device=DEV, dtype=torch.float64)
    losses = []
    for lam in CONS_LAMS:
        L = 0.0
        for g in groups:
            te = torch.tensor(tr.set.isin(g).to_numpy(), device=DEV)
            w = fit_logreg(D[~te], t[~te], lam)
            L += float(F.binary_cross_entropy_with_logits(D[te] @ w, t[te], reduction="sum"))
        losses.append(L)
    lam = CONS_LAMS[int(np.argmin(losses))]
    return fit_logreg(D, t, lam), lam


def fit_personal_cv(sc, Et, pi, pj, t, gen, folds=3):
    """lambda by K-fold CV over pairs (each text is in one pair only, so splitting pairs cannot leak items)."""
    n = len(t)
    fid = torch.randperm(n, device=DEV, generator=gen) % folds
    losses = []
    for lam in pb.LAMBDAS:
        L = 0.0
        for k in range(folds):
            tr, te = fid != k, fid == k
            th = pb.fit_personal(sc, Et, pi[tr], pj[tr], t[tr], lam)
            L += pb.pair_logloss(th, sc, Et, pi[te], pj[te], t[te]) * int(te.sum())
        losses.append(L)
    lam = pb.LAMBDAS[int(np.argmin(losses))]
    return pb.fit_personal(sc, Et, pi, pj, t, lam), lam


def choice_acc(y, d):
    """y in {0,1}; d = score(text1) - score(text2); ties count 0.5."""
    return float(np.mean(np.where(d == 0, 0.5, (d > 0) == (y > 0.5))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/literarytaste"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_b_text"))
    ap.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)

    t, ans = load(a.data)
    n = len(t)
    row = pd.Series(np.arange(n), index=t["id"].to_numpy())  # text1 of pair id -> row; text2 -> row + n
    cache = Path("embeddings") / f"literarytaste_{a.encoder.split('/')[-1]}.npz"
    if not cache.exists():
        texts = t["text1"].fillna("").tolist() + t["text2"].fillna("").tolist()
        np.savez(cache, ids=t["id"].to_numpy(), emb=embed(texts, a.encoder).astype(np.float32))
    z = np.load(cache)
    assert (z["ids"] == t["id"].to_numpy()).all()
    Et = torch.tensor(z["emb"], device=DEV, dtype=torch.float64)
    A = ans[ans.y.notna()].copy()
    A["set"] = A.iid.map(t.set_index("id")["set_id"])
    A["i1"] = row[A.iid].to_numpy()
    A["i2"] = A["i1"] + n
    votes = A.groupby("iid")["y"].apply(list)
    users_of = A.groupby("iid")["user"].apply(list)
    A["others"] = [np.mean([v for v, w in zip(votes[i], users_of[i]) if w != u] or [0.5]) for u, i in zip(A.user, A.iid)]
    log(f"{len(A)} answers, {A.user.nunique()} raters, {t.set_id.nunique()} sets, emb {tuple(Et.shape)} ({a.encoder})")

    all_sets = np.sort(t.set_id.unique())
    pair_fold = pd.Series(0, index=t["id"].to_numpy())
    for s in all_sets:
        ids = t.loc[t.set_id == s, "id"].to_numpy()
        pair_fold[ids] = rng.permutation(len(ids)) % a.folds

    cons, cons_info = {}, []
    for s in all_sets:
        w, lam = train_consensus(Et, A, [x for x in all_sets if x != s], rng)
        sc = Et @ w
        cons[s] = (sc - sc.mean()) / sc.std()
        As = A[A.set == s]
        d = (cons[s][As.i1.to_numpy()] - cons[s][As.i2.to_numpy()]).cpu().numpy()
        cons_info.append(dict(set=int(s), lam=lam, acc=choice_acc(As.y.to_numpy(), d)))
    CI = pd.DataFrame(cons_info)
    log(f"consensus critics: held-out-set accuracy mean {CI.acc.mean():.3f} (per set {CI.acc.min():.2f}-{CI.acc.max():.2f}), "
        f"lambdas {CI.lam.value_counts().to_dict()}")

    raters = A.groupby("user")
    curve, crit, test_rows = [], {}, {}
    for u, g in raters:
        s = int(g.set.iloc[0])
        sc = cons[s]
        f_of = pair_fold[g.iid].to_numpy()
        for f in range(a.folds):
            tr, te = g[f_of != f], g[f_of == f]
            test_rows[(u, f)] = te
            pi, pj = torch.tensor(tr.i1.to_numpy(), device=DEV), torch.tensor(tr.i2.to_numpy(), device=DEV)
            ty = torch.tensor(tr.y.to_numpy(), device=DEV, dtype=torch.float64)
            d_cons = (sc[te.i1.to_numpy()] - sc[te.i2.to_numpy()]).cpu().numpy()
            acc_c = choice_acc(te.y.to_numpy(), d_cons)
            for budget in BUDGETS:
                full = budget is None or budget >= len(tr)
                for sd in ([0] if full else range(a.seeds)):
                    sub = torch.arange(len(tr), device=DEV) if full else \
                        torch.tensor(rng.choice(len(tr), budget, replace=False), device=DEV)
                    th, lam = fit_personal_cv(sc, Et, pi[sub], pj[sub], ty[sub], gen)
                    s_all = th[0] * sc + Et @ th[1:]
                    d = (s_all[te.i1.to_numpy()] - s_all[te.i2.to_numpy()]).cpu().numpy()
                    curve.append(dict(user=u, set=s, fold=f, budget=budget or -1, n_train=len(sub), seed=sd, lam=lam,
                                      beta=float(th[0]), acc=choice_acc(te.y.to_numpy(), d), cons_acc=acc_c,
                                      n_test=len(te)))
                    if budget is None:
                        crit[(u, f)] = th
        log(f"rater {u} (set {s}) done")
    C = pd.DataFrame(curve)
    C.to_csv(a.out / "personal_curve.csv", index=False)

    # criterion 2: every critic (its beta, w from the same fold index) on A's held-out pairs, on A's consensus
    users = list(raters.groups)
    set_of = A.groupby("user")["set"].first()
    W = {f: torch.stack([crit[(v, f)] for v in users]) for f in range(a.folds)}
    cross = []
    for u in users:
        s = set_of[u]
        sc = cons[s]
        for f in range(a.folds):
            te = test_rows[(u, f)]
            if len(te) == 0:
                continue
            S_all = W[f][:, :1] * sc[None] + W[f][:, 1:] @ Et.T
            d = (S_all[:, te.i1.to_numpy()] - S_all[:, te.i2.to_numpy()]).cpu().numpy()
            y = te.y.to_numpy()
            hits = np.where(d == 0, 0.5, (d > 0) == (y > 0.5))
            for v, h in zip(users, hits):
                cross.append(dict(rater=u, critic=v, fold=f, hits=h.sum(), n=len(y)))
    X = pd.DataFrame(cross).groupby(["rater", "critic"])[["hits", "n"]].sum().reset_index()
    X["acc"] = X.hits / X.n
    X.to_csv(a.out / "cross_critics.csv", index=False)

    full = C[C.budget == -1]
    rows = []
    for u in users:
        fr = full[full.user == u]
        hits_p, hits_c = [], []
        for f in range(a.folds):
            te = test_rows[(u, f)]
            th = crit[(u, f)]
            sc = cons[set_of[u]]
            s_all = th[0] * sc + Et @ th[1:]
            dp = (s_all[te.i1.to_numpy()] - s_all[te.i2.to_numpy()]).cpu().numpy()
            dc = (sc[te.i1.to_numpy()] - sc[te.i2.to_numpy()]).cpu().numpy()
            y = te.y.to_numpy()
            hits_p += list(np.where(dp == 0, 0.5, (dp > 0) == (y > 0.5)))
            hits_c += list(np.where(dc == 0, 0.5, (dc > 0) == (y > 0.5)))
        hp, hc = np.array(hits_p, float), np.array(hits_c, float)
        bs = [(hp[i] - hc[i]).mean() for i in (rng.integers(0, len(hp), len(hp)) for _ in range(a.boot))]
        xr = X[X.rater == u].set_index("critic")["acc"]
        mates = [v for v in users if set_of[v] == set_of[u] and v != u]
        g = A[A.user == u]
        rows.append(dict(user=u, set=int(set_of[u]), n=len(g), acc_personal=hp.mean(), acc_consensus=hc.mean(),
                         gain=100 * (hp.mean() - hc.mean()), gain_ci_lo=100 * np.percentile(bs, 2.5),
                         gain_ci_hi=100 * np.percentile(bs, 97.5), lam_inf_frac=float(np.isinf(fr.lam).mean()),
                         acc_other2_votes=choice_acc(g.y.to_numpy(), g.others.to_numpy() - 0.5),
                         rank_own=int((xr > xr[u]).sum() + 1), n_critics=len(xr),
                         beats_both_mates=bool(xr[u] > xr[mates].max()) if mates else None))
    V = pd.DataFrame(rows)
    V.to_csv(a.out / "verdict.csv", index=False)
    med_boot = [V.gain.sample(len(V), replace=True, random_state=i).median() for i in range(a.boot)]
    mean_boot = [V.gain.sample(len(V), replace=True, random_state=i).mean() for i in range(a.boot)]
    eff = (C.assign(gain=100 * (C.acc - C.cons_acc)).groupby("budget")
           .agg(n_train=("n_train", "mean"), gain_mean=("gain", "mean"), lam_inf=("lam", lambda s: float(np.isinf(s).mean())))
           .reset_index())
    summary = dict(
        encoder=a.encoder, n_raters=len(V), n_answers=int(len(A)), folds=a.folds,
        consensus_heldout_set_acc_mean=float(CI.acc.mean()),
        acc_consensus_median=float(V.acc_consensus.median()), acc_personal_median=float(V.acc_personal.median()),
        acc_other2_votes_median=float(V.acc_other2_votes.median()),
        c1_gain_median_pts=float(V.gain.median()), c1_gain_median_ci=np.percentile(med_boot, [2.5, 97.5]).tolist(),
        c1_gain_mean_pts=float(V.gain.mean()), c1_gain_mean_ci=np.percentile(mean_boot, [2.5, 97.5]).tolist(),
        c1_frac_raters_ci_above_0=float((V.gain_ci_lo > 0).mean()), c1_frac_gain_pos=float((V.gain > 0).mean()),
        lam_inf_frac=float(V.lam_inf_frac.mean()),
        c2_frac_own_rank1=float((V.rank_own == 1).mean()), c2_rank_own_median=float(V.rank_own.median()),
        c2_rank_own_chance=(len(V) + 1) / 2, c2_frac_beats_both_mates=float(V.beats_both_mates.mean()),
        label_efficiency=eff.to_dict("records"), runtime_s=round(time.time() - T0))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
