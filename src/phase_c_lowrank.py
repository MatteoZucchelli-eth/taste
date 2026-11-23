"""Phase C: shared low-rank taste model + few-shot personalisation (CONTEXT "Phase C decisions").

Model (LoRe 2025 / PAL style, hierarchical):
    s_u(x) = z_u . U(x) + w_u . f(x)
  f    = frozen CLIP-L/14 embedding (cached)
  U    = shared MLP f -> R^(K+1) (one axis can play "consensus"; K extra taste axes), trained jointly with a
         per-rater weight table Z on pooled within-rater BT pairs of the *training* raters.
  z_u  = K+1 weights of a new person, MAP under the population prior N(mu, Sigma) estimated from the training
         raters' Z (Bayesian: few pairs -> stays near mu = consensus; many pairs -> moves freely).
  w_u  = free residual on f with an L2 prior (lambda_w = inf -> basis only).
  Pairwise only: scores are used solely to build within-rater pairs (exact ties dropped).

Raters: the 18 Phase B raters (same 5 folds, same test images, so results compare pair-for-pair with Phase B)
are never in the shared model; 40 validation raters (also never in the shared model) pick K, the prior
strength tau and lambda_w per budget. Shared model per fold: all other raters, without any S/V test image.

Stages:
  ksweep  fold 0: train the shared model for each K, evaluate few-shot on validation raters -> K*
  test    5 folds with K*: few-shot curves on the 18 raters (random pairs, Phase B budgets/seeds), criterion 2
          (cross critics), best-of-n, comparison with Phase B.

Usage: python src/phase_c_lowrank.py --stage all
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
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from phase_b_critics import all_pairs, load_emb, load_ratings, pairacc  # noqa: E402

DEV = "cuda"
BUDGETS = [25, 50, 100, 250, 500, 1000, 2500, None]
KS = [0, 1, 2, 4, 8, 16, 32, 64]
TAUS = [0.5, 2.0, 8.0, 32.0]
LAMW = [0.3, 3.0, 30.0, 300.0, 3000.0, np.inf]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def metrics(y, s):
    dy = y[:, None] - y[None, :]
    ds = s[:, None] - s[None, :]
    m = np.triu(dy != 0, 1)
    prod = np.sign(dy[m]) * np.sign(ds[m])
    hit = (prod > 0) + 0.5 * (prod == 0)
    gap = np.abs(dy[m])
    out = dict(acc=hit.mean(), n_test_pairs=int(len(hit)), acc_g10=hit[gap >= 10].mean(), n_g10=int((gap >= 10).sum()))
    for lo, hi in [(0, 10), (10, 30), (30, 101)]:
        b = (gap >= lo) & (gap < hi)
        out[f"acc_gap{lo}_{hi}"] = hit[b].mean() if b.any() else np.nan
    return out


# ---------------------------------------------------------------- shared model
class Shared(nn.Module):
    def __init__(self, dim, K, n_raters):
        super().__init__()
        self.trunk = nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.2),
                                   nn.Linear(512, K + 1))
        Z = torch.zeros(n_raters, K + 1)
        Z[:, 0] = 1.0
        Z[:, 1:] = 0.01 * torch.randn(n_raters, K)
        self.Z = nn.Parameter(Z)


class RaterPairSampler:
    def __init__(self, d, pos, ridx):
        d = d.sort_values("participant_id")
        self.img = torch.tensor(pos[d["image_id"]].to_numpy(), device=DEV)
        self.r = torch.tensor(ridx[d["participant_id"]].to_numpy(), device=DEV)
        self.y = torch.tensor(d["rating"].to_numpy(np.float32), device=DEV)
        sizes = d.groupby("participant_id", sort=True).size().to_numpy()
        self.start = torch.tensor(np.repeat(np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes), device=DEV)
        self.len = torch.tensor(np.repeat(sizes, sizes), device=DEV)
        self.n = len(d)

    def sample(self, b, gen):
        k = torch.randint(self.n, (b,), device=DEV, generator=gen)
        j = self.start[k] + (torch.rand(b, device=DEV, generator=gen) * self.len[k]).long()
        keep = self.y[k] != self.y[j]
        return self.img[k][keep], self.img[j][keep], self.r[k][keep], (self.y[k] > self.y[j])[keep].float()


def monitor_acc(model, Et, d_mon, pos, ridx):
    model.eval()
    with torch.no_grad():
        S = (model.trunk(Et) @ model.Z.T).cpu().numpy()
    model.train()
    accs = []
    for pid, g in d_mon.groupby("participant_id"):
        y = g["rating"].to_numpy(float)
        if len(np.unique(y)) > 1:
            accs.append(pairacc(y, S[pos[g["image_id"]].to_numpy(), ridx[pid]]))
    return float(np.mean(accs))


def train_shared(E, d_tr, pos, K, steps, seed, batch=16384, eval_every=500, return_model=False):
    """Returns U (N x K+1, float64 tensor), Z (R x K+1), monitor history. 5% of images held out for early stopping.
    return_model=True also returns the trained module (its trunk scores new images)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rng = np.random.default_rng(seed)
    imgs = d_tr["image_id"].unique()
    mon_imgs = set(rng.choice(imgs, int(0.05 * len(imgs)), replace=False))
    d_mon = d_tr[d_tr.image_id.isin(mon_imgs)]
    d_fit = d_tr[~d_tr.image_id.isin(mon_imgs)]
    raters = np.sort(d_tr["participant_id"].unique())
    ridx = pd.Series(np.arange(len(raters)), index=raters)
    Et = torch.tensor(E, device=DEV)
    model = Shared(E.shape[1], K, len(raters)).to(DEV)
    opt = torch.optim.AdamW([dict(params=model.trunk.parameters(), weight_decay=1e-2),
                             dict(params=[model.Z], weight_decay=0.0)], lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    sampler = RaterPairSampler(d_fit, pos, ridx)
    best, best_state, hist = -1, None, []
    for step in range(1, steps + 1):
        i, j, r, t = sampler.sample(batch, gen)
        z = model.Z[r]
        logit = ((model.trunk(Et[i]) - model.trunk(Et[j])) * z).sum(1)
        loss = F.binary_cross_entropy_with_logits(logit, t) + 1e-3 * (model.Z ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            acc = monitor_acc(model, Et, d_mon, pos, ridx)
            hist.append(dict(step=step, loss=loss.item(), mon_acc=acc))
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        U = model.trunk(Et).double()
    if return_model:
        return U, model.Z.detach().double(), hist, best, model
    return U, model.Z.detach().double(), hist, best


def population_prior(Z):
    mu = Z.mean(0)
    Sig = torch.cov(Z.T).reshape(Z.shape[1], Z.shape[1]) + 1e-4 * torch.eye(Z.shape[1], device=DEV, dtype=Z.dtype)
    return mu, torch.linalg.inv(Sig)


# ---------------------------------------------------------------- personalisation (batched MAP over many problems)
def fit_users(U, E, mu, Sinv, probs, tau, lam_w, max_iter=300):
    """probs: list of (pi, pj, t) with global image indices. Returns Z (P x K+1), W (P x D). Problems are
    independent and convex, so one LBFGS on the summed loss solves them all."""
    P = len(probs)
    b = torch.tensor(np.concatenate([np.full(len(p[0]), k) for k, p in enumerate(probs)]), device=DEV)
    pi = torch.tensor(np.concatenate([p[0] for p in probs]), device=DEV)
    pj = torch.tensor(np.concatenate([p[1] for p in probs]), device=DEV)
    t = torch.tensor(np.concatenate([p[2] for p in probs]), device=DEV, dtype=torch.float64)
    use_w = bool(np.isfinite(lam_w))
    Zb = mu.repeat(P, 1).clone().requires_grad_(True)
    Wb = torch.zeros(P, E.shape[1], device=DEV, dtype=torch.float64, requires_grad=use_w)
    params = [Zb] + ([Wb] if use_w else [])
    opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, tolerance_grad=1e-6, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        S = U @ Zb.T
        if use_w:
            S = S + E @ Wb.T
        logit = S[pi, b] - S[pj, b]
        dz = Zb - mu
        loss = F.binary_cross_entropy_with_logits(logit, t, reduction="sum") + 0.5 * tau * ((dz @ Sinv) * dz).sum()
        if use_w:
            loss = loss + lam_w * (Wb ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return Zb.detach(), Wb.detach()


def sample_pairs(I, J, T, budget, rng):
    if budget is None or budget >= len(I):
        return I, J, T.astype(float)
    sub = rng.choice(len(I), budget, replace=False)
    return I[sub], J[sub], T[sub].astype(float)


# ---------------------------------------------------------------- setup shared between stages
def setup(a, d):
    S = pd.read_csv(a.phase_b / "selected.csv")
    ids = S["participant_id"].tolist()
    rng = np.random.default_rng(a.seed)  # identical to Phase B -> identical folds
    rd = {pid: d[d.participant_id == pid].reset_index(drop=True) for pid in ids}
    fold_of = {pid: rng.permutation(len(g)) % a.folds for pid, g in rd.items()}
    cnt = d.groupby("participant_id").size()
    pool = np.sort(cnt[(cnt >= 300) & ~cnt.index.isin(ids)].index.to_numpy())
    rv = np.random.default_rng(a.seed + 1)
    vids = sorted(rv.choice(pool, a.n_val, replace=False).tolist())
    vd = {pid: d[d.participant_id == pid].reset_index(drop=True) for pid in vids}
    # one shared pool of held-out images for all val raters (per-rater 20% splits would remove ~1/3 of all images
    # from the shared model's training set)
    v_test_imgs = set(rv.choice(np.sort(d["image_id"].unique()), a.n_val_imgs, replace=False).tolist())
    vtest = {pid: g["image_id"].isin(v_test_imgs).to_numpy() for pid, g in vd.items()}
    return S, ids, rd, fold_of, vids, vd, vtest, v_test_imgs


def shared_train_set(d, ids, vids, rd, fold_of, f, v_test_imgs):
    s_test = set().union(*[set(rd[p].loc[fold_of[p] == f, "image_id"]) for p in ids])
    return d[~d.participant_id.isin(ids + vids) & ~d.image_id.isin(s_test | v_test_imgs)]


def val_fewshot(U, Eall, mu, Sinv, vids, vd, vtest, pos, seeds, rng):
    """Few-shot on validation raters for every (budget, tau, lam_w). Returns DataFrame of mean acc."""
    data = {}
    for pid in vids:
        g, te = vd[pid], vtest[pid]
        p_tr = pos[g.loc[~te, "image_id"]].to_numpy()
        I, J, T = all_pairs(g.loc[~te, "rating"].to_numpy(float))
        data[pid] = (p_tr, I, J, T, pos[g.loc[te, "image_id"]].to_numpy(), g.loc[te, "rating"].to_numpy(float))
    rows = []
    for budget in BUDGETS:
        probs, keys = [], []
        for pid in vids:
            p_tr, I, J, T, _, _ = data[pid]
            for s in range(1 if budget is None else seeds):
                i, j, t = sample_pairs(I, J, T, budget, rng)
                probs.append((p_tr[i], p_tr[j], t))
                keys.append(pid)
        for tau in TAUS:
            for lam in LAMW:
                Zb, Wb = fit_users(U, Eall, mu, Sinv, probs, tau, lam)
                Sc = (U @ Zb.T + Eall @ Wb.T).cpu().numpy()
                accs = [metrics(data[pid][5], Sc[data[pid][4], k]) for k, pid in enumerate(keys)]
                rows.append(dict(budget=budget or -1, tau=tau, lam_w=lam, acc=np.mean([m["acc"] for m in accs]),
                                 acc_g10=np.mean([m["acc_g10"] for m in accs])))
        # consensus-only reference: z = mu
        sc = (U @ mu).cpu().numpy()
        cons = np.mean([metrics(data[p][5], sc[data[p][4]])["acc"] for p in vids])
        rows.append(dict(budget=budget or -1, tau=np.nan, lam_w=np.nan, acc=cons, acc_g10=np.nan, consensus=True))
    return pd.DataFrame(rows)


def best_cfg(V):
    V = V[V.tau.notna()]
    full = V.loc[V.groupby("budget")["acc"].idxmax()].set_index("budget")[["tau", "lam_w", "acc"]]
    zo = V[np.isinf(V.lam_w)]
    zonly = zo.loc[zo.groupby("budget")["acc"].idxmax()].set_index("budget")[["tau", "acc"]]
    return full, zonly


# ---------------------------------------------------------------- stages
def stage_ksweep(a, d, E, pos, ctx):
    S, ids, rd, fold_of, vids, vd, vtest, v_test_imgs = ctx
    d_tr = shared_train_set(d, ids, vids, rd, fold_of, 0, v_test_imgs)
    log(f"ksweep: shared model on {d_tr.participant_id.nunique()} raters, {len(d_tr)} ratings; {len(vids)} val raters")
    Eall = torch.tensor(E, device=DEV, dtype=torch.float64)
    rows, allv, hists = [], [], []
    for K in KS:
        U, Z, hist, mon = train_shared(E, d_tr, pos, K, a.steps, a.seed)
        hists.append(pd.DataFrame(hist).assign(K=K))
        pd.concat(hists).to_csv(a.out / "ksweep_train_curves.csv", index=False)
        mu, Sinv = population_prior(Z)
        V = val_fewshot(U, Eall, mu, Sinv, vids, vd, vtest, pos, a.val_seeds, np.random.default_rng(a.seed))
        full, zonly = best_cfg(V)
        cons = V[V.get("consensus", False) == True]["acc"].mean()  # noqa: E712
        allv.append(V.assign(K=K))
        r = dict(K=K, monitor_acc=mon, val_consensus=cons, **{f"val_b{b}": full.loc[b, "acc"] for b in full.index},
                 **{f"val_zonly_b{b}": zonly.loc[b, "acc"] for b in zonly.index})
        sel_b = [b for b in full.index if b in (25, 50, 100, 250, 500, 1000)]
        r["score"] = float(full.loc[sel_b, "acc"].mean())
        rows.append(r)
        log(f"K={K:2d} monitor {mon:.4f} | val cons {cons:.4f} | " +
            " ".join(f"{b}:{full.loc[b, 'acc']:.3f}" for b in full.index) + f" | score {r['score']:.4f}")
        pd.DataFrame(rows).to_csv(a.out / "ksweep.csv", index=False)
        pd.concat(allv).to_csv(a.out / "ksweep_val_grid.csv", index=False)
    R = pd.DataFrame(rows)
    K_star = int(R.loc[R["score"].idxmax(), "K"])
    log(f"K* = {K_star}")
    (a.out / "kstar.json").write_text(json.dumps(dict(K=K_star), indent=2))
    return K_star


def stage_test(a, d, E, pos, ctx, K):
    S, ids, rd, fold_of, vids, vd, vtest, v_test_imgs = ctx
    Eall = torch.tensor(E, device=DEV, dtype=torch.float64)
    rng = np.random.default_rng(a.seed + 7)
    curve, cross, bon, cfgs = [], [], [], []
    ck = Path("checkpoints/phase_c")
    ck.mkdir(parents=True, exist_ok=True)
    for f in range(a.folds):
        d_tr = shared_train_set(d, ids, vids, rd, fold_of, f, v_test_imgs)
        U, Z, hist, mon = train_shared(E, d_tr, pos, K, a.steps, a.seed + f)
        mu, Sinv = population_prior(Z)
        torch.save(dict(U=U.cpu(), Z=Z.cpu(), hist=hist), ck / f"shared_K{K}_fold{f}.pt")
        V = val_fewshot(U, Eall, mu, Sinv, vids, vd, vtest, pos, a.val_seeds, np.random.default_rng(a.seed + f))
        full, zonly = best_cfg(V)
        cfgs.append(pd.concat([full.add_prefix("full_"), zonly.add_prefix("zonly_")], axis=1).assign(fold=f))
        log(f"fold {f}: shared on {len(d_tr)} ratings, monitor {mon:.4f}; cfg per budget:\n"
            f"{pd.concat([full, zonly.add_prefix('z_')], axis=1).round(4).to_string()}")
        prep = {}
        for pid in ids:
            g = rd[pid]
            tr, te = fold_of[pid] != f, fold_of[pid] == f
            p_tr = pos[g.loc[tr, "image_id"]].to_numpy()
            I, J, T = all_pairs(g.loc[tr, "rating"].to_numpy(float))
            prep[pid] = (p_tr, I, J, T, pos[g.loc[te, "image_id"]].to_numpy(), g.loc[te, "rating"].to_numpy(float))
        sc_cons = (U @ mu).cpu().numpy()
        full_critics = {}
        for budget in BUDGETS:
            probs, keys = [], []
            for pid in ids:
                p_tr, I, J, T, _, _ = prep[pid]
                for s in range(1 if budget is None else a.seeds):
                    i, j, t = sample_pairs(I, J, T, budget, rng)
                    probs.append((p_tr[i], p_tr[j], t))
                    keys.append((pid, s, len(i)))
            bk = budget or -1
            for variant, tau, lam in [("lowrank+res", full.loc[bk, "tau"], full.loc[bk, "lam_w"]),
                                      ("lowrank", zonly.loc[bk, "tau"], np.inf)]:
                Zb, Wb = fit_users(U, Eall, mu, Sinv, probs, tau, lam)
                Sc = (U @ Zb.T + Eall @ Wb.T).cpu().numpy()
                for k, (pid, s, n) in enumerate(keys):
                    p_te, y_te = prep[pid][4], prep[pid][5]
                    m = metrics(y_te, Sc[p_te, k])
                    mc = metrics(y_te, sc_cons[p_te])
                    curve.append(dict(participant_id=pid, fold=f, budget=bk, n_train_pairs=n, seed=s, variant=variant,
                                      tau=tau, lam_w=lam, **m, **{"cons_" + q: v for q, v in mc.items()}))
                    if budget is None and variant == "lowrank+res":
                        full_critics[pid] = Sc[:, k]
        for pa in ids:
            p_te, y_te = prep[pa][4], prep[pa][5]
            for pb in ids:
                cross.append(dict(fold=f, rater=pa, critic=pb, acc=pairacc(y_te, full_critics[pb][p_te])))
            g = rd[pa]
            z = (y_te - g["rating"].mean()) / g["rating"].std()
            k = max(5, int(round(0.1 * len(y_te))))
            s_te = full_critics[pa][p_te]
            bon.append(dict(participant_id=pa, fold=f, personal=z[np.argsort(-s_te)[:k]].mean(),
                            consensus=z[np.argsort(-sc_cons[p_te])[:k]].mean(), oracle=np.sort(z)[::-1][:k].mean()))
        pd.DataFrame(curve).to_csv(a.out / "curve.csv", index=False)
        log(f"fold {f} done")
    C = pd.DataFrame(curve)
    X = pd.DataFrame(cross)
    B = pd.DataFrame(bon)
    C.to_csv(a.out / "curve.csv", index=False)
    X.to_csv(a.out / "cross_critics.csv", index=False)
    B.to_csv(a.out / "best_of_n.csv", index=False)
    pd.concat(cfgs).to_csv(a.out / "chosen_cfg.csv")
    return report(a, S, C, X, B, K)


def phase_b_curve(a):
    P = pd.read_csv(a.phase_b / "personal_curve.csv")
    n10, n30 = P["n_gap10_30"], P["n_gap30_101"]
    P["acc_g10"] = (P["acc_gap10_30"] * n10 + P["acc_gap30_101"] * n30) / (n10 + n30)
    P["cons_acc_g10"] = (P["cons_acc_gap10_30"] * n10 + P["cons_acc_gap30_101"] * n30) / (n10 + n30)
    return P


def report(a, S, C, X, B, K):
    P = phase_b_curve(a)
    # sanity: identical test sets as Phase B
    chk = (C[C.variant == "lowrank"].groupby(["participant_id", "fold"])["n_test_pairs"].first()
           .sub(P.groupby(["participant_id", "fold"])["n_test_pairs"].first()).abs().max())
    assert chk == 0, f"test sets differ from Phase B ({chk})"
    grp = S.set_index("participant_id")["group"]
    agg = lambda df, cols: df.groupby(["participant_id", "budget"])[cols].mean()  # noqa: E731
    pb = agg(P, ["acc", "acc_g10", "cons_acc", "cons_acc_g10"]).add_prefix("B_")
    rows = [pb]
    for v in ["lowrank+res", "lowrank"]:
        rows.append(agg(C[C.variant == v], ["acc", "acc_g10"]).add_prefix(f"C_{v}_"))
    rows.append(agg(C[C.variant == "lowrank"], ["cons_acc", "cons_acc_g10"]).add_prefix("C_"))
    T = pd.concat(rows, axis=1).reset_index()
    T["group"] = T["participant_id"].map(grp)
    T.to_csv(a.out / "per_rater_budget.csv", index=False)

    cols = ["B_cons_acc_g10", "B_acc_g10", "C_cons_acc_g10", "C_lowrank_acc_g10", "C_lowrank+res_acc_g10",
            "B_acc", "C_lowrank+res_acc"]
    med = T.groupby(["group", "budget"])[cols].median()
    log("median over raters (headline = pairs with gap >= 10):\n" + med.round(4).to_string())

    # criterion summary at full data
    full = T[T.budget == -1].set_index("participant_id")
    xr = X.groupby(["rater", "critic"])["acc"].mean().unstack()
    rank = {p: int((xr.loc[p] > xr.loc[p, p]).sum() + 1) for p in xr.index}
    b = B.groupby("participant_id")[["personal", "consensus", "oracle"]].mean()
    V = full.assign(rank_own=pd.Series(rank), bon_personal=b["personal"], bon_consensus=b["consensus"],
                    bon_oracle=b["oracle"])
    V.to_csv(a.out / "verdict.csv")
    summary = dict(K=K, steps=a.steps, folds=a.folds, seeds=a.seeds, n_val_raters=a.n_val, taus=TAUS,
                   lam_w=[str(x) for x in LAMW], runtime_s=round(time.time() - T0),
                   n_own_critic_rank1=int(sum(r == 1 for r in rank.values())))
    for (g_, bud), r in med.iterrows():
        summary.setdefault(g_, {})[str(bud)] = {k: float(v) for k, v in r.items()}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    plots(a, T)
    return T


def plots(a, T):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k, grp in enumerate(["selected", "control"]):
        g = T[T.group == grp].copy()
        g["x"] = g["budget"].replace(-1, 1e5)
        m = g.groupby("x").median(numeric_only=True)
        ax[k].plot(m.index, m["B_cons_acc_g10"], "k--", label="consensus (Phase B)")
        ax[k].plot(m.index, m["B_acc_g10"], "-o", color="0.5", label="Phase B: per-rater linear")
        ax[k].plot(m.index, m["C_lowrank_acc_g10"], "-s", color="C0", label="Phase C: low-rank only")
        ax[k].plot(m.index, m["C_lowrank+res_acc_g10"], "-o", color="C3", label="Phase C: low-rank + residual")
        ax[k].set(xscale="log", xlabel="training pairs from the rater (rightmost = all)",
                  ylabel="pairwise acc, gap ≥ 10 (median over raters)", title=f"{grp} raters")
        ax[k].legend(fontsize=8)
    T = T.assign(d=100 * (T["C_lowrank+res_acc_g10"] - T["B_acc_g10"]), x=T["budget"].replace(-1, 1e5))
    for pid, g in T.groupby("participant_id"):
        g = g.sort_values("x")
        ax[2].plot(g["x"], g["d"], "-" if g["group"].iloc[0] == "selected" else ":",
                   color="C3" if g["group"].iloc[0] == "selected" else "0.6", lw=1)
    ax[2].axhline(0, color="k", lw=.5)
    ax[2].set(xscale="log", xlabel="training pairs", ylabel="Phase C − Phase B (acc pts, gap ≥ 10)",
              title="Per rater (red = selected, grey = controls)")
    fig.tight_layout()
    fig.savefig(a.out / "phase_c.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ksweep", "test", "all"], default="all")
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--emb", type=Path, default=Path("embeddings"))
    ap.add_argument("--encoder", default="clip")
    ap.add_argument("--phase-b", type=Path, default=Path("outputs/phase_b"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_c"))
    ap.add_argument("--K", type=int, default=None, help="skip the sweep and use this K")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--val-seeds", type=int, default=2)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-val-imgs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    _, d = load_ratings(a.data)
    image_ids = np.sort(d["image_id"].unique())
    E, pos = load_emb(a.emb, a.encoder, image_ids)
    ctx = setup(a, d)
    log(f"{len(d)} ratings; test raters {len(ctx[1])}; val raters {len(ctx[4])}")
    K = a.K
    if a.stage in ("ksweep", "all") and K is None:
        K = stage_ksweep(a, d, E, pos, ctx)
    elif K is None:
        K = json.loads((a.out / "kstar.json").read_text())["K"]
    if a.stage in ("test", "all"):
        stage_test(a, d, E, pos, ctx, K)
    log("done")


if __name__ == "__main__":
    main()
