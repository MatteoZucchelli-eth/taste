"""Phase B: pairwise critics on frozen image embeddings (CONTEXT §6, "Phase B decisions").

Pairwise only: LAPIS scores are used solely to form within-rater pairs (exact ties dropped). Every critic is a
Bradley-Terry model and every metric is pairwise accuracy.

Stages (run in order with --stage all):
  consensus  encoder + head selection. BT head trained on within-rater pairs pooled over all raters, on the official
             giaa train images; early-stopped and selected on the giaa validation images; test reported.
  select     pick 5-10 raters with *diverse taste directions* among the Phase A raters with strong, clean taste
             (direction = metadata ridge of the rater's deviation from LOO consensus, as in Phase A), greedy
             farthest-point on cosine distance. Plus random control raters (not selected on taste) to measure the
             selection optimism. Heuristic raters (§6b caveat) are flagged, not removed.
  personal   K folds over each rater's images. Per fold, ONE consensus critic trained on all raters except the
             evaluated set S, and on images outside every S-rater's test fold (so A and A's test images never enter
             it). Personal critic (Ren 2017): score_A = beta * consensus + w . emb, BT loss on A's train pairs,
             L2 on w (lambda by inner CV, incl. lambda=inf = consensus only). Label-efficiency in number of pairs.
             Evaluates the §5 criterion: (1) vs consensus, (2) vs other raters' critics, (3) vs ceiling,
             (4) best-of-n.

Usage: python src/phase_b_critics.py --stage all [--data data/lapis --emb embeddings --out outputs/phase_b]
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
import phase_a  # noqa: E402

DEV = "cuda"
GAP_BINS = [(0, 10), (10, 30), (30, 101)]
BUDGETS = [25, 50, 100, 250, 500, 1000, 2500, None]  # None = all pairs
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- data
def load_ratings(data):
    raw = pd.read_csv(data / "piaa/piaa.csv", keep_default_na=False, na_values=[""])
    d = raw.groupby(["participant_id", "image_id"], as_index=False)["rating"].mean()  # average the few repeats
    return raw, d


def load_emb(emb_dir, name, image_ids):
    z = np.load(emb_dir / f"{name}.npz")
    pos = pd.Series(np.arange(len(z["ids"])), index=z["ids"])
    assert pd.Index(image_ids).isin(pos.index).all()
    return z["emb"], pos


def pair_metrics(y, s):
    """Pairwise accuracy of score s on all non-tied pairs of y (ties in s = 0.5), overall and by |gap|."""
    dy = y[:, None] - y[None, :]
    ds = s[:, None] - s[None, :]
    m = np.triu(dy != 0, 1)
    prod = np.sign(dy[m]) * np.sign(ds[m])
    hit = (prod > 0) + 0.5 * (prod == 0)
    gap = np.abs(dy[m])
    out = dict(acc=hit.mean() if len(hit) else np.nan, n_test_pairs=int(len(hit)))
    for lo, hi in GAP_BINS:
        b = (gap >= lo) & (gap < hi)
        out[f"acc_gap{lo}_{hi}"] = hit[b].mean() if b.any() else np.nan
        out[f"n_gap{lo}_{hi}"] = int(b.sum())
    return out


def pairacc(y, s):
    dy = y[:, None] - y[None, :]
    ds = s[:, None] - s[None, :]
    m = np.triu(dy != 0, 1)
    prod = np.sign(dy[m]) * np.sign(ds[m])
    return ((prod > 0) + 0.5 * (prod == 0)).mean()


def per_rater_acc(d, score_by_pos, pos):
    """Mean/median over raters of within-rater pairwise accuracy (raters with >= 2 distinct ratings)."""
    accs = []
    for _, g in d.groupby("participant_id"):
        y = g["rating"].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        accs.append(pairacc(y, score_by_pos[pos[g["image_id"]].to_numpy()]))
    a = np.array(accs)
    return dict(mean=float(a.mean()), median=float(np.median(a)), n_raters=len(a))


# ---------------------------------------------------------------- consensus BT critic (pooled within-rater pairs)
def make_head(kind, dim):
    if kind == "linear":
        return nn.Linear(dim, 1)
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 1))


HEAD_CFG = {"linear": dict(lr=3e-3, weight_decay=1e-4), "mlp": dict(lr=1e-3, weight_decay=1e-2)}


class PairSampler:
    """Uniform rating k, then a random partner rating from the same rater; tied pairs masked out."""

    def __init__(self, d, pos):
        d = d.sort_values("participant_id")
        self.img = torch.tensor(pos[d["image_id"]].to_numpy(), device=DEV)
        self.y = torch.tensor(d["rating"].to_numpy(np.float32), device=DEV)
        sizes = d.groupby("participant_id", sort=True).size().to_numpy()
        starts = np.repeat(np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes)
        self.start = torch.tensor(starts, device=DEV)
        self.len = torch.tensor(np.repeat(sizes, sizes), device=DEV)
        self.n = len(d)

    def sample(self, b, gen):
        k = torch.randint(self.n, (b,), device=DEV, generator=gen)
        j = self.start[k] + (torch.rand(b, device=DEV, generator=gen) * self.len[k]).long()
        yk, yj = self.y[k], self.y[j]
        keep = yk != yj
        return self.img[k][keep], self.img[j][keep], (yk > yj)[keep].float()


def train_consensus(E, d_train, pos, kind, steps, seed, d_val=None, eval_every=500, batch=16384):
    """Returns (scores for all images (z-scored), best_step, history). Early-stops on d_val if given."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Et = torch.tensor(E, device=DEV)
    head = make_head(kind, E.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), **HEAD_CFG[kind])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=HEAD_CFG[kind]["lr"], total_steps=steps, pct_start=0.05)
    sampler = PairSampler(d_train, pos)
    best, best_step, best_state, hist = -1, steps, None, []

    def score():
        head.eval()
        with torch.no_grad():
            s = head(Et).squeeze(-1).cpu().numpy().astype(np.float64)
        head.train()
        return (s - s.mean()) / s.std()

    for step in range(1, steps + 1):
        i, j, t = sampler.sample(batch, gen)
        logit = head(Et[i]).squeeze(-1) - head(Et[j]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if d_val is not None and (step % eval_every == 0 or step == steps):
            acc = per_rater_acc(d_val, score(), pos)["mean"]
            hist.append(dict(step=step, loss=loss.item(), val_acc=acc))
            if acc > best:
                best, best_step = acc, step
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    return score(), best_step, hist, head


def stage_consensus(a, d, image_ids):
    split = pd.concat([pd.read_csv(a.data / f"giaa/{s}.csv", usecols=["image_id"]).assign(split=s)
                       for s in ["train", "validation", "test"]]).set_index("image_id")["split"]
    d = d.assign(split=split.reindex(d["image_id"]).to_numpy())
    log("giaa split sizes (ratings):", d["split"].value_counts().to_dict())
    rows = []
    for enc in a.encoders:
        E, pos = load_emb(a.emb, enc, image_ids)
        for kind in ["linear", "mlp"]:
            s, best_step, hist, head = train_consensus(E, d[d.split == "train"], pos, kind, a.cons_steps, a.seed,
                                                       d_val=d[d.split == "validation"])
            r = dict(encoder=enc, head=kind, dim=E.shape[1], best_step=best_step,
                     val=per_rater_acc(d[d.split == "validation"], s, pos)["mean"],
                     test=per_rater_acc(d[d.split == "test"], s, pos)["mean"],
                     test_median=per_rater_acc(d[d.split == "test"], s, pos)["median"])
            rows.append(r)
            log(f"consensus {enc:8s} {kind:6s} val {r['val']:.4f} test {r['test']:.4f} (step {best_step})")
            ck = Path("checkpoints/phase_b")
            ck.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), ck / f"consensus_{enc}_{kind}_giaa.pt")
    R = pd.DataFrame(rows)
    R.to_csv(a.out / "consensus_select.csv", index=False)
    best = R.loc[R["val"].idxmax()]
    choice = dict(encoder=best["encoder"], head=best["head"], steps=int(best["best_step"]), table=rows)
    (a.out / "consensus_select.json").write_text(json.dumps(choice, indent=2, default=float))
    log("chosen:", choice["encoder"], choice["head"], choice["steps"])
    return choice


# ---------------------------------------------------------------- rater selection
def stage_select(a, raw, d):
    R = pd.read_csv(a.phase_a / "raters.csv")
    R["heuristic"] = (R["acc_cons"] < 0.58) | (R["n_unique"] < 20)
    cand = R[(R["sig"] == True) & (R["gain"] >= 5) & (R["median_rt_ms"] >= 1500)  # noqa: E712
             & (R["frac_at_mode"] < 0.30) & (R["n"] >= a.min_ratings)].copy()
    log(f"candidates: {len(cand)} (of which heuristic {int(cand['heuristic'].sum())})")

    # taste direction: ridge (Phase A lambda) of A's deviation from LOO consensus on metadata features
    attrs = pd.read_csv(a.data / "image_attributes/image_attributes.csv")
    X = phase_a.build_features(raw, attrs)
    g = d.groupby("image_id")["rating"]
    dd = d.assign(cons=(g.transform("sum") - d["rating"]) / (g.transform("count") - 1))
    dirs = {}
    for pid in cand["participant_id"]:
        gr = dd[dd.participant_id == pid]
        y, c = gr["rating"].to_numpy(float), gr["cons"].to_numpy(float)
        e = y - phase_a.ols_fit(c[:, None], y)(c[:, None])
        Xa = X.loc[gr["image_id"]].to_numpy()
        Xa = Xa - Xa.mean(0)
        w = np.linalg.solve(Xa.T @ Xa + 1000 * np.eye(Xa.shape[1]), Xa.T @ (e - e.mean()))
        dirs[pid] = w / np.linalg.norm(w)
    W = np.stack([dirs[p] for p in cand["participant_id"]])
    cos = W @ W.T
    order = list(cand["participant_id"])
    chosen = [int(np.argmax(cand["taste_z"].to_numpy()))]
    while len(chosen) < a.n_select:
        mind = (1 - cos[:, chosen]).min(1)
        mind[chosen] = -1
        chosen.append(int(np.argmax(mind)))
    sel = cand.iloc[chosen].copy()
    names = np.array(X.columns)
    sel["taste_likes"] = [", ".join(names[np.argsort(-dirs[p])[:4]]) for p in sel["participant_id"]]
    sel["taste_dislikes"] = [", ".join(names[np.argsort(dirs[p])[:4]]) for p in sel["participant_id"]]
    sel["min_cos_to_others"] = [float(np.max(np.delete(cos[i, chosen], k))) for k, i in enumerate(chosen)]
    sel["group"] = "selected"

    rng = np.random.default_rng(a.seed)
    pool = R[(R["n"] >= a.min_ratings) & R["p"].notna() & ~R["participant_id"].isin(sel["participant_id"])]
    ctrl = pool.iloc[rng.choice(len(pool), a.n_control, replace=False)].copy()
    ctrl["group"] = "control"
    S = pd.concat([sel, ctrl])
    cols = ["participant_id", "group", "heuristic", "n", "n_unique", "acc_cons", "gain", "taste_z", "vaiak",
            "imgs_known", "taste_likes", "taste_dislikes", "min_cos_to_others"]
    S[cols].to_csv(a.out / "selected.csv", index=False)
    log("selected:\n" + S[cols[:9]].to_string(index=False))
    pd.DataFrame(cos[np.ix_(chosen, chosen)], index=sel["participant_id"], columns=sel["participant_id"]).to_csv(
        a.out / "selected_direction_cos.csv")
    return S


# ---------------------------------------------------------------- personal critic: beta*cons + w.emb, BT, L2 on w
def fit_personal(sc, Et, pi, pj, t, lam, init=None):
    """sc: consensus score per local image (torch), Et: local embeddings, pairs (pi, pj, t). Returns (beta, w)."""
    D = Et.shape[1]
    theta = torch.zeros(D + 1, device=DEV, dtype=torch.float64)
    theta[0] = 1.0
    if init is not None:
        theta.copy_(init)
    if np.isinf(lam):
        theta[1:] = 0
    theta.requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=300, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
    dsc = sc[pi] - sc[pj]

    def closure():
        opt.zero_grad()
        w = theta[1:] if not np.isinf(lam) else theta[1:].detach() * 0
        u = Et @ w
        logit = theta[0] * dsc + u[pi] - u[pj]
        loss = F.binary_cross_entropy_with_logits(logit, t)
        if not np.isinf(lam):
            loss = loss + lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    th = theta.detach()
    if np.isinf(lam):
        th[1:] = 0
    return th


def pair_logloss(th, sc, Et, pi, pj, t):
    u = Et @ th[1:]
    logit = th[0] * (sc[pi] - sc[pj]) + u[pi] - u[pj]
    return float(F.binary_cross_entropy_with_logits(logit, t))


def fit_with_cv(sc, Et, pi, pj, t, gen, folds=3):
    """Choose lambda by inner K-fold CV *over images* (held-out BT log-loss on pairs whose two images are both
    unseen, as at test time), then refit on all pairs. Splitting over pairs would leak: a high-dim head can memorise
    per-image utilities, so held-out pairs on seen images reward under-regularisation."""
    fid_img = torch.randperm(Et.shape[0], device=DEV, generator=gen) % folds
    fi, fj = fid_img[pi], fid_img[pj]
    splits = [((fi != k) & (fj != k), (fi == k) & (fj == k)) for k in range(folds)]
    splits = [(tr, te) for tr, te in splits if tr.sum() >= 2 and te.sum() >= 1]
    if not splits:
        lam = np.inf
    else:
        losses = []
        for lam in LAMBDAS:
            L, n = 0.0, 0
            for tr, te in splits:
                th = fit_personal(sc, Et, pi[tr], pj[tr], t[tr], lam)
                L += pair_logloss(th, sc, Et, pi[te], pj[te], t[te]) * int(te.sum())
                n += int(te.sum())
            losses.append(L / n)
        lam = LAMBDAS[int(np.argmin(losses))]
    return fit_personal(sc, Et, pi, pj, t, lam), lam


def all_pairs(y):
    i, j = np.triu_indices(len(y), 1)
    keep = y[i] != y[j]
    return i[keep], j[keep], (y[i] > y[j])[keep]


def retest_ceiling(y, var_noise, rng, sims=200):
    """Noise ceiling: expected pairwise agreement between A's observed ratings and A's *true* scores (a perfect
    critic). Model y = t + e, noise variance from the 614 repeated ratings (global), reliability r = 1 - var_e/var_y,
    so t | y ~ N(m + r(y - m), r(1 - r) var_y). Approximate: uses one global noise level for every rater."""
    m, v = y.mean(), y.var()
    r = min(1.0, max(0.0, 1 - var_noise / v))
    acc = [pairacc(y, m + r * (y - m) + rng.normal(0, np.sqrt(r * (1 - r) * v), len(y))) for _ in range(sims)]
    return float(np.mean(acc)), r


def stage_personal(a, raw, d, image_ids, choice, S):
    E, pos = load_emb(a.emb, choice["encoder"], image_ids)
    Eall = torch.tensor(E, device=DEV, dtype=torch.float64)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    ids = S["participant_id"].tolist()
    # noise variance per rating from repeated (rater, image) ratings
    dup = raw[raw.duplicated(["participant_id", "image_id"], keep=False)].sort_values(["participant_id", "image_id"])
    pr = dup.groupby(["participant_id", "image_id"])["rating"]
    var_noise = float(np.var(pr.nth(0).to_numpy() - pr.nth(1).to_numpy()) / 2)

    rd = {pid: d[d.participant_id == pid].reset_index(drop=True) for pid in ids}
    fold_of = {pid: rng.permutation(len(g)) % a.folds for pid, g in rd.items()}

    curve, cross, bon, oof, critics = [], [], [], [], {}
    for f in range(a.folds):
        test_imgs = set()
        for pid in ids:
            test_imgs |= set(rd[pid].loc[fold_of[pid] == f, "image_id"])
        d_cons = d[~d.participant_id.isin(ids) & ~d.image_id.isin(test_imgs)]
        sc_all, _, _, _ = train_consensus(E, d_cons, pos, choice["head"], choice["steps"], a.seed + f)
        sc_all_t = torch.tensor(sc_all, device=DEV, dtype=torch.float64)
        log(f"fold {f}: consensus trained on {len(d_cons)} ratings ({d_cons.image_id.nunique()} images)")
        critics[f] = {}
        for pid in ids:
            g = rd[pid]
            tr, te = fold_of[pid] != f, fold_of[pid] == f
            p_tr, p_te = pos[g.loc[tr, "image_id"]].to_numpy(), pos[g.loc[te, "image_id"]].to_numpy()
            y_tr, y_te = g.loc[tr, "rating"].to_numpy(float), g.loc[te, "rating"].to_numpy(float)
            sc_tr, E_tr = sc_all_t[p_tr], Eall[p_tr]
            sc_te_np = sc_all[p_te]
            m_cons = pair_metrics(y_te, sc_te_np)
            I, J, T = all_pairs(y_tr)
            for budget in BUDGETS:
                seeds = [0] if budget is None or budget >= len(I) else range(a.seeds)
                for s in seeds:
                    if budget is None or budget >= len(I):
                        sub = np.arange(len(I))
                    else:
                        sub = rng.choice(len(I), budget, replace=False)
                    pi = torch.tensor(I[sub], device=DEV)
                    pj = torch.tensor(J[sub], device=DEV)
                    t = torch.tensor(T[sub], device=DEV, dtype=torch.float64)
                    th, lam = fit_with_cv(sc_tr, E_tr, pi, pj, t, gen)
                    s_te = (th[0] * sc_all_t[p_te] + Eall[p_te] @ th[1:]).cpu().numpy()
                    m = pair_metrics(y_te, s_te)
                    curve.append(dict(participant_id=pid, fold=f, budget=budget or -1, n_train_pairs=len(sub), seed=s,
                                      lam=lam, beta=float(th[0]), w_norm=float(th[1:].norm()),
                                      **{k: v for k, v in m.items()},
                                      **{"cons_" + k: v for k, v in m_cons.items() if k.startswith("acc")}))
                    if budget is None:
                        critics[f][pid] = th
            # out-of-fold scores + best-of-n (full-data critic)
            th = critics[f][pid]
            s_te = (th[0] * sc_all_t[p_te] + Eall[p_te] @ th[1:]).cpu().numpy()
            z = (y_te - g["rating"].mean()) / g["rating"].std()
            k = max(5, int(round(0.1 * len(y_te))))
            bon.append(dict(participant_id=pid, fold=f, k=k, personal=z[np.argsort(-s_te)[:k]].mean(),
                            consensus=z[np.argsort(-sc_te_np)[:k]].mean(), oracle=np.sort(z)[::-1][:k].mean(),
                            random=0.0))
            oof.append(pd.DataFrame(dict(participant_id=pid, fold=f, image_id=g.loc[te, "image_id"].to_numpy(),
                                         rating=y_te, personal=s_te, consensus=sc_te_np)))
        # criterion 2: every S critic (full data, fold f) on every S rater's fold-f test pairs
        for pa in ids:
            g = rd[pa]
            te = fold_of[pa] == f
            p_te = pos[g.loc[te, "image_id"]].to_numpy()
            y_te = g.loc[te, "rating"].to_numpy(float)
            for pb in ids:
                th = critics[f][pb]
                s = (th[0] * sc_all_t[p_te] + Eall[p_te] @ th[1:]).cpu().numpy()
                cross.append(dict(fold=f, rater=pa, critic=pb, acc=pairacc(y_te, s)))
        log(f"fold {f} done")
        pd.DataFrame(curve).to_csv(a.out / "personal_curve.csv", index=False)

    C = pd.DataFrame(curve)
    X = pd.DataFrame(cross)
    B = pd.DataFrame(bon)
    O = pd.concat(oof)
    C.to_csv(a.out / "personal_curve.csv", index=False)
    X.to_csv(a.out / "cross_critics.csv", index=False)
    B.to_csv(a.out / "best_of_n.csv", index=False)
    O.to_csv(a.out / "oof_scores.csv", index=False)
    torch.save({f: {int(p): th.cpu() for p, th in c.items()} for f, c in critics.items()},
               Path("checkpoints/phase_b") / "personal_critics.pt")
    return evaluate(a, S, rd, fold_of, C, X, B, O, var_noise, rng, choice)


# ---------------------------------------------------------------- §5 evaluation
def boot_gain(O, reps, rng):
    """CI of (personal - consensus) pairwise accuracy: resample each fold's test images, average over folds."""
    folds = [(g["rating"].to_numpy(), g["personal"].to_numpy(), g["consensus"].to_numpy()) for _, g in
             O.groupby("fold")]
    out = []
    for _ in range(reps):
        gs = []
        for y, p, c in folds:
            i = rng.integers(0, len(y), len(y))
            gs.append(pairacc(y[i], p[i]) - pairacc(y[i], c[i]))
        out.append(np.mean(gs))
    return np.percentile(out, [2.5, 97.5])


def evaluate(a, S, rd, fold_of, C, X, B, O, var_noise, rng, choice):
    full = C[C.budget == -1]
    rows = []
    for _, r in S.iterrows():
        pid = r["participant_id"]
        fr = full[full.participant_id == pid]
        acc_p, acc_c = fr["acc"].mean(), fr["cons_acc"].mean()
        lo, hi = boot_gain(O[O.participant_id == pid], a.boot, rng)
        xr = X[X.rater == pid].groupby("critic")["acc"].mean()
        same = S.loc[S.participant_id != pid]
        others_sel = xr.drop(pid).reindex(same.loc[same.group == r["group"], "participant_id"]).dropna()
        ceil, rel = retest_ceiling(rd[pid]["rating"].to_numpy(float), var_noise, rng)
        b = B[B.participant_id == pid][["personal", "consensus", "oracle"]].mean()
        row = dict(participant_id=pid, group=r["group"], heuristic=bool(r["heuristic"]), n=int(r["n"]),
                   acc_personal=acc_p, acc_consensus=acc_c, gain_pts=100 * (acc_p - acc_c),
                   gain_ci_lo=100 * lo, gain_ci_hi=100 * hi,
                   c1_beats_consensus=bool(lo > 0),
                   best_other_critic=int(others_sel.idxmax()) if len(others_sel) else None,
                   acc_best_other=float(others_sel.max()) if len(others_sel) else np.nan,
                   rank_own_critic=int((xr > xr[pid]).sum() + 1), n_critics=len(xr),
                   c2_beats_all_others=bool(len(others_sel) and xr[pid] > others_sel.max()),
                   ceiling_retest=ceil, reliability_est=rel,
                   frac_gap_closed=(acc_p - acc_c) / (ceil - acc_c) if ceil > acc_c else np.nan,
                   bon_personal_z=b["personal"], bon_consensus_z=b["consensus"], bon_oracle_z=b["oracle"],
                   c4_bon_better=bool(b["personal"] > b["consensus"]),
                   **{f"acc_personal_gap{lo_}_{hi_}": fr[f"acc_gap{lo_}_{hi_}"].mean() for lo_, hi_ in GAP_BINS},
                   **{f"acc_consensus_gap{lo_}_{hi_}": fr[f"cons_acc_gap{lo_}_{hi_}"].mean() for lo_, hi_ in GAP_BINS},
                   phase_a_gain=r["gain"])
        rows.append(row)
    V = pd.DataFrame(rows)
    V.to_csv(a.out / "verdict.csv", index=False)

    eff = (C.assign(gain=100 * (C["acc"] - C["cons_acc"]))
           .groupby(["participant_id", "budget"]).agg(gain=("gain", "mean"), n_pairs=("n_train_pairs", "mean"),
                                                      lam_inf=("lam", lambda s: float(np.isinf(s).mean())))
           .reset_index())
    eff = eff.merge(S[["participant_id", "group"]], on="participant_id")
    eff.to_csv(a.out / "label_efficiency.csv", index=False)

    summary = dict(encoder=choice["encoder"], head=choice["head"], consensus_steps=choice["steps"],
                   folds=a.folds, seeds=a.seeds, budgets=[b or "all" for b in BUDGETS], lambdas=LAMBDAS,
                   noise_var_per_rating=var_noise, runtime_s=round(time.time() - T0))
    for grp, g in V.groupby("group"):
        summary[grp] = dict(n=len(g), median_gain_pts=float(g["gain_pts"].median()),
                            n_c1=int(g["c1_beats_consensus"].sum()), n_c2=int(g["c2_beats_all_others"].sum()),
                            n_c4=int(g["c4_bon_better"].sum()),
                            median_frac_gap_closed=float(g["frac_gap_closed"].median()))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plots(a, V, eff, X, S)
    log("verdict:\n" + V[["participant_id", "group", "heuristic", "acc_consensus", "acc_personal", "gain_pts",
                          "gain_ci_lo", "gain_ci_hi", "rank_own_critic", "ceiling_retest", "bon_personal_z",
                          "bon_consensus_z"]].round(3).to_string(index=False))
    log(json.dumps(summary, indent=1, default=float))
    return V


def plots(a, V, eff, X, S):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for (pid, grp), g in eff.groupby(["participant_id", "group"]):
        g = g.sort_values("n_pairs")
        ax[0, 0].plot(g["n_pairs"], g["gain"], "-o" if grp == "selected" else ":", ms=3,
                      color=None if grp == "selected" else "0.6", label=str(pid) if grp == "selected" else None)
    ax[0, 0].axhline(0, color="k", lw=.5)
    ax[0, 0].set(xscale="log", xlabel="training pairs from rater A", ylabel="gain over consensus critic (acc pts)",
                 title="Label efficiency (solid = selected, dotted = controls)")
    ax[0, 0].legend(fontsize=7, ncol=2)
    sel = S[S.group == "selected"]["participant_id"].tolist()
    M = X[X.rater.isin(sel) & X.critic.isin(sel)].groupby(["rater", "critic"])["acc"].mean().unstack()
    M = M.loc[sel, sel]
    im = ax[0, 1].imshow(M.to_numpy(), cmap="viridis")
    ax[0, 1].set(xticks=range(len(sel)), yticks=range(len(sel)), xlabel="critic of rater B", ylabel="evaluated on rater A",
                 title="Criterion 2: pairwise accuracy (diagonal = own critic)")
    ax[0, 1].set_xticklabels(sel, rotation=90, fontsize=7)
    ax[0, 1].set_yticklabels(sel, fontsize=7)
    fig.colorbar(im, ax=ax[0, 1])
    Vs = V.sort_values(["group", "gain_pts"])
    yy = np.arange(len(Vs))
    ax[1, 0].errorbar(Vs["gain_pts"], yy, xerr=[Vs["gain_pts"] - Vs["gain_ci_lo"], Vs["gain_ci_hi"] - Vs["gain_pts"]],
                      fmt="o", color="k", ms=3)
    ax[1, 0].scatter(Vs["gain_pts"], yy, c=np.where(Vs["group"] == "selected", "C3", "0.6"), zorder=3, s=20)
    ax[1, 0].axvline(0, color="k", lw=.5)
    ax[1, 0].set(yticks=yy, xlabel="gain over consensus, all pairs (acc pts, 95% CI)",
                 title="Criterion 1 (red = selected, grey = controls)")
    ax[1, 0].set_yticklabels([f"{p}{' (h)' if h else ''}" for p, h in zip(Vs["participant_id"], Vs["heuristic"])],
                             fontsize=7)
    w = 0.35
    labels = [f"<{GAP_BINS[0][1]}", f"{GAP_BINS[1][0]}-{GAP_BINS[1][1]}", f">{GAP_BINS[2][0]}"]
    Vsel = V[V.group == "selected"]
    pc = [Vsel[f"acc_personal_gap{lo}_{hi}"].mean() for lo, hi in GAP_BINS]
    cc = [Vsel[f"acc_consensus_gap{lo}_{hi}"].mean() for lo, hi in GAP_BINS]
    ax[1, 1].bar(np.arange(3) - w / 2, cc, w, label="consensus critic", color="0.6")
    ax[1, 1].bar(np.arange(3) + w / 2, pc, w, label="personal critic", color="C3")
    ax[1, 1].set(xticks=range(3), xlabel="score gap of the pair (pts)", ylabel="pairwise accuracy", ylim=(0.5, 1),
                 title="Accuracy by score gap (selected raters, mean)")
    ax[1, 1].set_xticklabels(labels)
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(a.out / "phase_b.png", dpi=130)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["consensus", "select", "personal", "all"], default="all")
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--emb", type=Path, default=Path("embeddings"))
    ap.add_argument("--phase-a", type=Path, default=Path("outputs/phase_a"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_b"))
    ap.add_argument("--encoders", nargs="+", default=["clip", "siglip2", "dinov2"])
    ap.add_argument("--cons-steps", type=int, default=6000)
    ap.add_argument("--n-select", type=int, default=8)
    ap.add_argument("--n-control", type=int, default=10)
    ap.add_argument("--min-ratings", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    raw, d = load_ratings(a.data)
    image_ids = np.sort(d["image_id"].unique())
    log(f"{len(d)} ratings, {d.participant_id.nunique()} raters, {len(image_ids)} images")

    if a.stage in ("consensus", "all"):
        choice = stage_consensus(a, d, image_ids)
    else:
        choice = json.loads((a.out / "consensus_select.json").read_text())
    if a.stage in ("select", "all"):
        S = stage_select(a, raw, d)
    else:
        S = pd.read_csv(a.out / "selected.csv")
    if a.stage in ("personal", "all"):
        stage_personal(a, raw, d, image_ids, choice, S)
    log("done")


if __name__ == "__main__":
    main()
