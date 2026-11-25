"""What if the app's test pairs were the model's MOST SURE pairs instead of random ones?

Simulates the app on LAPIS raters the app model never saw (fold-0 Phase C model, exactly app/data: U, mu, Sigma, tau):
  - each rater's images are split 50/50 per seed: learning half, test half (test images are never shown in learning);
  - learning: 50 picks = 5 random + 45 infogain (p(1-p) d^T H^-1 d over 3000 candidate pairs, refit after every pick),
    no image repeated, answer = higher LAPIS rating (tied pairs are not asked);
  - z = MAP under N(mu, Sigma/tau), as in the app (no side-bias term: LAPIS has no positions);
  - test: 20 pairs from the test half under several designs (DESIGNS): random; "sure" pairs (largest |s(a)-s(b)| under
    the person's z); "contrarian" pairs (the personal model is most sure AND the crowd model z = mu picks the other
    image); the rest random. No image reused. A pair the rater tied counts as half a hit (a forced choice on a tie is a coin flip).
Reports the score out of 20 for the personal model and the crowd model on the same pairs, and saves each fitted z
(z_app_fit.csv), the reference distribution for the app's "you vs other people" comparison.

Second part: the real app sessions (random test pairs only): accuracy on each session's most-confident test pairs.

Usage: python src/sure_test.py [--seeds 5]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_TEST, N_TRAIN, N_RANDOM, N_CAND = 20, 50, 5, 3000


def load_model(app):
    m = json.loads((app / "data/meta.json").read_text())
    U = np.fromfile(app / "data/U.bin", dtype="<f4").reshape(m["n"], m["k"]).astype(float)
    pos = pd.Series(np.arange(m["n"]), index=m["image_id"])
    return m, U, pos, np.array(m["mu"]), np.array(m["Sinv"]) * m["tau"]


def fit(D, y, mu, P):
    """MAP of z: logistic loss on D z (D = U[top] - U[bottom], y = 1 if top preferred) + 0.5 (z-mu)' P (z-mu)."""
    def obj(z):
        l = D @ z
        return 0.5 * (z - mu) @ P @ (z - mu) + np.sum(np.logaddexp(0, l) - y * l)
    z, f0 = mu.copy(), obj(mu)
    for _ in range(100):
        p = 1 / (1 + np.exp(-(D @ z)))
        g = P @ (z - mu) + D.T @ (p - y)
        H = P + (D * (p * (1 - p))[:, None]).T @ D
        step, t = np.linalg.solve(H, g), 1.0
        while t > 1e-8 and obj(z - t * step) > f0:  # damped Newton, as in the app
            t /= 2
        if t <= 1e-8:
            break
        z = z - t * step
        f0 = obj(z)
        if np.abs(t * step).max() < 1e-9:
            break
    return z, H


def learn(U, img, y, rng, mu, P):
    """App-style learning phase on images img (global rows) with ratings y. Returns z."""
    used, I, J, T = set(), [], [], []
    m = len(img)
    for q in range(N_TRAIN):
        free = np.array([i for i in range(m) if i not in used])
        a, b = rng.choice(free, N_CAND), rng.choice(free, N_CAND)
        ok = (a != b) & (y[a] != y[b])
        a, b = a[ok], b[ok]
        if len(a) == 0:
            break
        if q < N_RANDOM:
            k = 0
        else:
            z, H = fit(np.array(D_), np.array(T, float), mu, P)
            Dc = U[img[a]] - U[img[b]]
            p = 1 / (1 + np.exp(-(Dc @ z)))
            k = int(np.argmax(p * (1 - p) * np.einsum("ij,jk,ik->i", Dc, np.linalg.inv(H), Dc)))
        i, j = a[k], b[k]
        used |= {i, j}
        I.append(i), J.append(j), T.append(float(y[i] > y[j]))
        D_ = U[img[np.array(I)]] - U[img[np.array(J)]]
    return fit(D_, np.array(T), mu, P)[0]


def contra_pairs(s, c, m, k, used):
    """k disjoint pairs where the personal model (scores s) is most sure AND the crowd model (scores c) picks the other image.
    Greedy by personal margin over all pairs of the m images not yet used (marks the images it takes in `used`)."""
    i, j = np.triu_indices(m, 1)
    ds, dc = s[i] - s[j], c[i] - c[j]
    ok = (np.sign(ds) * np.sign(dc) < 0) & ~used[i] & ~used[j]
    i, j, ds = i[ok], j[ok], ds[ok]
    out = []
    for t in np.argsort(-np.abs(ds)):
        a, b = i[t], j[t]
        if used[a] or used[b]:
            continue
        out.append((a, b)); used[a] = used[b] = True
        if len(out) == k:
            break
    return out


def design_pairs(s, c, m, n_sure, n_contra, rng):
    """n_contra contrarian-sure pairs, then n_sure most-sure pairs, then random ones (all disjoint), N_TEST in total."""
    used, pairs = np.zeros(m, bool), contra_pairs(s, c, m, n_contra, np.zeros(m, bool)) if n_contra else []
    for a, b in pairs:
        used[a] = used[b] = True
    o = [x for x in np.argsort(s) if not used[x]]
    lo, hi = 0, len(o) - 1
    for _ in range(n_sure):
        pairs.append((o[hi], o[lo])); used[o[hi]] = used[o[lo]] = True
        lo, hi = lo + 1, hi - 1
    free = rng.permutation(np.flatnonzero(~used))
    n_rand = N_TEST - len(pairs)
    pairs += [(free[2 * t], free[2 * t + 1]) for t in range(n_rand)]
    return np.array(pairs), len(pairs) - n_rand  # pairs, number of designed (non-random) pairs


# (name, n_sure, n_contra): designs of the 20 test pairs
DESIGNS = [("random", 0, 0), ("sure10", 10, 0), ("contra5", 0, 5), ("contra10", 0, 10), ("sure5_contra5", 5, 5)]


def score(pairs, y, s):
    a, b = pairs[:, 0], pairs[:, 1]
    dy, ds = y[a] - y[b], s[a] - s[b]
    return np.where(dy == 0, 0.5, (np.sign(dy) == np.sign(ds)).astype(float))


def simulate(a, U, pos, mu, P):
    piaa = pd.read_csv(a.data / "piaa/piaa.csv", keep_default_na=False, na_values=[""],
                       usecols=["participant_id", "image_id", "rating"])
    d = piaa.groupby(["participant_id", "image_id"], as_index=False)["rating"].mean()
    # raters never used to train the shared fold-0 model: the 18 Phase B raters + the 40 Phase C validation raters
    ids = pd.read_csv(a.phase_b / "selected.csv")["participant_id"].tolist()
    cnt = d.groupby("participant_id").size()
    pool = np.sort(cnt[(cnt >= 300) & ~cnt.index.isin(ids)].index.to_numpy())
    vids = sorted(np.random.default_rng(1).choice(pool, 40, replace=False).tolist())  # = phase_c_lowrank.setup
    rows, zrows = [], []
    for pid in ids + vids:
        g = d[d.participant_id == pid]
        img_all, y_all = pos[g.image_id].to_numpy(), g.rating.to_numpy(float)
        for seed in range(a.seeds):
            rng = np.random.default_rng([seed, pid])
            perm = rng.permutation(len(g))
            tr, te = perm[: len(g) // 2], perm[len(g) // 2:]
            z = learn(U, img_all[tr], y_all[tr], rng, mu, P)
            img, y = img_all[te], y_all[te]
            s_me, s_crowd = U[img] @ z, U[img] @ mu
            zrows.append(dict(pid=pid, seed=seed, **{f"z{q}": v for q, v in enumerate(z)}))
            for name, ns, nc in DESIGNS:
                pr, nd = design_pairs(s_me, s_crowd, len(img), ns, nc, np.random.default_rng([seed, pid, ns, nc]))
                h_me, h_cr = score(pr, y, s_me), score(pr, y, s_crowd)
                rows.append(dict(pid=pid, group="phase_b" if pid in ids else "val", seed=seed, design=name,
                                 me=h_me.sum(), crowd=h_cr.sum(), me_designed=h_me[:nd].mean() if nd else np.nan,
                                 crowd_designed=h_cr[:nd].mean() if nd else np.nan,
                                 me_rand=h_me[nd:].mean() if nd < N_TEST else np.nan))
        print(f"rater {pid} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(zrows)


def real_sessions(dirs, U, pos, mu):
    out = []
    for f in sorted(sum([glob.glob(str(x / "*.json")) for x in dirs], [])):
        r = json.loads(Path(f).read_text())
        if r.get("partial") or len(r.get("test", [])) < 10:
            continue
        t = pd.DataFrame(r["test"])
        z = np.array(r["z"])
        a, b = pos[t.top].to_numpy(), pos[t.bottom].to_numpy()
        ds = (U[a] - U[b]) @ z
        hit = ((ds >= 0).astype(int) == (t.choice == 0).astype(int)).to_numpy()
        crowd = (((U[a] - U[b]) @ mu >= 0).astype(int) == (t.choice == 0).astype(int)).to_numpy()
        o = np.argsort(-np.abs(ds))
        n = len(t)
        out.append(dict(session=Path(f).stem[-6:], n_test=n, all=hit.mean(), crowd_all=crowd.mean(),
                        top_quarter=hit[o[: n // 4]].mean(), top_half=hit[o[: n // 2]].mean(),
                        bottom_half=hit[o[n // 2:]].mean()))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--app", type=Path, default=Path("app"))
    ap.add_argument("--phase-b", type=Path, default=Path("outputs/phase_b"))
    ap.add_argument("--sessions", type=Path, nargs="+", default=[Path("../taste-site/sessions"), Path("outputs/user_taste")])
    ap.add_argument("--out", type=Path, default=Path("outputs/sure_test"))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m, U, pos, mu, P = load_model(a.app)

    R, Z = simulate(a, U, pos, mu, P)
    R.to_csv(a.out / "sim.csv", index=False)
    Z.to_csv(a.out / "z_app_fit.csv", index=False)  # z fitted from 50 app-style picks: reference crowd for "you vs others"
    per = R.groupby(["design", "pid"])[["me", "crowd", "me_designed", "crowd_designed", "me_rand"]].mean()
    summ = per.groupby("design").agg(me_mean=("me", "mean"), crowd_mean=("crowd", "mean"),
                                     acc_designed=("me_designed", "mean"), crowd_acc_designed=("crowd_designed", "mean"),
                                     acc_rand=("me_rand", "mean"))
    g = R.groupby("design")
    summ["p_ge15"] = g["me"].apply(lambda v: (v >= 15).mean())
    summ["p_le10"] = g["me"].apply(lambda v: (v <= 10).mean())
    summ["p_beats_crowd"] = R.assign(w=R.me > R.crowd).groupby("design")["w"].mean()
    summ["mean_margin_vs_crowd"] = R.assign(w=R.me - R.crowd).groupby("design")["w"].mean()
    summ = summ.loc[[d[0] for d in DESIGNS]]
    print("\nSimulated score out of 20 (58 held-out LAPIS raters x", a.seeds, "seeds)\n", summ.round(3).to_string())
    summ.to_csv(a.out / "summary.csv")

    S = real_sessions(a.sessions, U, pos, mu)
    print("\nReal app sessions (random test pairs): accuracy by model confidence\n", S.round(2).to_string())
    S.to_csv(a.out / "real_sessions.csv", index=False)


if __name__ == "__main__":
    main()
