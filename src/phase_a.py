"""Phase A: does LAPIS contain consistent *individual* taste? (ratings + metadata only, no neural models)

For each rater A:
  1. LOO consensus c_i = mean rating of image i by everyone except A.
  2. agreement(A, consensus): pairwise accuracy on A's pairs (A-ties dropped, consensus ties = 0.5) and Spearman.
  3. deviation e = residual of A's rating regressed on c (removes A's offset/scale and the consensus part).
  4. taste test: ridge e ~ metadata fitted on a random half of A's images, correlation with e on the other half,
     averaged over K splits. Null: same statistic with e permuted across A's images. p from a Gaussian fit to the null
     (calibration checked on synthetic no-taste raters), BH-FDR over raters.
  5. effect size: held-out pairwise accuracy of (consensus + metadata taste, fitted on the other half) minus that of
     consensus alone = how much a (very simple) personal critic beats consensus for A.
Also: global consensus reliability, test-retest on the few repeated (rater, image) pairs, and a planted-taste self-test.

Usage: python src/phase_a.py --data data/lapis --out outputs/phase_a
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAT_COLS = ["2_styles", "4_styles", "genre", "style"]
GAIN_THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]  # accuracy points


# ---------------------------------------------------------------- data
def load(data: Path):
    d = pd.read_csv(data / "piaa/piaa.csv", keep_default_na=False, na_values=[""])
    attrs = pd.read_csv(data / "image_attributes/image_attributes.csv")
    return d, attrs


def build_features(d, attrs):
    """Per-image metadata matrix: one-hot style/genre labels + rank-gaussianised visual stats, all standardised."""
    img = d.drop_duplicates("image_id").set_index("image_id")[["path"] + CAT_COLS]
    cats = pd.get_dummies(img[CAT_COLS].fillna("NA"), dtype=float)
    num = attrs.drop(columns=["image_filename"]).set_index("path").select_dtypes("number")
    num = num.apply(lambda s: pd.Series(stats.norm.ppf((s.rank() - 0.5) / s.notna().sum()), index=s.index))
    num = num.reindex(img["path"]).set_axis(img.index).fillna(0.0)
    X = pd.concat([cats, num], axis=1)
    X = (X - X.mean()) / X.std().replace(0, 1)
    return X


def pairacc(y, s):
    """Fraction of A's non-tied pairs ordered the same way by score s (ties in s count 0.5)."""
    dy = np.sign(y[:, None] - y[None, :])
    ds = np.sign(s[:, None] - s[None, :])
    m = np.triu(dy != 0, 1)
    prod = (dy * ds)[m]
    return (prod > 0).mean() + 0.5 * (prod == 0).mean()


def colcorr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-12)


# ---------------------------------------------------------------- per-rater analysis
def ols_fit(B, y):
    A = np.column_stack([np.ones(len(y)), B])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda Bn: np.column_stack([np.ones(len(Bn)), Bn]) @ coef


def analyse(y, B, X, lam, splits, perms, rng):
    """Taste statistic, its permutation null, and held-out accuracy gain for one rater.

    B = baseline columns [LOO consensus c, generic metadata prediction g]; the baseline (= consensus critic) is OLS of
    y on B. Taste = A's residual from that baseline is predictable from metadata out-of-sample."""
    n, p = X.shape
    c = B[:, 0]
    e = y - ols_fit(B, y)(B)
    E = np.empty((n, perms + 1))
    E[:, 0] = e
    for k in range(1, perms + 1):
        E[:, k] = rng.permutation(e)
    stat = np.zeros(perms + 1)
    gains, gains_c, accm, accb, accc = [], [], [], [], []
    I = lam * np.eye(p)
    for _ in range(splits):
        idx = rng.permutation(n)
        i1, i2 = idx[: n // 2], idx[n // 2:]
        X1, X2 = X[i1], X[i2]
        mu = X1.mean(0)
        X1c, X2c = X1 - mu, X2 - mu
        W = np.linalg.solve(X1c.T @ X1c + I, X1c.T)  # p x n1
        E1 = E[i1] - E[i1].mean(0)
        stat += colcorr(X2c @ (W @ E1), E[i2])
        # out-of-sample personal predictor: consensus fit + metadata ridge, both fitted on half 1 only
        f1 = ols_fit(B[i1], y[i1])
        base2 = f1(B[i2])
        r1 = y[i1] - f1(B[i1])
        pred2 = base2 + X2c @ (W @ (r1 - r1.mean()))
        am, ab, ac = pairacc(y[i2], pred2), pairacc(y[i2], base2), pairacc(y[i2], c[i2])
        accm.append(am)
        accb.append(ab)
        accc.append(ac)
        gains.append(am - ab)
        gains_c.append(am - ac)
    stat /= splits
    null = stat[1:]
    z = (stat[0] - null.mean()) / null.std()
    return dict(
        taste_r=stat[0],
        taste_z=z,
        p_perm=(1 + (null >= stat[0]).sum()) / (perms + 1),
        p=stats.norm.sf(z),
        acc_cons_half=np.mean(accc),
        acc_base_half=np.mean(accb),
        acc_model_half=np.mean(accm),
        gain=100 * np.mean(gains),
        gain_vs_cons_only=100 * np.mean(gains_c),
        gain_sd=100 * np.std(gains),
    )


def bh(p, q=0.05):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    thr = q * np.arange(1, m + 1) / m
    passed = p[order] <= thr
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, bool)
    sig[order[:k]] = True
    qv = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(qv, 1)
    return sig, out


def generic_metadata_prediction(img_mean, X, rng, lam=10.0, folds=5):
    """Cross-fitted ridge of each image's panel mean on metadata: what metadata says about the *consensus*.
    Includes A's own ratings at ~1/24 weight, which can only absorb a little of A's taste (conservative)."""
    Xm = X.loc[img_mean.index].to_numpy()
    y = img_mean.to_numpy()
    out = np.empty(len(y))
    fold = rng.integers(0, folds, len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        mu, ym = Xm[tr].mean(0), y[tr].mean()
        A = Xm[tr] - mu
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
        out[te] = ym + (Xm[te] - mu) @ beta
    return pd.Series(out, index=img_mean.index)


# ---------------------------------------------------------------- extra checks
def consensus_reliability(d, rng, reps=20):
    """Split each image's raters in two, correlate half-means across images, Spearman-Brown to full panel."""
    rs = []
    g = d[["image_id", "rating"]].copy()
    for _ in range(reps):
        g["half"] = rng.random(len(g)) < 0.5
        m = g.groupby(["image_id", "half"])["rating"].mean().unstack().dropna()
        r = np.corrcoef(m[True], m[False])[0, 1]
        rs.append(2 * r / (1 + r))
    return float(np.mean(rs))


def test_retest(raw):
    dup = raw[raw.duplicated(["participant_id", "image_id"], keep=False)]
    pairs = dup.sort_values(["participant_id", "image_id"]).groupby(["participant_id", "image_id"])["rating"]
    first, second = pairs.nth(0).to_numpy(), pairs.nth(1).to_numpy()
    return dict(
        n_pairs=int(len(first)),
        n_raters=int(dup.participant_id.nunique()),
        pearson=float(np.corrcoef(first, second)[0, 1]),
        mean_abs_diff=float(np.abs(first - second).mean()),
        exact_same=float((first == second).mean()),
    )


def self_test(raters, Xall, img_mean, args, rng):
    """Synthetic raters on real image sets: y = panel mean + noise (+ planted metadata taste of delta points)."""
    noise = args.synth_noise
    abstract = Xall.columns.get_loc("2_styles_ABSTRACT")
    contrast = Xall.columns.get_loc("RMS contrast")
    res = []
    for pid, (img_ids, B) in list(raters.items())[: args.synth_n]:
        X = Xall.loc[img_ids].to_numpy()
        base = img_mean.loc[img_ids].to_numpy()
        for delta in [0, 3, 6, 10]:
            # planted taste: likes abstract art and high contrast, delta points per SD of each
            taste = delta * (np.sign(X[:, abstract]) + X[:, contrast]) / 2
            y = np.clip(base + taste + rng.normal(0, noise, len(base)), 0, 100).round()
            r = analyse(y, B, X, args.lam, args.splits, args.perms, rng)
            res.append(dict(delta=delta, p=r["p"], p_perm=r["p_perm"], gain=r["gain"]))
    s = pd.DataFrame(res)
    s["sig"] = False
    for delta, g in s.groupby("delta"):  # BH within each condition, as in the real run
        s.loc[g.index, "sig"] = bh(g["p"].to_numpy())[0]
    return s


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_a"))
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--min-n", type=int, default=40, help="raters with fewer ratings are reported but not tested")
    ap.add_argument("--synth-n", type=int, default=100)
    ap.add_argument("--synth-noise", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test-only", action="store_true", help="for tuning lam on synthetic raters only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    raw, attrs = load(args.data)
    retest = test_retest(raw)
    # average the few repeated (rater, image) ratings
    meta_cols = ["age", "demo_edu", "Art Interest VAIAK", "imgs_known"]
    d = raw.groupby(["participant_id", "image_id"], as_index=False).agg(
        rating=("rating", "mean"), rt=("response time", "median"), **{c: (c, "first") for c in meta_cols})
    g = d.groupby("image_id")["rating"]
    d["n_img"] = g.transform("count")
    d["cons"] = (g.transform("sum") - d["rating"]) / (d["n_img"] - 1)
    d = d[d["n_img"] > 1]
    img_mean = g.mean()
    X = build_features(raw, attrs)
    d["g"] = generic_metadata_prediction(img_mean, X, rng).reindex(d["image_id"]).to_numpy()
    rel = consensus_reliability(d, rng)
    print(f"loaded {len(d)} ratings, {d.participant_id.nunique()} raters, {X.shape[1]} metadata features "
          f"[{time.time() - t0:.0f}s]", flush=True)

    rows, raters = [], {}
    if args.self_test_only:
        for pid, gr in d.groupby("participant_id"):
            if len(gr) >= args.min_n:
                raters[pid] = (gr["image_id"].to_numpy(), gr[["cons", "g"]].to_numpy(float))
        S = self_test(raters, X, img_mean, args, rng)
        print(S.groupby("delta").agg(frac_p05=("p", lambda p: (p < .05).mean()), frac_bh=("sig", "mean"),
                                     gain=("gain", "mean")).round(3).to_string())
        return
    for pid, gr in d.groupby("participant_id"):
        y, c = gr["rating"].to_numpy(float), gr["cons"].to_numpy(float)
        B = gr[["cons", "g"]].to_numpy(float)
        row = dict(
            participant_id=pid, n=len(gr), mean=y.mean(), sd=y.std(), n_unique=len(np.unique(y)),
            frac_at_mode=pd.Series(y).value_counts(normalize=True).iloc[0], median_rt_ms=gr["rt"].median(),
            vaiak=gr["Art Interest VAIAK"].iloc[0], imgs_known=gr["imgs_known"].iloc[0], edu=gr["demo_edu"].iloc[0],
            age=gr["age"].iloc[0],
        )
        if len(gr) >= 2 and y.std() > 0:
            row["acc_cons"] = pairacc(y, c)
            row["spearman_cons"] = stats.spearmanr(y, c).statistic
        if len(gr) >= args.min_n and y.std() > 0:
            raters[pid] = (gr["image_id"].to_numpy(), B)
            row.update(analyse(y, B, X.loc[gr["image_id"]].to_numpy(), args.lam, args.splits, args.perms, rng))
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} raters [{time.time() - t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    tested = R["p"].notna()
    R.loc[tested, "sig"], R.loc[tested, "q"] = bh(R.loc[tested, "p"].to_numpy())
    R["sig"] = R["sig"].astype("boolean")
    R = R.sort_values("taste_z", ascending=False)
    R.to_csv(args.out / "raters.csv", index=False)

    print(f"self-test [{time.time() - t0:.0f}s]", flush=True)
    S = self_test(raters, X, img_mean, args, rng)
    S.to_csv(args.out / "self_test.csv", index=False)
    st = S.groupby("delta").agg(frac_p05=("p", lambda p: (p < .05).mean()), frac_bh=("sig", "mean"),
                                mean_gain=("gain", "mean")).reset_index()

    T = R[tested]
    curve = [dict(min_gain_pts=t, n_raters=int((T["sig"] & (T["gain"] >= t)).sum())) for t in GAIN_THRESHOLDS]
    summary = dict(
        gain_vs_cons_only_median=float(T["gain_vs_cons_only"].median()),
        n_ratings=int(len(d)), n_raters=int(R.shape[0]), n_tested=int(tested.sum()), n_features=int(X.shape[1]),
        params=dict(lam=args.lam, splits=args.splits, perms=args.perms, min_n=args.min_n, seed=args.seed),
        consensus_reliability_full_panel=rel,
        test_retest=retest,
        acc_cons_median=float(R["acc_cons"].median()),
        acc_cons_iqr=[float(R["acc_cons"].quantile(.25)), float(R["acc_cons"].quantile(.75))],
        n_sig_fdr05=int(T["sig"].sum()),
        frac_p05_uncorrected=float((T["p"] < .05).mean()),
        gain_median_all=float(T["gain"].median()),
        gain_median_sig=float(T.loc[T["sig"], "gain"].median()) if T["sig"].any() else None,
        threshold_curve=curve,
        self_test=st.to_dict("records"),
        corr_vaiak_taste_z=float(stats.spearmanr(T["vaiak"], T["taste_z"], nan_policy="omit").statistic),
        corr_acc_cons_taste_z=float(stats.spearmanr(T["acc_cons"], T["taste_z"]).statistic),
        top10=T.head(10)[["participant_id", "n", "acc_cons", "taste_r", "taste_z", "q", "gain", "gain_sd",
                          "acc_model_half", "vaiak", "imgs_known"]].to_dict("records"),
        runtime_s=round(time.time() - t0),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    ax[0, 0].hist(R["acc_cons"].dropna(), bins=40, color="0.4")
    ax[0, 0].set(xlabel="pairwise agreement with LOO consensus", ylabel="raters", title="How consensus-like is each rater")
    col = np.where(T["sig"], "C3", "0.6")
    ax[0, 1].scatter(T["gain"], -np.log10(T["p"].clip(lower=1e-300)), s=8, c=col)
    ax[0, 1].axvline(0, color="k", lw=.5)
    ax[0, 1].set(xlabel="held-out accuracy gain over consensus (pts)", ylabel="-log10 p (taste test)",
                 title="Taste: significance vs size (red = FDR 5%)")
    ax[1, 0].plot([c["min_gain_pts"] for c in curve], [c["n_raters"] for c in curve], "o-")
    ax[1, 0].set(xlabel="minimum gain (pts)", ylabel="raters (FDR-significant)", title="Raters with taste vs threshold")
    ax[1, 1].scatter(T["vaiak"], T["taste_z"], s=8, c=col)
    ax[1, 1].set(xlabel="Art Interest (VAIAK)", ylabel="taste z", title="Taste vs art interest")
    fig.tight_layout()
    fig.savefig(args.out / "phase_a.png", dpi=130)
    print(json.dumps({k: summary[k] for k in summary if k != "top10"}, indent=1, default=float))
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
