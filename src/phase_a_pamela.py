"""Phase A replication on PAMELA (bethgelab/PAMELA): is there consistent individual taste? Metadata only, no neural models.

Same statistics as src/phase_a.py (imported, unchanged params): LOO consensus + cross-fitted generic metadata prediction as
the baseline, taste = residual predictable from metadata out of sample (ridge, half-splits, permutation null, BH-FDR 5%),
effect = held-out within-rater pairwise accuracy gain over the baseline.

Features: one-hot group / style / type + pixel statistics computed here (PAMELA ships none), rank-gaussianised.
Checks: planted-taste synthetic raters (calibration + power) and pseudo-raters (no consistent person -> ~0 significant).

Pre-registered pass rule (CONTEXT.md §7): >=50% of tested raters significant at BH-FDR 5% AND median held-out gain >=2 pts.

Usage: python src/phase_a_pamela.py --data data/pamela --out outputs/phase_a_pamela
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats

from phase_a import analyse, bh, consensus_reliability, generic_metadata_prediction, pairacc

CAT_COLS = ["group", "style", "type"]
GAIN_THRESHOLDS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
SCALE = 4.0 / 100  # PAMELA 1-5 slider vs LAPIS 0-100, for the synthetic-rater noise and planted effects


# ---------------------------------------------------------------- data
def load(data: Path):
    rows = []
    for f in sorted((data / "annotations").glob("*.json")):
        rows += json.loads(f.read_text())
    d = pd.json_normalize(rows)
    d = d.rename(columns={"original_score": "rating", "image_metadata.group": "group", "image_metadata.style": "style",
                          "image_metadata.type": "type", "user_demographics.age": "age",
                          "user_demographics.gender": "gender"})
    return d


# ---------------------------------------------------------------- pixel statistics
def _rgb2lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def pixel_stats(path, size=256):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    a = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float64)
    hsv = np.asarray(im.resize((size, size), Image.BILINEAR).convert("HSV"), dtype=np.float64) / 255.0
    lab = _rgb2lab(a)
    L = lab[..., 0]
    gray = a @ np.array([0.299, 0.587, 0.114])
    out = {"aspect ratio": W / H, "RMS contrast": gray.std() / 255.0,
           "luminance entropy": _entropy(np.histogram(gray, 256, (0, 256))[0]),
           "color entropy": _entropy(np.histogramdd(a.reshape(-1, 3), bins=8, range=[(0, 256)] * 3)[0].ravel())}
    gm = np.hypot(ndimage.sobel(gray, 0), ndimage.sobel(gray, 1))
    out["edge density"] = float((gm > 0.1 * gm.max()).mean()) if gm.max() > 0 else 0.0
    out["gradient mean"] = float(gm.mean())
    for i, ch in enumerate("RGB"):
        out[f"mean {ch}"], out[f"std {ch}"] = a[..., i].mean(), a[..., i].std()
    for i, ch in enumerate("Lab"):
        out[f"mean {ch} (Lab)"], out[f"std {ch} (Lab)"] = lab[..., i].mean(), lab[..., i].std()
    h = hsv[..., 0] * 2 * np.pi
    out["hue cos"], out["hue sin"] = np.cos(h).mean(), np.sin(h).mean()
    out["hue concentration"] = np.hypot(out["hue cos"], out["hue sin"])
    for i, ch in [(1, "S"), (2, "V")]:
        out[f"mean {ch}"], out[f"std {ch}"] = hsv[..., i].mean(), hsv[..., i].std()
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    out["colorfulness"] = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    g0 = gray - gray.mean()
    den = (g0 * g0).sum() + 1e-9
    out["mirror symmetry lr"] = float((g0 * g0[:, ::-1]).sum() / den)
    out["mirror symmetry ud"] = float((g0 * g0[::-1]).sum() / den)
    dev = np.abs(L - L.mean())
    ys, xs = np.mgrid[0:size, 0:size] / (size - 1) - 0.5
    out["balance x"], out["balance y"] = (dev * xs).sum() / (dev.sum() + 1e-9), (dev * ys).sum() / (dev.sum() + 1e-9)
    out["balance"] = np.hypot(out["balance x"], out["balance y"])
    P = np.abs(np.fft.fftshift(np.fft.fft2(g0))) ** 2
    r = np.hypot(*(np.mgrid[0:size, 0:size] - size // 2)).astype(int)
    rad = np.bincount(r.ravel(), P.ravel()) / np.bincount(r.ravel())
    k = np.arange(4, size // 2)
    out["Fourier slope"] = float(np.polyfit(np.log(k), np.log(rad[k] + 1e-9), 1)[0])
    out["dark fraction"], out["bright fraction"] = float((L < 15).mean()), float((L > 90).mean())
    return out


def image_stats(data: Path, paths, cache: Path):
    if cache.exists():
        S = pd.read_csv(cache, index_col=0)
        if set(paths) <= set(S.index):
            return S.loc[list(paths)]
    t0 = time.time()
    rows = {}
    for i, p in enumerate(paths):
        rows[p] = pixel_stats(data / p)
        if (i + 1) % 1000 == 0:
            print(f"  pixel stats {i + 1}/{len(paths)} [{time.time() - t0:.0f}s]", flush=True)
    S = pd.DataFrame.from_dict(rows, orient="index")
    S.to_csv(cache)
    return S


def build_features(d, S):
    img = d.drop_duplicates("image_id").set_index("image_id")
    cats = pd.get_dummies(img[CAT_COLS].fillna("NA"), dtype=float)
    num = S.loc[img["image_path"]].set_axis(img.index)
    num = num.apply(lambda s: pd.Series(stats.norm.ppf((s.rank() - 0.5) / s.notna().sum()), index=s.index))
    X = pd.concat([cats, num.fillna(0.0)], axis=1)
    X = X.loc[:, X.std() > 0]
    return (X - X.mean()) / X.std()


# ---------------------------------------------------------------- checks
def self_test(raters, Xall, img_mean, args, rng):
    """Synthetic raters on real image sets: y = panel mean + noise (+ planted taste: likes Art and high contrast)."""
    art, contrast = Xall.columns.get_loc("type_Art"), Xall.columns.get_loc("RMS contrast")
    res = []
    for pid, (img_ids, B) in list(raters.items())[: args.synth_n]:
        X = Xall.loc[img_ids].to_numpy()
        base = img_mean.loc[img_ids].to_numpy()
        for delta in [0, 3, 6, 10]:  # LAPIS points per SD of each feature, rescaled
            taste = delta * SCALE * (np.sign(X[:, art]) + X[:, contrast]) / 2
            y = np.clip(base + taste + rng.normal(0, args.synth_noise * SCALE, len(base)), 1, 5)
            r = analyse(y, B, X, args.lam, args.splits, args.perms, rng)
            res.append(dict(delta=delta, p=r["p"], gain=r["gain"]))
    s = pd.DataFrame(res)
    s["sig"] = False
    for delta, g in s.groupby("delta"):
        s.loc[g.index, "sig"] = bh(g["p"].to_numpy())[0]
    return s.groupby("delta").agg(frac_p05=("p", lambda p: (p < .05).mean()), frac_bh=("sig", "mean"),
                                  mean_gain=("gain", "mean")).reset_index()


def pseudo_raters(d, X, args, rng):
    """For each image a real rater rated, borrow a random other rater's rating: real scale habits, no consistent person."""
    by_img = {k: v for k, v in d.groupby("image_id")}
    raters = [(pid, gr) for pid, gr in d.groupby("participant_id") if len(gr) >= args.min_n and gr["rating"].std() > 0]
    rows = []
    for k in rng.choice(len(raters), min(args.n_pseudo, len(raters)), replace=False):
        pid, gr = raters[k]
        ys, cs = [], []
        for img in gr["image_id"]:
            o = by_img[img]
            o = o[o["participant_id"] != pid]
            j = rng.integers(len(o))
            ys.append(o["rating"].iloc[j])
            cs.append((o["rating"].sum() - o["rating"].iloc[j]) / max(len(o) - 1, 1))
        B = np.column_stack([cs, gr["g"].to_numpy()])
        rows.append(analyse(np.array(ys, float), B, X.loc[gr["image_id"]].to_numpy(), args.lam, args.splits,
                            args.perms, rng))
    R = pd.DataFrame(rows)
    return dict(n=len(R), frac_sig_fdr05=float(bh(R["p"].to_numpy())[0].mean()), frac_p05=float((R["p"] < .05).mean()),
                gain_median=float(R["gain"].median()))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/pamela"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_a_pamela"))
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--synth-n", type=int, default=100)
    ap.add_argument("--synth-noise", type=float, default=20.0, help="LAPIS points, rescaled by SCALE")
    ap.add_argument("--n-pseudo", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    raw = load(args.data)
    assert not raw.duplicated(["participant_id", "image_id"]).any()
    S = image_stats(args.data, raw["image_path"].unique(), args.data / "image_stats.csv")
    X = build_features(raw, S)
    d = raw[["participant_id", "image_id", "rating", "age", "gender"]].copy()
    g = d.groupby("image_id")["rating"]
    d["n_img"] = g.transform("count")
    d["cons"] = (g.transform("sum") - d["rating"]) / (d["n_img"] - 1)
    d = d[d["n_img"] > 1]
    img_mean = g.mean()
    d["g"] = generic_metadata_prediction(img_mean, X, rng).reindex(d["image_id"]).to_numpy()
    rel = consensus_reliability(d, rng)
    print(f"loaded {len(d)} ratings, {d.participant_id.nunique()} raters, {X.shape[1]} features, "
          f"consensus reliability {rel:.3f} [{time.time() - t0:.0f}s]", flush=True)

    rows, raters = [], {}
    for pid, gr in d.groupby("participant_id"):
        y, c = gr["rating"].to_numpy(float), gr["cons"].to_numpy(float)
        row = dict(participant_id=pid, n=len(gr), mean=y.mean(), sd=y.std(), age=gr["age"].iloc[0],
                   gender=gr["gender"].iloc[0])
        if len(gr) >= 2 and y.std() > 0:
            row["acc_cons"] = pairacc(y, c)
            row["spearman_cons"] = stats.spearmanr(y, c).statistic
        if len(gr) >= args.min_n and y.std() > 0:
            B = gr[["cons", "g"]].to_numpy(float)
            raters[pid] = (gr["image_id"].to_numpy(), B)
            row.update(analyse(y, B, X.loc[gr["image_id"]].to_numpy(), args.lam, args.splits, args.perms, rng))
        rows.append(row)
    R = pd.DataFrame(rows)
    tested = R["p"].notna()
    R.loc[tested, "sig"], R.loc[tested, "q"] = bh(R.loc[tested, "p"].to_numpy())
    R["sig"] = R["sig"].astype("boolean")
    R = R.sort_values("taste_z", ascending=False)
    R.to_csv(args.out / "raters.csv", index=False)
    T = R[tested]
    print(f"real raters done [{time.time() - t0:.0f}s]", flush=True)

    st = self_test(raters, X, img_mean, args, rng)
    print(st.round(3).to_string(), flush=True)
    ps = pseudo_raters(d, X, args, rng)
    print("pseudo:", ps, flush=True)

    frac_sig, gain_med = float(T["sig"].mean()), float(T["gain"].median())
    summary = dict(
        verdict=dict(rule=">=50% of tested raters sig at BH-FDR 5% AND median held-out gain >=2 pts",
                     frac_sig=frac_sig, gain_median=gain_med, passed=bool(frac_sig >= 0.5 and gain_med >= 2.0)),
        n_ratings=int(len(d)), n_raters=int(len(R)), n_tested=int(tested.sum()), n_features=int(X.shape[1]),
        n_per_tested_rater_median=float(T["n"].median()),
        params=dict(lam=args.lam, splits=args.splits, perms=args.perms, min_n=args.min_n, seed=args.seed),
        consensus_reliability_full_panel=rel,
        acc_cons_median=float(R["acc_cons"].median()),
        n_sig_fdr05=int(T["sig"].sum()),
        frac_p05_uncorrected=float((T["p"] < .05).mean()),
        gain_median_sig=float(T.loc[T["sig"], "gain"].median()) if T["sig"].any() else None,
        gain_vs_cons_only_median=float(T["gain_vs_cons_only"].median()),
        threshold_curve=[dict(min_gain_pts=t, n_raters=int((T["sig"] & (T["gain"] >= t)).sum()))
                         for t in GAIN_THRESHOLDS],
        frac_sig_by_n=T.groupby(pd.cut(T["n"], [0, 100, 200, 400, 2000]), observed=True)["sig"]
                       .agg(["mean", "size"]).reset_index().astype({"n": str}).to_dict("records"),
        corr_n_taste_z=float(stats.spearmanr(T["n"], T["taste_z"]).statistic),
        corr_acc_cons_taste_z=float(stats.spearmanr(T["acc_cons"], T["taste_z"]).statistic),
        self_test=st.to_dict("records"),
        pseudo_raters=ps,
        runtime_s=round(time.time() - t0),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
