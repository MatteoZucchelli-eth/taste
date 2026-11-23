"""Robustness checks for Phase A (run after phase_a.py logic is validated).

  pseudo   : pseudo-raters — for each image a real rater rated, take a random *other* rater's rating of it.
             Realistic noise/scale habits, but no single consistent person → should show ~no taste.
  nonlin   : baseline adds c^2, c^3, g^2 (non-linear scale use must not masquerade as taste).
  abstract : taste features = only the abstract/figurative label (how much is this one axis?).

Usage: python src/phase_a_checks.py --data data/lapis --out outputs/phase_a
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase_a import analyse, bh, build_features, generic_metadata_prediction, load


def summarise(rows, name):
    R = pd.DataFrame(rows)
    sig = bh(R["p"].to_numpy())[0]
    out = dict(check=name, n=len(R), frac_sig_fdr05=float(sig.mean()), frac_p05=float((R["p"] < .05).mean()),
               gain_median=float(R["gain"].median()))
    print(out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_a"))
    ap.add_argument("--lam", type=float, default=1000.0)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--n-pseudo", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    raw, attrs = load(args.data)
    d = raw.groupby(["participant_id", "image_id"], as_index=False)["rating"].mean()
    g = d.groupby("image_id")["rating"]
    d["n_img"] = g.transform("count")
    d["cons"] = (g.transform("sum") - d["rating"]) / (d["n_img"] - 1)
    d = d[d["n_img"] > 1]
    X = build_features(raw, attrs)
    d["g"] = generic_metadata_prediction(g.mean(), X, rng).reindex(d["image_id"]).to_numpy()
    by_img = {k: v for k, v in d.groupby("image_id")}
    raters = [(pid, gr) for pid, gr in d.groupby("participant_id") if len(gr) >= 40 and gr["rating"].std() > 0]
    abstract_cols = [c for c in X.columns if c.startswith("2_styles_")][:1]

    results = []
    # pseudo-raters
    rows = []
    for k in rng.choice(len(raters), args.n_pseudo, replace=False):
        pid, gr = raters[k]
        ys, cs = [], []
        for img in gr["image_id"]:
            other = by_img[img]
            other = other[other["participant_id"] != pid]
            j = rng.integers(len(other))
            ys.append(other["rating"].iloc[j])
            # LOO consensus excluding the borrowed rater
            cs.append((other["rating"].sum() - other["rating"].iloc[j]) / max(len(other) - 1, 1))
        y = np.array(ys, float)
        B = np.column_stack([cs, gr["g"].to_numpy()])
        rows.append(analyse(y, B, X.loc[gr["image_id"]].to_numpy(), args.lam, args.splits, args.perms, rng))
    results.append(summarise(rows, "pseudo_raters (should be ~0)"))

    # non-linear baseline and abstract-only features, on all real raters
    rows_nl, rows_ab = [], []
    for pid, gr in raters:
        y = gr["rating"].to_numpy(float)
        c, gg = gr["cons"].to_numpy(float), gr["g"].to_numpy(float)
        Xa = X.loc[gr["image_id"]]
        B_nl = np.column_stack([c, gg, (c - 50) ** 2 / 50, (c - 50) ** 3 / 2500, (gg - 50) ** 2 / 50])
        rows_nl.append(analyse(y, B_nl, Xa.to_numpy(), args.lam, args.splits, args.perms, rng))
        rows_ab.append(analyse(y, np.column_stack([c, gg]), Xa[abstract_cols].to_numpy(), 10.0, args.splits,
                               args.perms, rng))
    results.append(summarise(rows_nl, "real raters, non-linear baseline"))
    results.append(summarise(rows_ab, "real raters, abstract/figurative label only"))
    (args.out / "checks.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
