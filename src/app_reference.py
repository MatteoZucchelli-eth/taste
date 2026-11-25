"""Reference crowd for the app's "you vs other people" comparison.

Reads the z vectors that src/sure_test.py fitted from 50 app-style picks for held-out LAPIS raters (the same shrinkage a
player gets in the app, so percentiles are fair) and writes app/data/ref.json:
  axes[k] = sorted standardised deviations (z_k - mu_k) / sqrt(Sig_kk) (the app's trait strength),
  uniq    = sorted Mahalanobis distances sqrt((z-mu)' Sinv (z-mu)) (how far a taste is from the average; no longer shown),
  mix     = mean and covariance of those deviations, for the app's "1 in N people share your mix" line (a Gaussian
            tail estimate for the top two traits; it matches the empirical share where 290 fits can resolve it).

Usage: python src/app_reference.py [--z outputs/sure_test/z_app_fit.csv] [--app app]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=Path, default=Path("outputs/sure_test/z_app_fit.csv"))
    ap.add_argument("--app", type=Path, default=Path("app"))
    a = ap.parse_args()
    m = json.loads((a.app / "data/meta.json").read_text())
    mu, Sig, Sinv = np.array(m["mu"]), np.array(m["Sig"]), np.array(m["Sinv"])
    Zdf = pd.read_csv(a.z)
    Z = Zdf[[f"z{k}" for k in range(m["k"])]].to_numpy()
    dev = (Z - mu) / np.sqrt(np.diag(Sig))
    d = Z - mu
    uniq = np.sqrt(np.einsum("ij,jk,ik->i", d, Sinv, d))
    out = dict(source=str(a.z), people=int(Zdf.pid.nunique()), n=len(Z),
               axes=[np.round(np.sort(dev[:, k]), 4).tolist() for k in range(m["k"])],
               uniq=np.round(np.sort(uniq), 4).tolist(),
               mix=dict(mean=np.round(dev.mean(0), 4).tolist(), cov=np.round(np.cov(dev.T), 4).tolist()))
    (a.app / "data/ref.json").write_text(json.dumps(out))
    print(f"ref.json: {len(Z)} fits of {out['people']} raters; uniqueness median {np.median(uniq):.2f}")


if __name__ == "__main__":
    main()
