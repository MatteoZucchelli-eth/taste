"""Step 2, build step 7 (CONTEXT §6g decision 10): analyse the user's blind multi-arm pairwise eval of generated images.

Input: outputs/rl/human_eval/*.json (app/eval.html; choice 0 = top image chosen), optionally filtered by --tag.
  * Bradley-Terry strength per arm (base fixed at 0; tiny ridge for identifiability when an arm never loses), 95% CI by
    bootstrap over answers.
  * Win rate of every arm against base on the shared (prompt, seed) pairs, with a Wilson CI.
  * Critic check on generated images: how often the optimised CLIP reward, the held-out SigLIP critic and the consensus
    reward predict the user's choice (scores from outputs/rl/final/per_image.csv).
-> outputs/rl/human_eval/{analysis.json,bt.csv}

Usage: python src/rl_human.py [--tag me]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT = Path("outputs/rl/human_eval")


def bt(df, arms, ridge=1e-3):
    k = {a: i for i, a in enumerate(arms)}
    w = df.apply(lambda r: k[r.top if r.choice == 0 else r.bottom], axis=1).to_numpy()
    l = df.apply(lambda r: k[r.bottom if r.choice == 0 else r.top], axis=1).to_numpy()
    base = k["base"]

    def f(th):
        s = np.insert(th, base, 0.0)
        d = s[w] - s[l]
        p = 1 / (1 + np.exp(-d))
        g = np.zeros(len(arms))
        np.add.at(g, w, -(1 - p))
        np.add.at(g, l, (1 - p))
        g = np.delete(g, base) + 2 * ridge * th
        return -np.log(p + 1e-12).sum() + ridge * (th ** 2).sum(), g

    th = minimize(f, np.zeros(len(arms) - 1), jac=True, method="L-BFGS-B").x
    return np.insert(th, base, 0.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    recs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json")) if f.name not in ("analysis.json",)]
    recs = [r for r in recs if a.tag is None or r.get("tag") == a.tag]
    df = pd.DataFrame([dict(session=r["id"], **x) for r in recs for x in r["answers"]])
    arms = sorted(set(df.top) | set(df.bottom), key=lambda x: (x != "base", x))
    s = bt(df, arms)
    rng = np.random.default_rng(0)
    B = np.array([bt(df.iloc[rng.integers(0, len(df), len(df))], arms) for _ in range(a.boot)])
    rows = []
    for i, arm in enumerate(arms):
        vs = df[((df.top == arm) & (df.bottom == "base")) | ((df.bottom == arm) & (df.top == "base"))]
        wins = int((((vs.top == arm) & (vs.choice == 0)) | ((vs.bottom == arm) & (vs.choice == 1))).sum())
        lo, hi = wilson(wins, len(vs))
        tot = df[(df.top == arm) | (df.bottom == arm)]
        tw = int((((tot.top == arm) & (tot.choice == 0)) | ((tot.bottom == arm) & (tot.choice == 1))).sum())
        rows.append(dict(arm=arm, bt=s[i], bt_lo=np.percentile(B[:, i], 2.5), bt_hi=np.percentile(B[:, i], 97.5),
                         n_vs_base=len(vs), win_vs_base=wins / len(vs) if len(vs) else np.nan, win_lo=lo, win_hi=hi,
                         n_all=len(tot), win_all=tw / len(tot) if len(tot) else np.nan))
    R = pd.DataFrame(rows).sort_values("bt", ascending=False)
    R.to_csv(OUT / "bt.csv", index=False)
    rep = dict(n_answers=len(df), sessions=len(recs), median_ms=float(df.ms.median()),
               top_rate=float((df.choice == 0).mean()), arms=R.round(4).to_dict("records"))
    f = Path("outputs/rl/final/per_image.csv")
    if f.exists():
        P = pd.read_csv(f)
        P["seed"] = P["seed"] % 1000
        P = P.set_index(["arm", "prompt_idx", "seed"])
        for c in ["r_user", "siglip_user", "r_cons", "siglip_cons"]:
            st = P.loc[list(zip(df.top, df.prompt_idx, df.seed)), c].to_numpy()
            sb = P.loc[list(zip(df.bottom, df.prompt_idx, df.seed)), c].to_numpy()
            hit = ((st > sb) == (df.choice == 0)).to_numpy()
            lo, hi = wilson(hit.sum(), len(hit))
            rep[f"critic_acc_{c}"] = [float(hit.mean()), lo, hi]
    print(R.round(3).to_string(index=False))
    print({k: v for k, v in rep.items() if k != "arms"})
    (OUT / "analysis.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
