"""Phase A for text, on LiteraryTaste (github.com/mj-storytelling/LiteraryTaste): is there consistent individual taste?

Data are pairwise: 60 raters x 100 pairs of ~140-word creative-writing snippets, every pair seen by exactly 3 raters (20 sets
of 3 raters x 100 pairs), every text in exactly one pair. "I am not sure" = tie, dropped. Text A is always text1.

Same logic as src/phase_a.py, adapted to binary choices:
  y = 1 if the rater picked text1.  Baseline (= consensus critic) = OLS of y on [intercept (side bias), mean vote of the other two
  raters, g], where g = generic prediction of the pair's mean vote from feature differences, cross-fitted by SET (the model for
  set s never sees set s, so it cannot absorb this rater's own taste).
  Taste: the baseline residual is predictable from text-feature differences out of sample (ridge on a random half of the rater's
  pairs, correlation on the other half, averaged over splits), vs a permutation null of the residual; BH-FDR over raters.
  Effect: held-out choice accuracy of baseline + taste minus baseline alone (pts).
Features: hand-built text stats (length, sentence/word length, lexical diversity, readability, punctuation, pronouns, verse form,
end rhyme, archaic words, dialogue, VADER sentiment) + source labels (5 sources, LLM-written flag); X = f(text1) - f(text2).

lambda is chosen on synthetic raters only (planted taste), before the real analysis. Null check: synthetic no-taste raters.
"Mixture" raters (each answer borrowed from a random one of the other two raters of that pair) are NOT a clean null here: sets have
only 3 raters, so a mixture is always the same two people and keeps part of their taste; it should sit between null and real.
Pre-registered gate (CONTEXT.md §7): >=50% of raters sig at BH-FDR 5% AND median held-out gain >=2 pts.
Secondary, exploratory (declared before running): pooled Stouffer z of the per-rater taste z (null: synthetic no-taste raters).

Usage: python src/phase_a_text.py --data data/literarytaste --out outputs/phase_a_text
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from phase_a import bh, colcorr, ols_fit

LAMS = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
PLANT = ["sent compound", "src_poetry"]  # planted synthetic taste: likes positive sentiment and poetry
ARCHAIC = {"thee", "thou", "thy", "thine", "hath", "doth", "art", "ye", "'tis", "tis", "o'er", "whilst", "ere", "nay", "shalt",
           "wilt", "hither", "thither", "upon"}
FIRST, SECOND, THIRD = {"i", "me", "my", "mine", "myself", "we", "us", "our"}, {"you", "your", "yours", "yourself"}, \
    {"he", "him", "his", "she", "her", "they", "them", "their"}
WORD = re.compile(r"[A-Za-z']+")
VADER = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------- text features
def syllables(w):
    w = w.lower().strip("'")
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def mattr(toks, win=50):
    if len(toks) <= win:
        return len(set(toks)) / max(len(toks), 1)
    return float(np.mean([len(set(toks[i:i + win])) / win for i in range(len(toks) - win + 1)]))


def rhyme_rate(lines):
    ends = [WORD.findall(l)[-1].lower() for l in lines if WORD.findall(l)]
    if len(ends) < 4:
        return 0.0
    def rate(k):
        pr = [(a, b) for a, b in zip(ends, ends[k:]) if a != b and len(a) >= 2 and len(b) >= 2]
        return np.mean([a[-2:] == b[-2:] for a, b in pr]) if pr else 0.0
    return float(max(rate(1), rate(2)))


def text_stats(s):
    s = s.strip()
    toks = WORD.findall(s)
    low = [t.lower() for t in toks]
    n = max(len(toks), 1)
    sents = [x for x in re.split(r"(?<=[.!?])\s+", s.replace("\n", " ")) if WORD.search(x)]
    lines = [l for l in s.split("\n") if l.strip()]
    syl = sum(syllables(t) for t in toks)
    ns = max(len(sents), 1)
    quoted = sum(len(m) for m in re.findall(r"[\"“”](.*?)[\"“”]", s, re.S))
    sc = [VADER.polarity_scores(x)["compound"] for x in sents] or [0.0]
    v = VADER.polarity_scores(s)
    return {
        "n words": len(toks), "sentence length": n / ns, "word length": np.mean([len(t) for t in toks]) if toks else 0,
        "long word rate": np.mean([len(t) >= 7 for t in toks]) if toks else 0,
        "MATTR": mattr(low), "Flesch": 206.835 - 1.015 * n / ns - 84.6 * syl / n,
        "comma rate": s.count(",") / n, "exclaim rate": s.count("!") / n, "question rate": s.count("?") / n,
        "semicolon colon rate": (s.count(";") + s.count(":")) / n, "dash rate": (s.count("—") + s.count("--")) / n,
        "dialogue frac": quoted / max(len(s), 1),
        "first person": sum(t in FIRST for t in low) / n, "second person": sum(t in SECOND for t in low) / n,
        "third person": sum(t in THIRD for t in low) / n, "archaic rate": sum(t in ARCHAIC for t in low) / n,
        "lines per word": len(lines) / n, "short line frac": np.mean([len(WORD.findall(l)) <= 8 for l in lines]) if lines else 0,
        "end rhyme": rhyme_rate(lines), "capitalised rate": sum(t[0].isupper() for t in toks[1:]) / n,
        "sent compound": v["compound"], "sent pos": v["pos"], "sent neg": v["neg"], "sent spread": float(np.std(sc)),
    }


def load(data: Path):
    t = pd.read_csv(data / "preference_tasks.csv")
    a = pd.read_csv(data / "annotated_instances.csv", low_memory=False)
    iid = pd.to_numeric(a["instance_id"], errors="coerce")
    a = a[iid.notna()].assign(iid=iid[iid.notna()].astype(int))
    yA, yB = a["preference:::Text A"].notna(), a["preference:::Text B"].notna()
    a = a.assign(y=np.where(yA, 1.0, np.where(yB, 0.0, np.nan)))
    return t, a[["user", "iid", "y"]]


def build_features(t):
    texts = pd.concat([t[["text1", "text1_source"]].set_axis(["text", "source"], axis=1),
                       t[["text2", "text2_source"]].set_axis(["text", "source"], axis=1)], ignore_index=True)
    F = pd.DataFrame([text_stats(x) for x in texts["text"]])
    F = F.apply(lambda c: pd.Series(stats.norm.ppf((c.rank() - 0.5) / len(c)), index=c.index))
    src = texts["source"].str.replace("llm_", "", regex=False)
    lab = pd.get_dummies(src, prefix="src", dtype=float)
    lab["llm written"] = texts["source"].str.startswith("llm_").astype(float)
    F = pd.concat([F, lab], axis=1)
    n = len(t)
    D = F.iloc[:n].to_numpy() - F.iloc[n:].to_numpy()
    X = pd.DataFrame(D, index=t["id"].to_numpy(), columns=F.columns)
    X = X.loc[:, X.std() > 0]
    return X / X.std()


def generic_by_set(t, a, X, lam=10.0):
    """Ridge of each pair's mean vote on feature diffs, cross-fitted by set (leave-one-set-out)."""
    mv = a.groupby("iid")["y"].mean().reindex(t["id"]).fillna(0.5)
    sets = t.set_index("id")["set_id"].reindex(mv.index).to_numpy()
    Xm, y = X.loc[mv.index].to_numpy(), mv.to_numpy()
    out = np.empty(len(y))
    for s in np.unique(sets):
        tr, te = sets != s, sets == s
        mu, ym = Xm[tr].mean(0), y[tr].mean()
        A = Xm[tr] - mu
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
        out[te] = ym + (Xm[te] - mu) @ beta
    return pd.Series(out, index=mv.index)


# ---------------------------------------------------------------- per-rater analysis
def choice_acc(y, s):
    """y in {0,1}; s = predicted P(text1). Ties (s == 0.5) count 0.5."""
    d = np.sign(s - 0.5)
    return float(np.mean(np.where(d == 0, 0.5, (d > 0) == (y > 0.5))))


def analyse(y, B, X, lam, splits, perms, rng):
    n, p = X.shape
    e = y - ols_fit(B, y)(B)
    E = np.empty((n, perms + 1))
    E[:, 0] = e
    for k in range(1, perms + 1):
        E[:, k] = rng.permutation(e)
    stat = np.zeros(perms + 1)
    gains, accm, accb = [], [], []
    I = lam * np.eye(p)
    for _ in range(splits):
        idx = rng.permutation(n)
        i1, i2 = idx[: n // 2], idx[n // 2:]
        mu = X[i1].mean(0)
        X1c, X2c = X[i1] - mu, X[i2] - mu
        W = np.linalg.solve(X1c.T @ X1c + I, X1c.T)
        E1 = E[i1] - E[i1].mean(0)
        stat += colcorr(X2c @ (W @ E1), E[i2])
        f1 = ols_fit(B[i1], y[i1])
        base2 = f1(B[i2])
        r1 = y[i1] - f1(B[i1])
        pred2 = base2 + X2c @ (W @ (r1 - r1.mean()))
        am, ab = choice_acc(y[i2], pred2), choice_acc(y[i2], base2)
        accm.append(am), accb.append(ab), gains.append(am - ab)
    stat /= splits
    null = stat[1:]
    z = (stat[0] - null.mean()) / null.std()
    return dict(taste_r=stat[0], taste_z=z, p=stats.norm.sf(z), p_perm=(1 + (null >= stat[0]).sum()) / (perms + 1),
                acc_base_half=np.mean(accb), acc_model_half=np.mean(accm), gain=100 * np.mean(gains),
                gain_sd=100 * np.std(gains))


def rater_table(t, a, X, g):
    """Per (rater, pair): y, mean vote of the other raters of that pair, their individual votes."""
    a = a.copy()
    a["set"] = a["iid"].map(t.set_index("id")["set_id"])
    votes = a.groupby("iid")["y"].apply(list)
    users = a.groupby("iid")["user"].apply(list)
    others = []
    for u, i in zip(a["user"], a["iid"]):
        others.append([v for v, w in zip(votes[i], users[i]) if w != u and not np.isnan(v)])
    a["others"] = others
    a["cons"] = [np.mean(o) if o else 0.5 for o in others]
    a["g"] = g.reindex(a["iid"]).to_numpy()
    return a


def summarise(rows):
    R = pd.DataFrame(rows)
    sig = bh(R["p"].to_numpy())[0]
    return dict(n=len(R), frac_sig_fdr05=float(sig.mean()), frac_p05=float((R["p"] < .05).mean()),
                gain_median=float(R["gain"].median()), stouffer_z=float(R["taste_z"].sum() / np.sqrt(len(R))),
                mean_taste_z=float(R["taste_z"].mean()))


def synthetic(a, X, lam, deltas, args, rng):
    """Replace each real rater by a synthetic one on the same pairs: P(text1) from the other two votes (+ planted taste)."""
    cols = [X.columns.get_loc(c) for c in PLANT]
    out = []
    for u, gr in a.groupby("user"):
        Xa = X.loc[gr["iid"]].to_numpy()
        k = np.array([len(o) for o in gr["others"]])
        s = np.array([np.sum(o) for o in gr["others"]])
        logit0 = np.log((s + 0.5) / (k - s + 0.5))
        B = gr[["cons", "g"]].to_numpy(float)
        for d in deltas:
            pr = 1 / (1 + np.exp(-(logit0 + d * Xa[:, cols].mean(1))))
            y = (rng.random(len(pr)) < pr).astype(float)
            out.append(dict(delta=d, **analyse(y, B, Xa, lam, args.splits, args.perms, rng)))
    S = pd.DataFrame(out)
    return pd.DataFrame([dict(delta=d, lam=lam, **summarise(g.to_dict("records"))) for d, g in S.groupby("delta")])


def pseudo(a, X, lam, args, rng):
    rows = []
    for u, gr in a.groupby("user"):
        ys, cs = [], []
        for o in gr["others"]:
            if len(o) == 2:
                j = rng.integers(2)
                ys.append(o[j]), cs.append(o[1 - j])
            elif len(o) == 1:
                ys.append(o[0]), cs.append(0.5)
            else:
                ys.append(np.nan), cs.append(0.5)
        ys, cs = np.array(ys), np.array(cs)
        m = ~np.isnan(ys)
        B = np.column_stack([cs, gr["g"].to_numpy()])[m]
        rows.append(analyse(ys[m], B, X.loc[gr["iid"]].to_numpy()[m], lam, args.splits, args.perms, rng))
    return summarise(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/literarytaste"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_a_text"))
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    t, a_all = load(args.data)
    X = build_features(t)
    X.to_csv(args.out / "pair_features.csv")
    g = generic_by_set(t, a_all, X)
    A = rater_table(t, a_all, X, g)
    a = A[A["y"].notna()]
    agree = a.groupby("iid")["y"].agg(lambda v: max(v.mean(), 1 - v.mean()) if len(v) == 3 else np.nan).dropna()
    print(f"{len(a)} answers ({a_all['y'].isna().mean():.1%} unsure dropped), {a.user.nunique()} raters, "
          f"{X.shape[1]} features [{time.time() - t0:.0f}s]", flush=True)

    # 1. choose lambda on synthetic raters only (max median gain at the middle planted effect, no-taste must stay ~0 sig)
    syn = pd.concat([synthetic(a, X, lam, [0.0, 1.0], args, rng) for lam in LAMS], ignore_index=True)
    print(syn.round(3).to_string(), flush=True)
    planted = syn[syn["delta"] == 1.0].set_index("lam")
    lam = float(planted["gain_median"].idxmax())
    print(f"chosen lambda = {lam}", flush=True)
    power = synthetic(a, X, lam, [0.0, 0.5, 1.0, 2.0], args, rng)
    print(power.round(3).to_string(), flush=True)

    # 2. real raters
    rows = []
    for u, gr in a.groupby("user"):
        y = gr["y"].to_numpy(float)
        B = gr[["cons", "g"]].to_numpy(float)
        acc_cons = choice_acc(y, gr["cons"].to_numpy())
        rows.append(dict(user=u, n=len(gr), frac_text1=y.mean(), acc_cons=acc_cons,
                         **analyse(y, B, X.loc[gr["iid"]].to_numpy(), lam, args.splits, args.perms, rng)))
    R = pd.DataFrame(rows)
    R["sig"], R["q"] = bh(R["p"].to_numpy())
    R.sort_values("taste_z", ascending=False).to_csv(args.out / "raters.csv", index=False)
    real = summarise(rows)
    ps = pseudo(a, X, lam, args, rng)
    print("real:", real, "\nmixture:", ps, flush=True)

    # feature view (descriptive): pooled ridge of all answers on feature diffs (what the crowd likes)
    Xa = X.loc[a["iid"]].to_numpy()
    yy = a["y"].to_numpy() - a["y"].mean()
    beta = np.linalg.solve(Xa.T @ Xa + 10 * np.eye(Xa.shape[1]), Xa.T @ yy)
    crowd = pd.Series(beta, index=X.columns).sort_values()

    summary = dict(
        verdict=dict(rule=">=50% of raters sig at BH-FDR 5% AND median held-out gain >=2 pts",
                     frac_sig=real["frac_sig_fdr05"], gain_median=real["gain_median"],
                     passed=bool(real["frac_sig_fdr05"] >= 0.5 and real["gain_median"] >= 2.0)),
        secondary_pooled=dict(real_stouffer_z=real["stouffer_z"],
                              null_stouffer_z=float(power.loc[power["delta"] == 0, "stouffer_z"].iloc[0]),
                              mixture_stouffer_z=ps["stouffer_z"], note="exploratory; null = synthetic no-taste raters"),
        n_answers=int(len(a)), n_raters=int(a.user.nunique()), n_features=int(X.shape[1]), lam=lam,
        frac_unsure=float(a_all["y"].isna().mean()),
        pair_agreement_3of3=float((agree == 1).mean()),
        acc_cons_median=float(R["acc_cons"].median()),
        side_bias_text1=float(a["y"].mean()),
        real=real, mixture_2_raters=ps, synthetic_lambda_search=syn.to_dict("records"), synthetic_power=power.to_dict("records"),
        crowd_feature_weights=dict(top=crowd.tail(6).round(4).to_dict(), bottom=crowd.head(6).round(4).to_dict()),
        runtime_s=round(time.time() - t0),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["verdict"], indent=1), json.dumps(summary["secondary_pooled"], indent=1))


if __name__ == "__main__":
    main()
