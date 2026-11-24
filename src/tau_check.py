"""Which prior strength tau would have predicted the real app players best?

For every collected session (web sessions pulled from the Blob store + local sessions in outputs/user_taste/), fit theta = [z, b]
exactly like the app (MAP, prior z ~ N(mu, Sigma/tau), side bias b ~ N(0, 0.5^2)) on the first n learning picks, for a grid
of tau, and score the session's test picks (random pairs the fit never saw): log loss and accuracy.
Budgets n in {25, 50, 100} use every session that has at least n learning picks (truncation = what the app would have
known after n picks; the infogain pairs were chosen under tau=0.5, a small bias in favour of 0.5).
Second view: leave-one-out over ALL pairs (learning + test) of each session at its full length.

Usage: python3 src/tau_check.py [SESSION_DIR ...]   (default: ../taste-site/sessions and outputs/user_taste)
Pure Python (no numpy), so it runs on the host.
"""
import glob
import json
import math
import random
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAUS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
BUDGETS = [25, 50, 100]
SIDE_SD = 0.5


def load_model():
    m = json.loads((REPO / "app/data/meta.json").read_text())
    raw = (REPO / "app/data/U.bin").read_bytes()
    n, k = m["n"], m["k"]
    flat = struct.unpack(f"<{n * k}f", raw)
    U = [flat[i * k:(i + 1) * k] for i in range(n)]
    idx = {iid: i for i, iid in enumerate(m["image_id"])}
    return m, U, idx


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def objective(th, X, y, m0, P):
    n = len(m0)
    v = 0.5 * sum((th[i] - m0[i]) * P[i][j] * (th[j] - m0[j]) for i in range(n) for j in range(n))
    for x, t in zip(X, y):
        l = sum(a * b for a, b in zip(x, th))
        v += math.log1p(math.exp(-abs(l))) + max(l, 0) - t * l
    return v


def fit(X, y, m0, P):
    """MAP by Newton with backtracking, same as app.js fit() (fixed 2026-11-24: plain Newton diverged on real sessions)."""
    n = len(m0)
    th = m0[:]
    f0 = objective(th, X, y, m0, P)
    for _ in range(100):
        g = [sum(P[i][j] * (th[j] - m0[j]) for j in range(n)) for i in range(n)]
        H = [r[:] for r in P]
        for x, t in zip(X, y):
            p = sigmoid(sum(a * b for a, b in zip(x, th)))
            w = p * (1 - p)
            for i in range(n):
                g[i] += (p - t) * x[i]
                xi = w * x[i]
                for j in range(n):
                    H[i][j] += xi * x[j]
        step = solve(H, g)
        a = 1.0
        while a > 1e-8:
            nxt = [q - a * s for q, s in zip(th, step)]
            f1 = objective(nxt, X, y, m0, P)
            if f1 <= f0:
                break
            a /= 2
        if not f1 <= f0:
            break
        th, f0 = nxt, f1
        if max(abs(a * s) for s in step) < 1e-9:
            break
    return th


def score(th, X, y):
    ll = acc = 0.0
    for x, t in zip(X, y):
        p = min(max(sigmoid(sum(a * b for a, b in zip(x, th))), 1e-9), 1 - 1e-9)
        ll -= t * math.log(p) + (1 - t) * math.log(1 - p)
        acc += (p > 0.5) == (t == 1)
    return ll, acc


def main():
    m, U, idx = load_model()
    K = m["k"]
    dirs = sys.argv[1:] or [str(REPO.parent / "taste-site/sessions"), str(REPO / "outputs/user_taste")]
    sessions = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.json")):
            r = json.loads(Path(f).read_text())
            if r.get("partial") or not r.get("train") or not r.get("test"):
                continue
            row = lambda a: [u - v for u, v in zip(U[idx[a["top"]]], U[idx[a["bottom"]]])] + [1.0]
            tr = ([row(a) for a in r["train"]], [a["y"] for a in r["train"]])
            te = ([row(a) for a in r["test"]], [1 if a["choice"] == 0 else 0 for a in r["test"]])
            sessions.append((Path(f).stem, tr, te))
    print(f"{len(sessions)} complete sessions: " + ", ".join(f"{len(s[1][0])}+{len(s[2][0])}" for s in sessions))

    m0 = list(m["mu"]) + [0.0]

    def prior(tau):
        P = [[(tau * m["Sinv"][i][j] if i < K and j < K else 0.0) for j in range(K + 1)] for i in range(K + 1)]
        P[K][K] = 1 / SIDE_SD ** 2
        return P

    priors = {t: prior(t) for t in TAUS}
    print("\nA. Fit on the first n learning picks, score the session's random test picks")
    print("   (log loss per pick, lower = better; acc = share of test picks predicted)")
    best = {}
    for n in BUDGETS:
        use = [s for s in sessions if len(s[1][0]) >= n]
        if not use:
            continue
        nt = sum(len(s[2][0]) for s in use)
        print(f"\n   n = {n}: {len(use)} sessions, {nt} test picks")
        res = {}
        for tau in TAUS:
            per = []  # per test pick, for a paired bootstrap
            for _, (X, y), (Xt, yt) in use:
                th = fit(X[:n], y[:n], m0, priors[tau])
                for x, t in zip(Xt, yt):
                    per.append(score(th, [x], [t]))
            res[tau] = per
            ll = sum(p[0] for p in per) / nt
            acc = sum(p[1] for p in per) / nt
            print(f"     tau {tau:7.4g}: log loss {ll:.4f}   acc {acc:.3f}")
        tb = min(TAUS, key=lambda t: sum(p[0] for p in res[t]))
        # paired bootstrap over test picks: how often is each tau the best?
        rng = random.Random(0)
        wins = {t: 0 for t in TAUS}
        B = 2000
        for _ in range(B):
            ii = [rng.randrange(nt) for _ in range(nt)]
            w = min(TAUS, key=lambda t: sum(res[t][i][0] for i in ii))
            wins[w] += 1
        d05 = [res[0.5][i][0] - res[tb][i][0] for i in range(nt)]
        mean = sum(d05) / nt
        se = math.sqrt(sum((v - mean) ** 2 for v in d05) / (nt - 1) / nt)
        print(f"     best tau {tb:g}; log-loss gain vs 0.5: {mean:.4f} +/- {se:.4f} (SE) per pick;"
              f" bootstrap P(best): " + ", ".join(f"{t:g}:{wins[t] / B:.2f}" for t in TAUS if wins[t]))
        best[n] = tb

    print("\nB. Leave-one-out over all pairs of each session (full length; learning pairs are infogain, test pairs random)")
    for name, (X, y), (Xt, yt) in sessions:
        XA, yA = X + Xt, y + yt
        line = []
        for tau in TAUS:
            ll = 0.0
            for i in range(len(XA)):
                th = fit(XA[:i] + XA[i + 1:], yA[:i] + yA[i + 1:], m0, priors[tau])
                ll += score(th, [XA[i]], [yA[i]])[0]
            line.append(ll / len(XA))
        tb = TAUS[line.index(min(line))]
        print(f"   {name[:24]:24s} {len(XA):4d} pairs: best tau {tb:g}  "
              + " ".join(f"{t:g}:{v:.3f}" for t, v in zip(TAUS, line)))


if __name__ == "__main__":
    main()
