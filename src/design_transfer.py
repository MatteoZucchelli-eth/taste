"""Transfer test: does the LAPIS-trained personal critic predict the user's taste outside artworks?

Nothing is trained before the test. The user's model is frozen (checkpoints/rl/user.pt, fitted on the 100-pick LAPIS
session) and its predictions for every test pair are written and hashed BEFORE the user sees any pair (pre-registration).

Domains (--domain)
  crello  graphic-design templates, HF `cyberagent/crello` (CDLA-Permissive-2.0), test shards 0-1 in data/crello/raw/.
          Run 2026-11-24: personal 0.58, crowd 0.48 (CONTEXT §6i). The user found the designs bland -> weak evidence.
  products  home products from Amazon Berkeley Objects (ABO, CC BY 4.0), main catalogue photo of each listing:
          chairs, sofas, tables, rugs, lamps (+ light fixtures). Pairs are like-for-like (chair vs chair, ...), 20 per
          category; near-duplicates (colour variants of one model, CLIP cos > 0.92) removed. (Unsplash Lite was tried
          first and dropped: too few real interiors, mostly lifestyle shots.)

Stages
  prep     crello: extract previews (aspect 0.75-1.34); products: sample listings per category -> download ->
           de-duplicate; sample 2*N images, pair them (each image once, sides random)
           -> OUT/pairs.json, app/data/<domain>_manifest.json (no predictions) and OUT/contact.jpg for a visual check.
  predict  score each image with 4 frozen LAPIS models and lock predictions -> OUT/predictions.json + predictions.sha256.
           Models: personal CLIP (z_user . U(f)), crowd CLIP (z = population mu), personal SigLIP held-out critic,
           SigLIP consensus. Also stores each image's 9 axis scores and CLIP embedding (OUT/clip_emb.npz) for the
           in-domain analysis below.
  analyze  zero-shot: accuracy per model (Wilson CI, binomial p vs chance), exact McNemar personal vs crowd, accuracy by
           confidence tercile. In-domain (products; pre-registered 2026-11-24 before the test): 200 random 50/50 splits of
           the user's answered pairs; fit on one half, test on the other, mean accuracy over splits, for
             (a) frozen LAPIS z (same test halves, reference)
             (b) LAPIS axes, z refit from the population prior N(mu, Sig/tau), tau=0.5 (Phase C choice at 50 pairs)
             (c) LAPIS axes, z refit from the user's LAPIS z as prior mean (same Sig/tau)
             (d) CLIP-only personal logistic head (no LAPIS parts), L2 chosen by inner 5-fold CV
           -> OUT/summary.json

Usage (in Docker):  python src/design_transfer.py --domain products --stage prep|predict|analyze [--n 100]
"""
import argparse
import hashlib
import io
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np

DOMAINS = {
    "crello": dict(raw=Path("data/crello/raw"), img=Path("data/crello/img"), out=Path("outputs/design_transfer"),
                   manifest=Path("app/data/design_manifest.json"), img_base="design/", aspect=(0.75, 1.34)),
    "products": dict(raw=Path("data/abo/raw"), img=Path("data/abo/img"), out=Path("outputs/products_transfer"),
                     manifest=Path("app/data/products_manifest.json"), img_base="products/"),
}
SEED = 20261124
ABO = "https://amazon-berkeley-objects.s3.amazonaws.com"
PRODUCT_CATS = {"chair": ["CHAIR"], "sofa": ["SOFA"], "table": ["TABLE"], "rug": ["RUG"],
                "lamp": ["LAMP", "LIGHT_FIXTURE"]}


def sys_path_fix():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def to_rgb(im):
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = __import__("PIL.Image", fromlist=["Image"]).new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def _load_images(img_dir, ids):
    import torch
    from PIL import Image
    for i in ids:
        im = np.asarray(Image.open(img_dir / f"{i}.jpg").convert("RGB"), dtype=np.float32) / 255.0
        yield torch.from_numpy(im).permute(2, 0, 1)[None]


# ---------------------------------------------------------------- prep
def prep_crello(D, a):
    import pyarrow.parquet as pq
    from PIL import Image
    rows = []
    for f in sorted(D["raw"].glob("test-*.parquet")):
        t = pq.read_table(f, columns=["id", "group", "format", "canvas_width", "canvas_height", "category", "preview"])
        for r in t.to_pylist():
            w, h = r["canvas_width"], r["canvas_height"]
            if not 0.75 <= w / h <= 1.34:
                continue
            p = D["img"] / f"{r['id']}.jpg"
            if not p.exists():
                im = to_rgb(Image.open(io.BytesIO(r["preview"]["bytes"])))
                im.thumbnail((1080, 1080))
                im.save(p, quality=88)
            rows.append(dict(id=r["id"], group=r["group"], format=r["format"], category="design", w=w, h=h))
    rng = np.random.default_rng(SEED)
    rng.shuffle(rows)
    return rows, {"design": rows}  # Crello test designs are distinct templates (`group` is a 6-way category)


def prep_products(D, a, per_cat_candidates=600, max_cos=0.92):
    import csv
    import glob
    import gzip
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    sys_path_fix()
    from rl_critic import DEV, ClipEncoder

    want = {t: c for c, ts in PRODUCT_CATS.items() for t in ts}
    lst = {c: [] for c in PRODUCT_CATS}
    for f in sorted(glob.glob(str(D["raw"] / "listings/metadata/*.json.gz"))):
        for line in gzip.open(f, "rt"):
            d = json.loads(line)
            t = (d.get("product_type") or [{}])[0].get("value")
            if t in want and d.get("main_image_id"):
                name = next((x["value"] for x in d.get("item_name", []) if x["language_tag"].startswith("en")), "")
                lst[want[t]].append(dict(id=d["main_image_id"], item=d["item_id"], name=name))
    paths = {r["image_id"]: (r["path"], int(r["width"]), int(r["height"]))
             for r in csv.DictReader(gzip.open(D["raw"] / "images.csv.gz", "rt"))}
    rng = np.random.default_rng(SEED)
    cand = []
    for c, L in lst.items():
        L = sorted({r["id"]: r for r in L if r["id"] in paths and min(paths[r["id"]][1:]) >= 500}.values(),
                   key=lambda r: r["id"])
        for j in rng.permutation(len(L))[:per_cat_candidates]:
            cand.append(dict(L[j], category=c))

    def get(r):
        p = D["img"] / f"{r['id']}.jpg"
        if p.exists():
            return True
        try:
            req = urllib.request.Request(f"{ABO}/images/original/{paths[r['id']][0]}",
                                         headers={"User-Agent": "research-image-fetch/1.0"})
            im = to_rgb(Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=60).read())))
            im.thumbnail((1080, 1080))
            tmp = p.with_suffix(".part")
            im.save(tmp, "JPEG", quality=88)
            tmp.replace(p)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(get, cand))
    cand = [r for r, o in zip(cand, ok) if o]
    import open_clip
    import torch
    clip = ClipEncoder()
    E = np.stack([clip(x.to(DEV))[0].cpu().numpy() for x in _load_images(D["img"], [r["id"] for r in cand])])
    # zero-shot check that the main photo shows the product (ABO main images are sometimes fabric swatches,
    # cushions, collages or bulbs)
    labels = [f"a product photo of a {c}" for c in PRODUCT_CATS] + [
        "a close-up of a fabric swatch or texture sample", "a seat cushion or pillow",
        "a collage or diagram of many small products", "a light bulb"]
    tok = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    with torch.no_grad():
        T = torch.nn.functional.normalize(clip.model.encode_text(tok(labels).to(DEV)).float(), dim=-1).cpu().numpy()
    top = (E @ T.T).argmax(1)
    cats = list(PRODUCT_CATS)
    n0 = len(cand)
    cand, E = [r for r, t in zip(cand, top) if t < len(cats) and cats[t] == r["category"]], E[[
        t < len(cats) and cats[t] == r["category"] for r, t in zip(cand, top)]]
    print(f"zero-shot check kept {len(cand)} of {n0}")
    by, keep_e = {c: [] for c in PRODUCT_CATS}, []
    for r, e in zip(cand, E):  # greedy de-dup in the (random) candidate order: drop colour variants / repeats
        if keep_e and max(float(e @ k) for k in keep_e) > max_cos:
            continue
        keep_e.append(e)
        by[r["category"]].append(r)
    print({c: len(v) for c, v in by.items()}, f"of {len(cand)} downloaded")
    return [r for v in by.values() for r in v], by


def stage_prep(D, a):
    from PIL import Image
    D["img"].mkdir(parents=True, exist_ok=True)
    rows, by = (prep_crello if a.domain == "crello" else prep_products)(D, a)
    rng = np.random.default_rng(SEED + 1)
    cats = list(by)
    per = {c: a.n // len(cats) + (1 if k < a.n % len(cats) else 0) for k, c in enumerate(cats)}
    pick, pairs = [], []
    for c in cats:
        assert len(by[c]) >= 2 * per[c], (c, len(by[c]), per[c])
        sel = by[c][: 2 * per[c]]
        pick += sel
        for j in range(per[c]):
            x, y = sel[2 * j], sel[2 * j + 1]
            if rng.random() < 0.5:
                x, y = y, x
            pairs.append(dict(k=len(pairs), category=c, top=x["id"], bottom=y["id"]))
    D["out"].mkdir(parents=True, exist_ok=True)
    (D["out"] / "pairs.json").write_text(json.dumps(dict(seed=SEED, n_candidates=len(rows),
                                                         per_category={c: len(v) for c, v in by.items()},
                                                         designs=pick, pairs=pairs), indent=1, default=str))
    D["manifest"].write_text(json.dumps(dict(source=f"{a.domain}-transfer", imgBase=D["img_base"], pairs=pairs)))
    # contact sheet of all test images (for a visual check before locking predictions)
    th, cols = 128, 20
    sheet = Image.new("RGB", (cols * th, ((len(pick) + cols - 1) // cols) * th), "white")
    for k, r in enumerate(pick):
        im = Image.open(D["img"] / f"{r['id']}.jpg").convert("RGB")
        im.thumbnail((th, th))
        sheet.paste(im, ((k % cols) * th, (k // cols) * th))
    sheet.save(D["out"] / "contact.jpg", quality=85)
    print(f"pairs {len(pairs)} ({per}); contact sheet {D['out'] / 'contact.jpg'}")


# ---------------------------------------------------------------- predict
def stage_predict(D, a):
    import torch
    sys_path_fix()
    from rl_critic import DEV, ClipEncoder, SiglipCritic, TasteReward, load_user

    P = json.loads((D["out"] / "pairs.json").read_text())
    ids = sorted({d["id"] for d in P["designs"]})
    u = load_user()
    clip = ClipEncoder()
    pers, crowd = TasteReward(u["z_user"], clip), TasteReward(u["z_cons"], clip)
    sig = SiglipCritic()
    S = {m: {} for m in ("personal_clip", "crowd_clip", "personal_siglip", "consensus_siglip")}
    axes, embs = {}, []
    for i, x in zip(ids, _load_images(D["img"], ids)):
        x = x.to(DEV)
        e = clip(x)
        es = sig.enc.image(x)
        S["personal_clip"][i] = float(pers(emb=e))
        S["crowd_clip"][i] = float(crowd(emb=e))
        S["personal_siglip"][i] = float(sig(emb=es))
        S["consensus_siglip"][i] = float(sig.consensus(es))
        axes[i] = pers.axes(emb=e)[0].tolist()
        embs.append(e[0].cpu().numpy())
    np.savez(D["out"] / "clip_emb.npz", ids=np.array(ids), emb=np.stack(embs))
    pred = []
    for p in P["pairs"]:
        rec = dict(k=p["k"], category=p.get("category"), top=p["top"], bottom=p["bottom"])
        for m, s in S.items():
            d = s[p["top"]] - s[p["bottom"]]
            rec[m] = dict(pick_top=bool(d > 0), margin=d)
        pred.append(rec)
    E = np.load("embeddings/clip.npz")
    with torch.no_grad():
        L = pers.trunk(torch.as_tensor(E["emb"][:4000], dtype=torch.float32, device=DEV)).cpu().numpy()
    A = np.array([axes[i] for i in ids])
    lo, hi = np.percentile(L, 1, 0), np.percentile(L, 99, 0)
    ood = dict(frac_outside_1_99_per_axis=((A < lo) | (A > hi)).mean(0).round(3).tolist(),
               axis_mean_in_lapis_sd=((A.mean(0) - L.mean(0)) / L.std(0)).round(2).tolist())
    body = json.dumps(dict(model="checkpoints/rl/user.pt + trunk.pt + siglip_critic.pt (frozen, LAPIS-trained)",
                           scores=S, axes=axes, predictions=pred, ood=ood), indent=1)
    (D["out"] / "predictions.json").write_text(body)
    h = hashlib.sha256(body.encode()).hexdigest()
    (D["out"] / "predictions.sha256").write_text(h + "\n")
    agree = np.mean([r["personal_clip"]["pick_top"] == r["crowd_clip"]["pick_top"] for r in pred])
    print(f"locked {len(pred)} predictions, sha256 {h}")
    print(f"personal vs crowd agree on {agree:.0%} of pairs; OOD: {json.dumps(ood)}")


# ---------------------------------------------------------------- analyze
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def binom_p_ge(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def fit_z(Ud, y, m, Sinv, tau, side_sd=0.5, iters=100):
    """Damped Newton MAP of [z, b] under N(m, Sig/tau) and b ~ N(0, side_sd^2) (as rl_setup.fit_side, + backtracking)."""
    K = Ud.shape[1]
    X = np.hstack([Ud, np.ones((len(Ud), 1))])
    mm = np.concatenate([m, [0.0]])
    Pr = np.zeros((K + 1, K + 1))
    Pr[:K, :K] = tau * Sinv
    Pr[K, K] = 1 / side_sd ** 2

    def obj(t):
        s = X @ t
        return 0.5 * (t - mm) @ Pr @ (t - mm) + np.sum(np.logaddexp(0, s) - y * s)

    th = mm.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ th))
        g = Pr @ (th - mm) + X.T @ (p - y)
        H = Pr + (X * (p * (1 - p))[:, None]).T @ X
        step, f0, lr = np.linalg.solve(H, g), obj(th), 1.0
        while obj(th - lr * step) > f0 and lr > 1e-6:
            lr /= 2
        th = th - lr * step
        if np.abs(lr * step).max() < 1e-10:
            break
    return th[:K], th[K]


def fit_logreg(X, y, lam, iters=100):
    """L2 logistic regression on pair differences (no intercept, pairs are side-symmetrised by the caller)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def in_domain(pred_body, ans, D, n_splits=200):
    import torch
    sys_path_fix()
    from rl_critic import load_user
    u = load_user()
    Sig = np.asarray(u["Sig"], float)
    Sinv = np.asarray(u["Sinv"], float)
    zu, mu = np.asarray(u["z_user"], float), np.asarray(u["z_cons"], float)
    ax = {k: np.array(v) for k, v in pred_body["axes"].items()}
    E = np.load(D["out"] / "clip_emb.npz")
    emb = dict(zip(E["ids"], E["emb"].astype(np.float64)))
    Ud = np.array([ax[x["top"]] - ax[x["bottom"]] for x in ans])
    Ed = np.array([emb[x["top"]] - emb[x["bottom"]] for x in ans])
    y = np.array([x["choice"] == 0 for x in ans], float)
    n = len(y)
    rng = np.random.default_rng(SEED)
    lams = [0.1, 0.3, 1, 3, 10, 30]
    acc = {m: [] for m in ("a_frozen_lapis_z", "b_refit_from_crowd_prior", "c_refit_from_your_lapis_z", "d_clip_only")}
    for _ in range(n_splits):
        perm = rng.permutation(n)
        tr, te = perm[: n // 2], perm[n // 2:]
        hit = lambda s: float(((s[te] > 0) == (y[te] == 1)).mean())  # noqa: E731
        acc["a_frozen_lapis_z"].append(hit(Ud @ zu))
        zb, _ = fit_z(Ud[tr], y[tr], mu, Sinv, 0.5)
        acc["b_refit_from_crowd_prior"].append(hit(Ud @ zb))
        zc, _ = fit_z(Ud[tr], y[tr], zu, Sinv, 0.5)
        acc["c_refit_from_your_lapis_z"].append(hit(Ud @ zc))
        Xs, ys = np.vstack([Ed[tr], -Ed[tr]]), np.concatenate([y[tr], 1 - y[tr]])  # side-symmetrised
        folds = np.arange(len(tr)) % 5
        rng.shuffle(folds)
        cv = []
        for lam in lams:
            ok = 0
            for f in range(5):
                itr, ite = tr[folds != f], tr[folds == f]
                w = fit_logreg(np.vstack([Ed[itr], -Ed[itr]]), np.concatenate([y[itr], 1 - y[itr]]), lam)
                ok += ((Ed[ite] @ w > 0) == (y[ite] == 1)).sum()
            cv.append(ok)
        w = fit_logreg(Xs, ys, lams[int(np.argmax(cv))])
        acc["d_clip_only"].append(hit(Ed @ w))
    return {m: dict(mean=float(np.mean(v)), sd_over_splits=float(np.std(v)), n_test_per_split=n - n // 2)
            for m, v in acc.items()}


def stage_analyze(D, a):
    body = (D["out"] / "predictions.json").read_text()
    h = hashlib.sha256(body.encode()).hexdigest()
    assert h == (D["out"] / "predictions.sha256").read_text().strip(), "predictions changed after locking"
    PB = json.loads(body)
    pred = {r["k"]: r for r in PB["predictions"]}
    files = sorted((D["out"] / "answers").glob("*.json"))
    recs = [json.loads(f.read_text()) for f in files]
    recs = [r for r in recs if not r.get("partial")] or recs
    rec = max(recs, key=lambda r: len(r["answers"]))
    ans = rec["answers"]
    models = ["personal_clip", "crowd_clip", "personal_siglip", "consensus_siglip"]
    y = np.array([x["choice"] == 0 for x in ans])  # True = top chosen
    hit = {m: np.array([pred[x["k"]][m]["pick_top"] for x in ans]) == y for m in models}
    n = len(ans)
    res = dict(domain=a.domain, session=rec["id"], n=n, predictions_sha256=h,
               median_rt_s=float(np.median([x["ms"] for x in ans]) / 1000), top_rate=float(y.mean()))
    for m in models:
        k = int(hit[m].sum())
        res[m] = dict(acc=k / n, ci95=wilson(k, n), p_vs_chance=binom_p_ge(k, n))
    b = int((hit["personal_clip"] & ~hit["crowd_clip"]).sum())
    c = int((~hit["personal_clip"] & hit["crowd_clip"]).sum())
    res["mcnemar_personal_vs_crowd"] = dict(personal_only=b, crowd_only=c,
                                            p_two_sided=min(1.0, 2 * binom_p_ge(max(b, c), b + c)) if b + c else 1.0)
    mg = np.abs([pred[x["k"]]["personal_clip"]["margin"] for x in ans])
    t = np.quantile(mg, [1 / 3, 2 / 3])
    res["personal_clip_by_confidence"] = {
        name: dict(n=int(sel.sum()), acc=float(hit["personal_clip"][sel].mean()))
        for name, sel in (("low", mg <= t[0]), ("mid", (mg > t[0]) & (mg <= t[1])), ("high", mg > t[1]))}
    cat = np.array([pred[x["k"]].get("category") or "all" for x in ans])
    res["by_category"] = {c: {m: float(hit[m][cat == c].mean()) for m in models} | dict(n=int((cat == c).sum()))
                          for c in sorted(set(cat))}
    if "axes" in PB and (D["out"] / "clip_emb.npz").exists():
        res["in_domain_cv"] = in_domain(PB, ans, D)
    (D["out"] / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="crello")
    ap.add_argument("--stage", choices=["prep", "predict", "analyze"], required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    D = DOMAINS[a.domain]
    dict(prep=stage_prep, predict=stage_predict, analyze=stage_analyze)[a.stage](D, a)


if __name__ == "__main__":
    main()
