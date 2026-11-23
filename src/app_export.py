"""Export the Phase C model (fold 0, K=8, frozen CLIP) for the "this or that" web app (app/).

The app only needs per-image axis scores U(x) (N x 9), the population prior (mu, Sigma^-1) and the prior strength
tau chosen on validation raters for the 25-pair budget (z-only variant). A new person is z in R^9:
    s(x) = z . U(x),   P(x preferred to y) = sigmoid(s(x) - s(y)),   z ~ N(mu, Sigma / tau).

Writes app/data/U.bin (float32, row-major N x 9), app/data/meta.json (image paths, sizes, prior, axis stats).
Usage: python src/app_export.py [--fold 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--emb", type=Path, default=Path("embeddings"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--budget", type=int, default=25, help="budget whose validation-chosen tau is used")
    ap.add_argument("--out", type=Path, default=Path("app/data"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(Path("checkpoints/phase_c") / f"shared_K8_fold{a.fold}.pt")
    U, Z = ck["U"].numpy(), ck["Z"].numpy()  # U rows follow clip.npz ids
    ids = np.load(a.emb / "clip.npz")["ids"]
    assert len(ids) == len(U)
    mu = Z.mean(0)
    Sig = np.cov(Z.T) + 1e-4 * np.eye(Z.shape[1])  # same as phase_c_lowrank.population_prior
    cfg = pd.read_csv("outputs/phase_c/chosen_cfg.csv")
    tau = float(cfg[(cfg.fold == a.fold) & (cfg.budget == a.budget)]["zonly_tau"].iloc[0])

    paths = pd.read_csv(a.data / "piaa/piaa.csv", usecols=["image_id", "path"]).drop_duplicates("image_id") \
        .set_index("image_id")["path"]
    P = [paths[i] for i in ids]
    sizes = []
    for p in P:
        with Image.open(a.data / "images" / p) as im:
            sizes.append(im.size)

    # nudity flag from the generated captions: such images never go on the share card, and ?sfw=1 drops them
    prompts = pd.read_csv(a.data / "image_attributes/image_prompts.csv").set_index("path")["prompt"]
    pat = r"\b(?:nude|nudes|naked|erotic|nudity|breasts?|genital\w*|sexual\w*|intercourse|odalisque)\b"
    nsfw = [int(bool(pd.Series([str(prompts.get(p, ""))]).str.contains(pat, case=False).iloc[0])) for p in P]

    U.astype("<f4").tofile(a.out / "U.bin")
    cons = U @ mu
    meta = dict(
        source=f"Phase C shared model, fold {a.fold}, K=8, frozen CLIP-L/14; tau from validation at {a.budget} pairs",
        n=len(ids), k=int(U.shape[1]), tau=tau, mu=mu.tolist(), Sinv=np.linalg.inv(Sig).tolist(),
        Sig=Sig.tolist(), axis_mean=U.mean(0).tolist(), axis_sd=U.std(0).tolist(),
        cons_mean=float(cons.mean()), cons_sd=float(cons.std()),
        image_id=[int(i) for i in ids], path=P, w=[s[0] for s in sizes], h=[s[1] for s in sizes], nsfw=nsfw,
    )
    (a.out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"{len(ids)} images, K+1={U.shape[1]}, tau={tau}, mu={np.round(mu, 3).tolist()}")
    print(f"axis sd {np.round(U.std(0), 3).tolist()}  prior sd {np.round(np.sqrt(np.diag(Sig)), 3).tolist()}")


if __name__ == "__main__":
    main()
