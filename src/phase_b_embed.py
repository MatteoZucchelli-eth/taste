"""Phase B step 1: embed every LAPIS image once with frozen encoders and cache to disk.

Encoders (CONTEXT Phase B decisions):
  clip    open_clip ViT-L-14-quickgelu / openai (768-d; same family as PickScore/ImageReward)
  siglip2 open_clip ViT-SO400M-14-SigLIP2-378   (1152-d)
  dinov2  timm vit_large_patch14_reg4_dinov2    (1024-d CLS token)
Output: <out>/<name>.npz with `ids` (image_id, sorted) and `emb` (float32, L2-normalised).

Usage: python src/phase_b_embed.py --data data/lapis --out embeddings [--encoders clip siglip2 dinov2]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ENCODERS = {
    "clip": ("open_clip", "ViT-L-14-quickgelu", "openai"),  # openai weights were trained with QuickGELU
    "siglip2": ("open_clip", "ViT-SO400M-14-SigLIP2-378", "webli"),
    "dinov2": ("timm", "vit_large_patch14_reg4_dinov2.lvd142m", None),
}


def load_encoder(name, device):
    kind, arch, tag = ENCODERS[name]
    if kind == "open_clip":
        import open_clip
        model, _, prep = open_clip.create_model_and_transforms(arch, pretrained=tag)
        fn = model.encode_image
    else:
        import timm
        model = timm.create_model(arch, pretrained=True, num_classes=0)
        cfg = timm.data.resolve_data_config({}, model=model)
        prep = timm.data.create_transform(**cfg)
        fn = model
    model.eval().to(device)
    return fn, prep


class Images(torch.utils.data.Dataset):
    def __init__(self, root, paths, prep):
        self.root, self.paths, self.prep = root, paths, prep

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.prep(Image.open(self.root / self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--out", type=Path, default=Path("embeddings"))
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    d = pd.read_csv(a.data / "piaa/piaa.csv", usecols=["image_id", "path"])
    img = d.drop_duplicates("image_id").sort_values("image_id")
    ids, paths = img["image_id"].to_numpy(), img["path"].tolist()
    print(f"{len(ids)} images")

    for name in a.encoders:
        f = a.out / f"{name}.npz"
        if f.exists():
            print(f"{name}: exists, skipping")
            continue
        t0 = time.time()
        fn, prep = load_encoder(name, device)
        dl = torch.utils.data.DataLoader(Images(a.data / "images", paths, prep), batch_size=a.batch,
                                         num_workers=8, pin_memory=True)
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x in dl:
                e = fn(x.to(device, non_blocking=True)).float()
                out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        emb = torch.cat(out).numpy().astype(np.float32)
        np.savez(f, ids=ids, emb=emb)
        print(f"{name}: {emb.shape} in {time.time() - t0:.0f}s")
        del fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
