"""Page of the 9 Phase C shared-model axes (fold 0, K=8): for each axis, the LAPIS images with the highest and lowest
score. Usage: python src/axes_page.py [--fold 0 --n 8]  -> outputs/phase_c/axes.html
"""
import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

CSS = """
:root{
  /* layout: one column of axis sections, each two labelled thumbnail strips */
  --bg:#f4f5f7;--surface:#ffffff;--fg:#1c2028;--muted:#5f6775;--line:#dde1e8;--chip:#eceff4;--hi:#1f6f8b;--lo:#9a4a2c;
  --display:"Space Grotesk",system-ui,sans-serif;--body:"Inter",system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#121419;--surface:#1a1d24;--fg:#e7e9ee;--muted:#9aa2b1;--line:#2b303a;--chip:#232731;--hi:#6cc3e0;--lo:#ec9c7c;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#121419;--surface:#1a1d24;--fg:#e7e9ee;--muted:#9aa2b1;--line:#2b303a;--chip:#232731;--hi:#6cc3e0;--lo:#ec9c7c;
  color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:32px 64px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:flex;flex-direction:column;gap:24px}
h1{font-family:var(--display);font-size:1.8rem;margin:0;letter-spacing:-.01em;text-wrap:balance}
section{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;
  display:flex;flex-direction:column;gap:14px;min-width:0}
h2{font-family:var(--display);font-size:1.2rem;margin:0}
.row{display:flex;flex-direction:column;gap:8px;min-width:0}
h3{margin:0;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;font-weight:600}
h3.hi{color:var(--hi)} h3.lo{color:var(--lo)}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:8px}
.strip img{width:100%;max-width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;background:var(--chip);display:block}
"""


def thumb(path, size=240):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--emb", type=Path, default=Path("embeddings"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_c/axes.html"))
    a = ap.parse_args()
    U = torch.load(Path("checkpoints/phase_c") / f"shared_K8_fold{a.fold}.pt")["U"].numpy()
    ids = np.load(a.emb / "clip.npz")["ids"]  # row order of U
    paths = pd.read_csv(a.data / "piaa/piaa.csv", usecols=["image_id", "path"]).drop_duplicates("image_id") \
        .set_index("image_id")["path"]
    secs = []
    for k in range(U.shape[1]):
        o = np.argsort(-U[:, k])
        rows = []
        for label, cls, idx in [("Highest", "hi", o[:a.n]), ("Lowest", "lo", o[::-1][:a.n])]:
            imgs = "".join(f'<img src="{thumb(a.data / "images" / paths[ids[i]])}" alt="" loading="lazy">' for i in idx)
            rows.append(f'<div class="row"><h3 class="{cls}">{label}</h3><div class="strip">{imgs}</div></div>')
        secs.append(f'<section><h2>Axis {k}</h2>{"".join(rows)}</section>')
    page = ('<title>Taste Axes</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600'
            '&family=Space+Grotesk:wght@500;600&display=swap">\n'
            f'<style>{CSS}</style>\n<div class="wrap"><h1>Taste Axes</h1>{"".join(secs)}</div>\n')
    a.out.write_text(page)
    print(a.out, f"{len(page) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
