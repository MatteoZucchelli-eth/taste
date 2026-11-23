"""Phase B visual sanity page: for each selected rater, the held-out images their personal critic ranks highest and
lowest, next to what the consensus critic ranks highest, each tagged with the rater's actual rating.
All images are out-of-fold (never in the training set of the critic that scored them).

Usage: python src/phase_b_page.py [--data data/lapis --out outputs/phase_b]  -> outputs/phase_b/sanity.html
"""
import argparse
import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CSS = """
:root{
  /* layout: one column of rater sections, each a stats strip over three labelled image rows */
  --bg:#f5f6f8;--surface:#ffffff;--fg:#1b1f27;--muted:#5d6573;--line:#dde1e8;
  --accent:#3346c8;--good:#1f7a4d;--bad:#b3392f;--chip:#eceff5;
  --display:"Instrument Sans",system-ui,sans-serif;--body:"Instrument Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12151b;--surface:#1a1e26;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2b313c;
  --accent:#8e9bff;--good:#5cc28e;--bad:#ef7f74;--chip:#232834;color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#12151b;--surface:#1a1e26;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2b313c;
  --accent:#8e9bff;--good:#5cc28e;--bad:#ef7f74;--chip:#232834;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:32px 64px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:flex;flex-direction:column;gap:28px}
h1{font-family:var(--display);font-size:1.9rem;margin:0;text-wrap:balance;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:68ch;margin:6px 0 0}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;color:var(--muted);font-size:.85rem}
section{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:20px;
  display:flex;flex-direction:column;gap:16px;min-width:0}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px}
.head h2{font-family:var(--display);font-size:1.25rem;margin:0}
.tag{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;padding:2px 8px;border-radius:99px;
  background:var(--chip);color:var(--muted)}
.tag.h{color:var(--bad)}
.taste{color:var(--muted);font-size:.9rem;margin:0}
.taste b{color:var(--fg);font-weight:600}
.stats{display:flex;flex-wrap:wrap;gap:6px 22px;font-family:var(--mono);font-size:.82rem;
  font-variant-numeric:tabular-nums;color:var(--muted)}
.stats span b{color:var(--fg);font-weight:500}
.row{display:flex;flex-direction:column;gap:8px;min-width:0}
.row h3{margin:0;font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}
.row h3 em{font-style:normal;font-family:var(--mono);text-transform:none;letter-spacing:0;margin-left:8px}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px}
figure{margin:0;display:flex;flex-direction:column;gap:4px}
figure img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;background:var(--chip);display:block}
figcaption{font-family:var(--mono);font-size:.75rem;font-variant-numeric:tabular-nums;color:var(--muted);
  display:flex;justify-content:space-between}
figcaption b{font-weight:500}
.hi{color:var(--good)}.lo{color:var(--bad)}
"""


def thumb(path, size=220):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def strip(rows, root, paths, q_lo, q_hi):
    out = []
    for _, r in rows.iterrows():
        cls = "hi" if r.rating >= q_hi else "lo" if r.rating <= q_lo else ""
        p = paths[r.image_id]
        out.append(f'<figure><img loading="lazy" src="{thumb(root / p)}" alt="{html.escape(p)}">'
                   f'<figcaption><b class="{cls}">{r.rating:.0f}</b><span>{r.pct:.0f}%ile</span></figcaption></figure>')
    return '<div class="strip">' + "".join(out) + "</div>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lapis"))
    ap.add_argument("--out", type=Path, default=Path("outputs/phase_b"))
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    O = pd.read_csv(a.out / "oof_scores.csv")
    S = pd.read_csv(a.out / "selected.csv")
    V = pd.read_csv(a.out / "verdict.csv").set_index("participant_id")
    piaa = pd.read_csv(a.data / "piaa/piaa.csv", usecols=["image_id", "path"]).drop_duplicates("image_id")
    paths = piaa.set_index("image_id")["path"]
    root = a.data / "images"

    secs = []
    for _, s in S[S.group == "selected"].iterrows():
        pid = s.participant_id
        g = O[O.participant_id == pid].copy()
        # scores from different folds come from different models: rank within fold, then pool the ranks
        for col in ["personal", "consensus"]:
            g[col + "_r"] = g.groupby("fold")[col].rank(pct=True)
        g["pct"] = 100 * g["rating"].rank(pct=True)
        q_lo, q_hi = g["rating"].quantile([.25, .75])
        v = V.loc[pid]
        top_p = g.sort_values("personal_r", ascending=False).head(a.k)
        top_c = g.sort_values("consensus_r", ascending=False).head(a.k)
        bot_p = g.sort_values("personal_r").head(a.k)
        tags = '<span class="tag h">heuristic flag</span>' if s.heuristic else ""
        secs.append(f"""
<section>
  <div class="head"><h2>Rater {pid}</h2><span class="tag">{int(s.n)} ratings</span>{tags}</div>
  <p class="taste">Metadata taste direction (Phase A). Likes: <b>{html.escape(str(s.taste_likes))}</b>.
  Dislikes: <b>{html.escape(str(s.taste_dislikes))}</b>.</p>
  <div class="stats">
    <span>consensus critic <b>{v.acc_consensus:.3f}</b></span>
    <span>personal critic <b>{v.acc_personal:.3f}</b></span>
    <span>gain <b>{v.gain_pts:+.1f}</b> pts [{v.gain_ci_lo:+.1f}, {v.gain_ci_hi:+.1f}]</span>
    <span>retest ceiling ≈ <b>{v.ceiling_retest:.3f}</b></span>
    <span>own critic rank <b>{int(v.rank_own_critic)}/{int(v.n_critics)}</b></span>
  </div>
  <div class="row"><h3>Their critic: top {a.k}<em>mean rating {top_p.rating.mean():.0f}</em></h3>
    {strip(top_p, root, paths, q_lo, q_hi)}</div>
  <div class="row"><h3>Consensus critic: top {a.k}<em>mean rating {top_c.rating.mean():.0f}</em></h3>
    {strip(top_c, root, paths, q_lo, q_hi)}</div>
  <div class="row"><h3>Their critic: bottom {a.k}<em>mean rating {bot_p.rating.mean():.0f}</em></h3>
    {strip(bot_p, root, paths, q_lo, q_hi)}</div>
</section>""")

    page = f"""<title>Taste Critics Sanity Check</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Instrument+Sans:wght@400;600;700&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <h1>Taste critics: what each one picks</h1>
    <p class="lede">LAPIS raters selected for distinct taste directions. Every image here was held out from the critic
    that scored it (5-fold, out-of-fold). The number under each image is the rater's own rating (0–100) and its
    percentile among that rater's ratings. If a personal critic has captured the rater's taste, its top row should score
    higher for that rater than the consensus row, and its bottom row should score low.</p>
  </header>
  <div class="legend"><span><b class="hi">green</b> = rater's top quartile</span>
    <span><b class="lo">red</b> = rater's bottom quartile</span>
    <span>accuracies are pairwise, on held-out pairs</span></div>
  {''.join(secs)}
</div>"""
    (a.out / "sanity.html").write_text(page)
    print("wrote", a.out / "sanity.html", f"{len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
