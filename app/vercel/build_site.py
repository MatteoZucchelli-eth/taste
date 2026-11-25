"""Build the deployable Vercel site (separate git repo) from app/.

Usage: python3 app/vercel/build_site.py [OUT]   (default OUT = ../taste-site next to the research repo)
Copies the app, the model data and the Vercel files, and copies the LAPIS images as img/<index>.jpg (index = position in
meta.json, so URLs never contain odd characters). Images are only copied when missing or changed in size.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
REPO = APP.parent
IMAGES = REPO / "data/lapis/images"
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO.parent / "taste-site"


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(exist_ok=True)
    for f in ("index.html", "app.js", "style.css"):
        shutil.copy2(APP / f, OUT / f)
    for f in ("meta.json", "U.bin", "ref.json"):
        shutil.copy2(APP / "data" / f, OUT / "data" / f)
    for f in ("config.js", "package.json", "vercel.json", "README.md", "og.jpg"):
        shutil.copy2(HERE / f, OUT / f)
    shutil.copy2(HERE / "gitignore", OUT / ".gitignore")
    shutil.copytree(HERE / "api", OUT / "api", dirs_exist_ok=True)
    shutil.copytree(HERE / "scripts", OUT / "scripts", dirs_exist_ok=True)
    paths = json.loads((APP / "data/meta.json").read_text())["path"]
    copied = 0
    for i, p in enumerate(paths):
        src, dst = IMAGES / p, OUT / "img" / f"{i}.jpg"
        assert src.suffix.lower() in (".jpg", ".jpeg"), src
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            copied += 1
    print(f"site at {OUT}: {len(paths)} images ({copied} copied)")


if __name__ == "__main__":
    main()
