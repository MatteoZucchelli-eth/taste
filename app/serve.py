"""Local server for the taste app, reachable from a phone on the same Wi-Fi.

Serves app/ at / and the LAPIS images (data/lapis/images) at /img/. Standard library only (no container needed).
Usage: python3 app/serve.py [--port 8000]   then open http://<this machine's LAN IP>:8000 on the iPhone.
"""
import argparse
import functools
import http.server
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT.parent / "data/lapis/images"


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/img/"):
            target = (IMAGES / clean[len("/img/"):]).resolve()
            return str(target) if target.is_relative_to(IMAGES) else str(IMAGES / "__missing__")
        return super().translate_path(path)

    def end_headers(self):
        cache = "public, max-age=86400" if self.path.startswith("/img/") else "no-store"
        self.send_header("Cache-Control", cache)
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the terminal quiet except errors
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    assert IMAGES.is_dir(), f"missing {IMAGES}"
    assert (ROOT / "data/U.bin").exists(), "run: docker compose run --rm taste python src/app_export.py"
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", a.port), functools.partial(Handler, directory=str(ROOT)))
    print(f"Taste app on http://{lan_ip()}:{a.port}  (local: http://localhost:{a.port})  Ctrl+C to stop")
    srv.serve_forever()


if __name__ == "__main__":
    main()
