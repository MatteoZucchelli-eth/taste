"""Local server for the taste app, reachable from a phone on the same Wi-Fi.

Serves app/ at / and the LAPIS images (data/lapis/images) at /img/. Standard library only (no container needed).
POST /api/session stores a finished (or checkpointed) session as outputs/user_taste/<id>.json.
Long data-collection session (step 2 reward): http://localhost:8000/?train=100&test=20&mix=1&tag=<name>
Usage: python3 app/serve.py [--port 8000]   then open http://<this machine's LAN IP>:8000 on the iPhone.
"""
import argparse
import functools
import http.server
import json
import re
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT.parent / "data/lapis/images"
SESSIONS = ROOT.parent / "outputs/user_taste"
MAX_BODY = 2_000_000


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/img/"):
            target = (IMAGES / clean[len("/img/"):]).resolve()
            return str(target) if target.is_relative_to(IMAGES) else str(IMAGES / "__missing__")
        return super().translate_path(path)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/session":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BODY:
            return self.send_error(413)
        try:
            rec = json.loads(self.rfile.read(n))
            sid = re.sub(r"[^\w-]", "", str(rec.get("id") or ""))[:80]
            assert sid and isinstance(rec.get("train"), list)
        except Exception:
            return self.send_error(400)
        SESSIONS.mkdir(parents=True, exist_ok=True)
        tmp = SESSIONS / f".{sid}.tmp"
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(SESSIONS / f"{sid}.json")
        print(f"saved {sid}: {len(rec['train'])} train, {len(rec.get('test', []))} test, partial={rec.get('partial')}", flush=True)
        self.send_response(204)
        self.end_headers()

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
    ap.add_argument("--host", default="0.0.0.0", help="127.0.0.1 = reachable only through an SSH tunnel")
    a = ap.parse_args()
    assert IMAGES.is_dir(), f"missing {IMAGES}"
    assert (ROOT / "data/U.bin").exists(), "run: docker compose run --rm taste python src/app_export.py"
    srv = http.server.ThreadingHTTPServer((a.host, a.port), functools.partial(Handler, directory=str(ROOT)))
    print(f"Taste app on http://{lan_ip()}:{a.port}  (local: http://localhost:{a.port})  Ctrl+C to stop")
    srv.serve_forever()


if __name__ == "__main__":
    main()
