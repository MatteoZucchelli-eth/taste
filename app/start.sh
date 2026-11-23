#!/usr/bin/env bash
# Start the "This or That" taste app (Ctrl+C to stop). Usage: bash app/start.sh [port]
# Long data-collection session: http://localhost:<port>/?train=100&test=20&mix=1&tag=me
# Sessions are saved to outputs/user_taste/<id>.json.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 app/serve.py --port "${1:-8000}"
