#!/usr/bin/env bash
# Start the proxy + admin dashboard (no venv activation needed).
# Usage: ./run.sh           (browser dashboard)
#        ./run.sh --native  (desktop window)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found — run ./setup.sh first." >&2
  exit 1
fi
exec .venv/bin/python -m app "$@"
