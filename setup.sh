#!/usr/bin/env bash
# One-shot setup for the Grounded RPG proxy (LightRAG backend).
# Creates a venv, installs deps, and scaffolds your .env.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies (this can take a few minutes)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example"
else
  echo "==> .env already exists, leaving it untouched"
fi

cat <<'EOF'

✅ Setup complete.

Next steps:
  1. Start it (opens the dashboard in your browser):
       ./run.sh

  2. Add your FREE Gemini key — get one (no credit card) at
       https://aistudio.google.com/apikey
     then paste it in the dashboard: Settings → Gemini key → Save.

  3. Add a game: name it, drop in your rulebook PDFs, click Ingest.

  4. In PUM → AI Settings → Text Generation → "Other (OpenAI-compatible)":
       Base URL: http://localhost:8000/v1
       Model:    <your game name>   (e.g. forbidden-lands)

  Full instructions + optional quality upgrades are in the README.
EOF
