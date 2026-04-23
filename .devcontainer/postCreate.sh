#!/usr/bin/env bash
set -euo pipefail

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "[ouroboros] syncing Python dependencies"
uv sync --all-extras

if ! command -v codex >/dev/null 2>&1; then
  echo "[ouroboros] installing Codex CLI"
  npm install -g @openai/codex
else
  echo "[ouroboros] Codex CLI already present: $(command -v codex)"
fi

echo "[ouroboros] configuring Ouroboros for Codex runtime"
uv run ouroboros setup --runtime codex || true

echo "[ouroboros] bootstrap complete"
echo "[ouroboros] first time only: run codex and choose Sign in with ChatGPT"
echo "[ouroboros] then run: uv run ouroboros init start --llm-backend codex \"Build a REST API\""
