#!/usr/bin/env bash
set -euo pipefail

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PATH="$HOME/.local/bin:$PATH"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "[ouroboros] uv already present: $(command -v uv)"
    return 0
  fi

  echo "[ouroboros] uv not found; installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    echo "[ouroboros] uv install failed or uv is not on PATH" >&2
    exit 1
  fi
}

ensure_codex() {
  if command -v codex >/dev/null 2>&1; then
    echo "[ouroboros] Codex CLI already present: $(command -v codex)"
    return 0
  fi

  echo "[ouroboros] installing Codex CLI"
  npm install -g @openai/codex
}

ensure_uv

echo "[ouroboros] syncing Python dependencies"
uv sync --all-extras

ensure_codex

echo "[ouroboros] configuring Ouroboros for Codex runtime"
uv run ouroboros setup --runtime codex || true

echo "[ouroboros] bootstrap complete"
echo "[ouroboros] first time only: run codex and choose Sign in with ChatGPT"
echo "[ouroboros] then run: uv run ouroboros init start --llm-backend codex \"Build a REST API\""
