#!/usr/bin/env bash
set -euo pipefail

if ! command -v codex >/dev/null 2>&1; then
  echo "[ouroboros] Codex CLI not found. Installing @openai/codex..." >&2
  npm install -g @openai/codex
fi

exec codex "$@"
