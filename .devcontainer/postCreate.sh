#!/usr/bin/env bash
set -euo pipefail

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OUROBOROS_CODEX_AUTH_MODE="${OUROBOROS_CODEX_AUTH_MODE:-chatgpt}"
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

ensure_shell_auth_env() {
  local bashrc="$HOME/.bashrc"
  local marker_begin="# >>> ouroboros codex chatgpt auth >>>"
  local marker_end="# <<< ouroboros codex chatgpt auth <<<"
  local tmp

  tmp="$(mktemp)"
  if [ -f "$bashrc" ]; then
    awk \
      -v begin="$marker_begin" \
      -v end="$marker_end" \
      'BEGIN {skip=0} $0 == begin {skip=1; next} $0 == end {skip=0; next} skip == 0 {print}' \
      "$bashrc" > "$tmp"
  fi

  cat >> "$tmp" <<'EOF'
# >>> ouroboros codex chatgpt auth >>>
# Codespaces default: use Codex CLI ChatGPT sign-in, not API-key env auth.
export OUROBOROS_CODEX_AUTH_MODE="${OUROBOROS_CODEX_AUTH_MODE:-chatgpt}"
if [ "${OUROBOROS_CODEX_AUTH_MODE}" = "chatgpt" ]; then
  unset OPENAI_API_KEY
  unset OPENAI_BASE_URL
  unset OPENAI_ORG_ID
  unset OPENAI_ORGANIZATION
  unset OPENAI_PROJECT
fi
# <<< ouroboros codex chatgpt auth <<<
EOF

  mv "$tmp" "$bashrc"
  echo "[ouroboros] installed Codex ChatGPT auth shell defaults in $bashrc"
}

ensure_uv
ensure_shell_auth_env

echo "[ouroboros] syncing Python dependencies"
uv sync --all-extras

ensure_codex

echo "[ouroboros] configuring Ouroboros for Codex runtime"
uv run ouroboros setup --runtime codex || true

echo "[ouroboros] bootstrap complete"
echo "[ouroboros] first time only: run codex and choose Sign in with ChatGPT"
echo "[ouroboros] then run: uv run ouroboros init start --llm-backend codex \"Build a REST API\""
