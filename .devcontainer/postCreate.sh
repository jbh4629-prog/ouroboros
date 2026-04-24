#!/usr/bin/env bash
set -euo pipefail

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OUROBOROS_CODEX_AUTH_MODE="${OUROBOROS_CODEX_AUTH_MODE:-chatgpt}"
export OUROBOROS_CODEX_DEFAULT_MODEL="${OUROBOROS_CODEX_DEFAULT_MODEL:-gpt-5.4-mini}"
export OUROBOROS_CODEX_REASONING_EFFORT="${OUROBOROS_CODEX_REASONING_EFFORT:-low}"
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
export OUROBOROS_CODEX_DEFAULT_MODEL="${OUROBOROS_CODEX_DEFAULT_MODEL:-gpt-5.4-mini}"
export OUROBOROS_CODEX_REASONING_EFFORT="${OUROBOROS_CODEX_REASONING_EFFORT:-low}"
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

configure_codex_model() {
  mkdir -p "$HOME/.codex"
  python - <<'PY'
from pathlib import Path
import os

path = Path.home() / ".codex" / "config.toml"
text = path.read_text() if path.exists() else ""
updates = {
    "model": os.environ.get("OUROBOROS_CODEX_DEFAULT_MODEL", "gpt-5.4-mini"),
    "model_reasoning_effort": os.environ.get("OUROBOROS_CODEX_REASONING_EFFORT", "low"),
}
lines = text.splitlines()
seen = set()
out = []
for line in lines:
    stripped = line.strip()
    replaced = False
    for key, value in updates.items():
        if stripped.startswith(f"{key} ="):
            out.append(f'{key} = "{value}"')
            seen.add(key)
            replaced = True
            break
    if not replaced:
        out.append(line)

for key, value in updates.items():
    if key not in seen:
        if out and out[-1].strip():
            out.append("")
        out.append(f'{key} = "{value}"')

path.write_text("\n".join(out) + "\n")
print(f"[ouroboros] configured Codex model defaults in {path}")
PY
}

configure_ouroboros_model() {
  python - <<'PY'
from pathlib import Path
import os
import yaml

path = Path.home() / ".ouroboros" / "config.yaml"
path.parent.mkdir(parents=True, exist_ok=True)
data = yaml.safe_load(path.read_text()) if path.exists() else {}
data = data or {}
model = os.environ.get("OUROBOROS_CODEX_DEFAULT_MODEL", "gpt-5.4-mini")

data.setdefault("orchestrator", {})["runtime_backend"] = "codex"
data.setdefault("llm", {})["backend"] = "codex"
data["llm"]["qa_model"] = model
data.setdefault("clarification", {})["default_model"] = model
data.setdefault("evaluation", {})["semantic_model"] = model
data.setdefault("consensus", {})["advocate_model"] = model
data["consensus"]["devil_model"] = model
data["consensus"]["judge_model"] = model

path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
print(f"[ouroboros] configured Ouroboros Codex model defaults in {path}")
PY
}

ensure_uv
ensure_shell_auth_env

echo "[ouroboros] syncing Python dependencies"
uv sync --all-extras

ensure_codex
configure_codex_model

echo "[ouroboros] configuring Ouroboros for Codex runtime"
uv run ouroboros setup --runtime codex || true
configure_ouroboros_model

echo "[ouroboros] bootstrap complete"
echo "[ouroboros] first time only: run codex and choose Sign in with ChatGPT"
echo "[ouroboros] Codex model default: ${OUROBOROS_CODEX_DEFAULT_MODEL} (${OUROBOROS_CODEX_REASONING_EFFORT} reasoning)"
echo "[ouroboros] then run: uv run ouroboros init start --llm-backend codex \"Build a REST API\""
