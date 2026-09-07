#!/usr/bin/env bash
# Switch which service is warm on the deployment box: the ASR (vLLM) server
# or the local LLM (llama.cpp) server. They can't both run at once — the
# 8GB RAM box doesn't have room for both loaded simultaneously. See
# docs/LLM_DEPLOYMENT.md for why.
#
# Usage:
#   scripts/toggle_server.sh asr   # stop the LLM, start the ASR server
#   scripts/toggle_server.sh llm   # stop the ASR server, start the LLM
#   scripts/toggle_server.sh status
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env the same way main.py does: only fills in vars not already set.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

: "${COHEREX_SERVER_SSH:?Set COHEREX_SERVER_SSH in .env (e.g. root@1.2.3.4) or export it}"

usage() {
  echo "Usage: $0 {asr|llm|status}" >&2
  exit 1
}

case "${1:-}" in
  asr)
    echo "Stopping coherex-llm, starting coherex-vllm on $COHEREX_SERVER_SSH ..."
    ssh "$COHEREX_SERVER_SSH" "systemctl stop coherex-llm; systemctl start coherex-vllm"
    echo "Done. Model load takes a minute or two — check with: $0 status"
    ;;
  llm)
    echo "Stopping coherex-vllm, starting coherex-llm on $COHEREX_SERVER_SSH ..."
    ssh "$COHEREX_SERVER_SSH" "systemctl stop coherex-vllm; systemctl start coherex-llm"
    echo "Done. Model load takes a few seconds — check with: $0 status"
    ;;
  status)
    ssh "$COHEREX_SERVER_SSH" "systemctl is-active coherex-vllm coherex-llm | paste -sd' ' - | { read -r vllm llm; echo \"coherex-vllm: \$vllm\"; echo \"coherex-llm:  \$llm\"; }"
    ;;
  *)
    usage
    ;;
esac
