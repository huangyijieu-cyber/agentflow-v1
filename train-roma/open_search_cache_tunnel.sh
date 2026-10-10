#!/usr/bin/env bash
# Foreground SSH forward, one on every training node. Needs an authorized SSH key.
set -euo pipefail
_search_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_search_local_port="${SEARCH_CACHE_LOCAL_PORT:-8091}"
_search_remote_port="${SEARCH_SERVICE_PORT:-8091}"
_search_ssh_command=(ssh)
if [[ -n "${SEARCH_CACHE_SSH_IDENTITY_FILE:-}" ]]; then
    # Same metadata-only validation as the automatic bootstrap. Relative key
    # paths are rooted in this checkout, not the invoking shell's directory.
    _search_identity_path="$("${SEARCH_CACHE_PYTHON:-python3}" - \
        "${_search_repo_dir}/train-roma" "${SEARCH_CACHE_SSH_IDENTITY_FILE}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from search_cache_bootstrap import BootstrapError, validate_ssh_identity_file
try:
    print(validate_ssh_identity_file(sys.argv[2]))
except BootstrapError as error:
    print(f"Search cache SSH identity error: {error}", file=sys.stderr)
    sys.exit(1)
PY
    )"
    _search_ssh_command+=(-i "${_search_identity_path}" -o IdentitiesOnly=yes)
fi
exec "${_search_ssh_command[@]}" -N -T \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=yes \
    -o 'SendEnv=-*' \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -p "${SEARCH_CACHE_SSH_PORT:-31753}" \
    -L "127.0.0.1:${_search_local_port}:127.0.0.1:${_search_remote_port}" \
    "${SEARCH_CACHE_SSH_TARGET:-ma-user@7.150.11.99}"
