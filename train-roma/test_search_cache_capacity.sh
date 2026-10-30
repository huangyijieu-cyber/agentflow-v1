#!/usr/bin/env bash
# Run on the development server, using the shared service's actual proxy/key.
set -euo pipefail

_capacity_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_capacity_env="${SEARCH_SERVICE_ENV_FILE:-${SEARCH_CACHE_DIR:-/home/ma-user/work/code-rl/cache}/search-service.env}"
if [[ -f "${_capacity_env}" ]]; then
    set -a
    source "${_capacity_env}"
    set +a
fi
if [[ "${SEARCH_SERVICE_USE_PROXY:-1}" == "1" ]]; then
    source "${_capacity_repo}/train-roma/enable_search_proxy.sh" --service
fi
export PYTHONPATH="${_capacity_repo}/agentflow${PYTHONPATH:+:${PYTHONPATH}}"
exec "${SEARCH_SERVICE_PYTHON:-python3}" "${_capacity_repo}/train-roma/test_search_cache_capacity.py" "$@"
