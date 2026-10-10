#!/usr/bin/env bash
# Source on EVERY training node before starting Ray and the rollout server.
_agentflow_enable_search_cache() {
    local repo_dir env_file
    repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    env_file="${SEARCH_CACHE_ENV_FILE:-${repo_dir}/train-roma/search-cache.local.env}"
    if [[ -f "${env_file}" ]]; then
        set -a
        source "${env_file}"
        set +a
    fi
    if [[ -z "${SEARCH_CACHE_BASE_URL:-}" || -z "${SEARCH_CACHE_TOKEN:-}" ]]; then
        echo "Set SEARCH_CACHE_BASE_URL and SEARCH_CACHE_TOKEN (or use ${env_file})." >&2
        return 1
    fi
    export SEARCH_CACHE_ENABLED=1
    export SEARCH_CACHE_BASE_URL SEARCH_CACHE_TOKEN
    export SEARCH_CACHE_CONNECT_TIMEOUT_SECONDS="${SEARCH_CACHE_CONNECT_TIMEOUT_SECONDS:-5}"
    export SEARCH_CACHE_READ_TIMEOUT_SECONDS="${SEARCH_CACHE_READ_TIMEOUT_SECONDS:-600}"

    # Only this preflight subprocess uses the lightweight source package.
    # Do not change the training process's package resolution.
    (
        cd "${repo_dir}/agentflow" || exit 1
        PYTHONPATH="${repo_dir}/agentflow${PYTHONPATH:+:${PYTHONPATH}}" \
        "${SEARCH_CACHE_PYTHON:-python3}" - <<'PY'
import os
import sys
from agentflow.tools.search_gateway import SearchGatewayClient, SearchGatewayError

try:
    with SearchGatewayClient(
        base_url=os.environ["SEARCH_CACHE_BASE_URL"],
        token=os.environ["SEARCH_CACHE_TOKEN"],
        connect_timeout=5,
        read_timeout=10,
    ) as client:
        status = client.health()
    if status.get("status") != "ok":
        raise RuntimeError("Search cache service is not ready")
except (SearchGatewayError, RuntimeError) as exc:
    print(f"Search cache preflight failed: {exc}", file=sys.stderr)
    sys.exit(1)
print("[OK] Shared search cache connected; tool requests will use the service.")
PY
    )
}

_agentflow_enable_search_cache
_search_cache_enable_status=$?
unset -f _agentflow_enable_search_cache
return "${_search_cache_enable_status}" 2>/dev/null || exit "${_search_cache_enable_status}"
