#!/usr/bin/env bash
# Source on EVERY training node before starting Ray and the rollout server.
_agentflow_enable_search_cache() {
    local repo_dir env_file name index original_flags
    local -a supplied_names supplied_values
    repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    env_file="${SEARCH_CACHE_ENV_FILE:-${repo_dir}/train-roma/search-cache.local.env}"
    supplied_names=()
    supplied_values=()
    # Platform/job environment settings take precedence over the local file.
    # Preserve values in memory; never echo secrets or put them in arguments.
    while IFS= read -r name; do
        supplied_names+=("${name}")
        supplied_values+=("${!name}")
    done < <(compgen -A variable SEARCH_CACHE_ || true; compgen -A variable SEARCH_SERVICE_ || true)
    original_flags="$-"
    if [[ -f "${env_file}" ]]; then
        set -a
        source "${env_file}"
        if [[ "${original_flags}" != *a* ]]; then set +a; fi
    fi
    for ((index=0; index<${#supplied_names[@]}; index++)); do
        printf -v "${supplied_names[index]}" '%s' "${supplied_values[index]}"
        export "${supplied_names[index]}"
    done
    case "${SEARCH_CACHE_ENABLED:-1}" in
        0|false|FALSE|False|no|NO|off|OFF)
            export SEARCH_CACHE_ENABLED=0
            return 0
            ;;
    esac
    export SEARCH_CACHE_BASE_URL="${SEARCH_CACHE_BASE_URL:-http://127.0.0.1:${SEARCH_CACHE_LOCAL_PORT:-8091}}"
    if [[ -z "${SEARCH_CACHE_TOKEN:-}" ]]; then
        echo "Set SEARCH_CACHE_TOKEN in ${env_file} on the development server before the platform copies this repository to training nodes." >&2
        return 1
    fi
    export SEARCH_CACHE_ENABLED=1
    export SEARCH_CACHE_BASE_URL SEARCH_CACHE_TOKEN
    export SEARCH_CACHE_CONNECT_TIMEOUT_SECONDS="${SEARCH_CACHE_CONNECT_TIMEOUT_SECONDS:-5}"
    export SEARCH_CACHE_READ_TIMEOUT_SECONDS="${SEARCH_CACHE_READ_TIMEOUT_SECONDS:-600}"
    # The platform copies this checkout to each training node. Prepare local
    # SSH runtime files automatically; no chmod/login is needed on the node.
    export SEARCH_CACHE_SSH_AUTO_PREPARE="${SEARCH_CACHE_SSH_AUTO_PREPARE:-1}"
    export SEARCH_CACHE_SSH_IDENTITY_FILE="${SEARCH_CACHE_SSH_IDENTITY_FILE-pem/h50065774.pem}"

    # The bootstrap is stdlib-only: it can reuse/start the development service
    # and manage the node's SSH forward before any models or Ray workers start.
    # It never installs a cache service on this training node.
    "${SEARCH_CACHE_PYTHON:-python3}" "${repo_dir}/train-roma/search_cache_bootstrap.py"
}

_agentflow_enable_search_cache
_search_cache_enable_status=$?
unset -f _agentflow_enable_search_cache
return "${_search_cache_enable_status}" 2>/dev/null || exit "${_search_cache_enable_status}"
