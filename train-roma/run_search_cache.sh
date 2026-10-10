#!/usr/bin/env bash
# Run on the development server. No model/GPU dependencies are required.
set -euo pipefail

_search_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SEARCH_CACHE_DIR="${SEARCH_CACHE_DIR:-/home/ma-user/work/code-rl/cache}"
_search_env_file="${SEARCH_SERVICE_ENV_FILE:-${SEARCH_CACHE_DIR}/search-service.env}"
if [[ -f "${_search_env_file}" ]]; then
    set -a
    source "${_search_env_file}"
    set +a
fi

if [[ -z "${SEARCH_SERVICE_TOKEN:-${SEARCH_CACHE_TOKEN:-}}" ]]; then
    echo "Set SEARCH_SERVICE_TOKEN or put it in ${_search_env_file}." >&2
    exit 1
fi

# Keep the current EC2 -> personal host egress chain. Set this to 0 if the
# service environment already supplies HTTP_PROXY/HTTPS_PROXY itself.
if [[ "${SEARCH_SERVICE_USE_PROXY:-1}" == "1" ]]; then
    source "${_search_repo_dir}/train-roma/enable_search_proxy.sh"
fi

export PYTHONPATH="${_search_repo_dir}/agentflow${PYTHONPATH:+:${PYTHONPATH}}"
# Use the inner, lightweight package rather than importing the training SDK.
cd "${_search_repo_dir}/agentflow"
exec "${SEARCH_SERVICE_PYTHON:-python3}" -m agentflow.search_service \
    --host "${SEARCH_SERVICE_HOST:-127.0.0.1}" \
    --port "${SEARCH_SERVICE_PORT:-8091}" \
    --cache-dir "${SEARCH_CACHE_DIR}" "$@"
