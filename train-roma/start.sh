#!/usr/bin/env bash
# Task-aware entry point; run from any directory without editing machine paths.
set -e

_agentflow_start_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "${_agentflow_start_repo}" || exit 1
source "${_agentflow_start_repo}/train-roma/task_network.sh"
agentflow_resolve_task

if agentflow_task_needs_search; then
    echo "[network] TASK=${TASK}: search components follow their configured switches."
    # QA can use direct access, a gateway/cache, or the existing egress proxy.
    # Require proxy credentials only when that proxy is explicitly requested.
    case "${SEARCH_PROXY_ENABLED:-auto}" in
        1|true|True|TRUE|yes|YES|on|ON)
            source "${_agentflow_start_repo}/train-roma/enable_search_proxy.sh"
            ;;
        0|false|False|FALSE|no|NO|off|OFF) ;;
        auto)
            if [[ -n "${PROXY_HOST:-}" && -n "${PROXY_TOKEN:-}" ]]; then
                source "${_agentflow_start_repo}/train-roma/enable_search_proxy.sh"
            fi
            ;;
        *) echo "SEARCH_PROXY_ENABLED must be auto, 0 or 1." >&2; exit 2 ;;
    esac
else
    _agentflow_start_status=$?
    if [[ "${_agentflow_start_status}" -ne 1 ]]; then exit "${_agentflow_start_status}"; fi
    agentflow_disable_search_components
    echo "[network] TASK=${TASK}: search proxy, cache and SSH tunnel skipped."
fi

exec bash "${_agentflow_start_repo}/train-roma/run_train.sh" "$@"
