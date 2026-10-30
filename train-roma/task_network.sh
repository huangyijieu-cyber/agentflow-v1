#!/usr/bin/env bash
# Source before loading search proxy/cache configuration or opening SSH tunnels.

agentflow_resolve_task() {
    local resolved config_path helper_dir
    if [[ "${TASK+x}" == x ]]; then
        resolved="${TASK#"${TASK%%[![:space:]]*}"}"
        resolved="${resolved%"${resolved##*[![:space:]]}"}"
        resolved="$(printf '%s' "${resolved}" | LC_ALL=C tr '[:upper:]' '[:lower:]')"
    else
        helper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 2
        config_path="${1:-${AGENTFLOW_CONFIG_FILE:-${helper_dir}/config.yaml}}"
        resolved="$("${AGENTFLOW_TASK_PYTHON:-python3}" "${helper_dir}/task_network.py" task --config "${config_path}")" || return 2
    fi
    if [[ ! "${resolved}" =~ ^[a-z][a-z0-9_-]*$ ]]; then
        echo "Task network routing failed: TASK must be a nonempty task name." >&2
        return 2
    fi
    export TASK="${resolved}"
}

agentflow_task_needs_search() {
    local selected="${1:-${TASK:-}}"
    if [[ -z "${selected}" ]]; then
        agentflow_resolve_task || return 2
        selected="${TASK}"
    fi
    selected="${selected#"${selected%%[![:space:]]*}"}"
    selected="${selected%"${selected##*[![:space:]]}"}"
    selected="$(printf '%s' "${selected}" | LC_ALL=C tr '[:upper:]' '[:lower:]')"
    if [[ ! "${selected}" =~ ^[a-z][a-z0-9_-]*$ ]]; then return 2; fi
    case "${selected}" in
        alfworld|webshop|rule) return 1 ;;
        *) return 0 ;;
    esac
}

agentflow_disable_search_network() {
    local name managed_url managed_path remaining component rebuilt first more
    managed_url="${AGENTFLOW_SEARCH_PROXY_URL:-}"
    managed_path="${PROXY_PY_DIR:-}"
    # Remove only this search component's compatibility hook. Other Python
    # paths and user-supplied general-purpose proxies remain unchanged.
    if [[ -n "${managed_path}" && "${PYTHONPATH+x}" == x ]]; then
        remaining="${PYTHONPATH}"; rebuilt=""; first=1
        while :; do
            more=0
            if [[ "${remaining}" == *:* ]]; then
                component="${remaining%%:*}"; remaining="${remaining#*:}"; more=1
            else
                component="${remaining}"
            fi
            if [[ "${component}" != "${managed_path}" ]]; then
                if [[ "${first}" == 1 ]]; then rebuilt="${component}"; first=0
                else rebuilt="${rebuilt}:${component}"; fi
            fi
            [[ "${more}" == 1 ]] || break
        done
        export PYTHONPATH="${rebuilt}"
    fi
    if [[ -n "${managed_url}" ]]; then
        for name in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
            if [[ "${!name-}" == "${managed_url}" ]]; then unset "${name}"; fi
        done
    fi
    unset AGENTFLOW_SEARCH_PROXY_ACTIVE AGENTFLOW_SEARCH_PROXY_URL PROXY_PY_DIR
    export SEARCH_CACHE_ENABLED=0 SEARCH_PROXY_ENABLED=0
    export SEARCH_CACHE_AUTO_START=0 SEARCH_CACHE_AUTO_TUNNEL=0
}

# Name used by launchers that disable all optional search components at once.
agentflow_disable_search_components() { agentflow_disable_search_network; }
