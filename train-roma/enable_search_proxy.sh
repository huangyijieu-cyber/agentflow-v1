#!/bin/bash

# ==============================
# AgentFlow 一键外网代理
# 训练机 -> EC2 -> Windows -> 公司代理 -> Internet
# ==============================

# Search-service management explicitly opts in; training follows its TASK.
_agentflow_proxy_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_agentflow_proxy_repo}/train-roma/task_network.sh" || { return 1 2>/dev/null || exit 1; }
if [[ "${1:-}" != "--service" ]]; then
    agentflow_resolve_task || { return 1 2>/dev/null || exit 1; }
    if agentflow_task_needs_search; then
        :
    else
        _agentflow_proxy_status=$?
        if [[ "${_agentflow_proxy_status}" -ne 1 ]]; then
            return "${_agentflow_proxy_status}" 2>/dev/null || exit "${_agentflow_proxy_status}"
        fi
        agentflow_disable_search_components
        return 0 2>/dev/null || exit 0
    fi
fi

_agentflow_reuse_proxy_hook=0
if [[ "${AGENTFLOW_SEARCH_PROXY_ACTIVE:-}" == 1 && -n "${PROXY_PY_DIR:-}" && -f "${PROXY_PY_DIR}/sitecustomize.py" ]]; then
    _agentflow_reuse_proxy_hook=1
fi

PROXY_TOKEN="${PROXY_TOKEN:?Please set PROXY_TOKEN in the environment}"
PROXY_HOST="${PROXY_HOST:?Please set PROXY_HOST in the environment}"
PROXY_PORT="18090"

# ---------- HTTP/HTTPS Proxy ----------
export HTTP_PROXY="http://agentflow:${PROXY_TOKEN}@${PROXY_HOST}:${PROXY_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export AGENTFLOW_SEARCH_PROXY_ACTIVE=1 SEARCH_PROXY_ENABLED=1
export AGENTFLOW_SEARCH_PROXY_URL="$HTTP_PROXY"

# ---------- 内网流量不走代理 ----------
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export no_proxy="$NO_PROXY"

# ---------- 当前 v3 Search Gateway ----------
export SEARCH_GATEWAY_BASE_URL="${SEARCH_GATEWAY_BASE_URL:-http://${PROXY_HOST}/agentflow-search}"
export SEARCH_GATEWAY_TOKEN="$PROXY_TOKEN"

# ---------- Wikimedia API identity (shared by all workers) ----------
export WIKIMEDIA_USER_AGENT="${WIKIMEDIA_USER_AGENT:-AgentFlowResearchBot/1.0}"

# ---------- Python requests / Wikipedia 兼容 ----------
# Root training startup and the SSH service user must not overwrite one
# shared file. A private directory also isolates concurrent invocations.
if [[ "${_agentflow_reuse_proxy_hook}" != 1 ]]; then
if ! PROXY_PY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agentflow_search_proxy.XXXXXXXX")"; then
    echo "Cannot create the private search proxy runtime directory." >&2
    return 1 2>/dev/null || exit 1
fi

if ! cat > "$PROXY_PY_DIR/sitecustomize.py" <<'PY'
import os
import requests
import requests.adapters as adapters
from urllib.parse import urlsplit, urlunsplit

_original_proxy_manager_for = adapters.HTTPAdapter.proxy_manager_for

def _fixed_proxy_manager_for(self, proxy, **proxy_kwargs):
    if proxy in self.proxy_manager:
        return self.proxy_manager[proxy]

    parsed = urlsplit(proxy)

    if parsed.username is not None or parsed.password is not None:
        proxy_headers = self.proxy_headers(proxy)

        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"

        clean_proxy = urlunsplit((
            parsed.scheme,
            host,
            parsed.path or "",
            parsed.query or "",
            parsed.fragment or "",
        ))

        manager = adapters.proxy_from_url(
            clean_proxy,
            proxy_headers=proxy_headers,
            **proxy_kwargs,
        )

        self.proxy_manager[proxy] = manager
        return manager

    return _original_proxy_manager_for(self, proxy, **proxy_kwargs)

adapters.HTTPAdapter.proxy_manager_for = _fixed_proxy_manager_for


# Wikipedia:
# 1. http 自动改 https
# 2. 公司代理证书环境关闭 verify
# 3. 显式 User-Agent
_original_session_request = requests.sessions.Session.request

def _request_with_web_user_agent(self, method, url, **kwargs):
    if "wikipedia.org" in str(url):
        if str(url).startswith("http://"):
            url = "https://" + str(url)[7:]

        kwargs.setdefault("verify", False)

        headers = dict(kwargs.get("headers") or {})
        headers["User-Agent"] = os.environ["WIKIMEDIA_USER_AGENT"]
        kwargs["headers"] = headers

    return _original_session_request(self, method, url, **kwargs)

requests.sessions.Session.request = _request_with_web_user_agent
PY
then
    echo "Cannot write search proxy compatibility code." >&2
    return 1 2>/dev/null || exit 1
fi
fi

export PROXY_PY_DIR
case ":${PYTHONPATH:-}:" in
    *":${PROXY_PY_DIR}:"*) ;;
    *) export PYTHONPATH="$PROXY_PY_DIR:${PYTHONPATH:-}" ;;
esac

echo "========================================"
echo "[OK] AgentFlow Search Proxy 已启用"
echo "[OK] 路径: Training -> EC2 -> Windows -> Internet"
echo "[OK] Python requests compatibility loaded"
echo "[OK] Wikipedia compatibility loaded"
echo "========================================"
