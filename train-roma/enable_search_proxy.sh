#!/bin/bash

# ==============================
# AgentFlow 一键外网代理
# 训练机 -> EC2 -> Windows -> 公司代理 -> Internet
# ==============================

: "${PROXY_TOKEN:?Set PROXY_TOKEN before sourcing this script}"
: "${PROXY_HOST:?Set PROXY_HOST before sourcing this script}"
PROXY_PORT="${PROXY_PORT:-18090}"

# ---------- HTTP/HTTPS Proxy ----------
export HTTP_PROXY="http://agentflow:${PROXY_TOKEN}@${PROXY_HOST}:${PROXY_PORT}"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

# ---------- 内网流量不走代理 ----------
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export no_proxy="$NO_PROXY"

# ---------- 当前 v3 Search Gateway ----------
export SEARCH_GATEWAY_BASE_URL="${SEARCH_GATEWAY_BASE_URL:-http://${PROXY_HOST}/agentflow-search}"
export SEARCH_GATEWAY_TOKEN="$PROXY_TOKEN"

# ---------- Python requests / Wikipedia 兼容 ----------
PROXY_PY_DIR="/tmp/agentflow_search_proxy"
mkdir -p "$PROXY_PY_DIR"

cat > "$PROXY_PY_DIR/sitecustomize.py" <<'PY'
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
        headers.setdefault("User-Agent", "AgentFlow-WebSearch/1.0")
        kwargs["headers"] = headers

    return _original_session_request(self, method, url, **kwargs)

requests.sessions.Session.request = _request_with_web_user_agent
PY

export PYTHONPATH="$PROXY_PY_DIR:${PYTHONPATH:-}"

echo "========================================"
echo "[OK] AgentFlow Search Proxy 已启用"
echo "[OK] 路径: Training -> EC2 -> Windows -> Internet"
echo "[OK] Python requests compatibility loaded"
echo "[OK] Wikipedia compatibility loaded"
echo "========================================"
