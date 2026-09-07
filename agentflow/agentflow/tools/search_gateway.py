from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import requests


DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 60.0


class SearchGatewayError(RuntimeError):
    """Search Gateway 访问或返回异常。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "search_gateway_error",
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        self.message = " ".join(str(message).split())[:500]
        self.code = code
        self.status_code = status_code
        self.request_id = request_id

        details = [code]

        if status_code is not None:
            details.append(f"HTTP {status_code}")

        if request_id:
            details.append(f"request_id={request_id}")

        super().__init__(
            f"{self.message} ({', '.join(details)})"
        )


class SearchGatewayConfigurationError(SearchGatewayError):
    """Gateway 配置错误。"""


def _positive_float(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise SearchGatewayConfigurationError(
            f"{name} must be a number",
            code="invalid_gateway_configuration",
        ) from exc

    if not math.isfinite(value) or value <= 0:
        raise SearchGatewayConfigurationError(
            f"{name} must be greater than zero",
            code="invalid_gateway_configuration",
        )

    return value


class SearchGatewayClient:
    """
    AgentFlow -> EC2 -> Windows Search Gateway Client.

    兼容两种调用：

        SearchGatewayClient()

    和：

        SearchGatewayClient.from_env()

    后续 Tool 逐步迁移到 from_env()。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        *,
        connect_timeout: Optional[float] = None,
        read_timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ):
        # 为当前服务器 v1 保留无参构造兼容性
        if base_url is None:
            base_url = os.environ.get("SEARCH_GATEWAY_BASE_URL", "")

        if token is None:
            token = os.environ.get("SEARCH_GATEWAY_TOKEN", "")

        self.base_url = self._validate_base_url(base_url)
        self.token = self._validate_token(token)

        if connect_timeout is None:
            connect_timeout = os.environ.get(
                "SEARCH_GATEWAY_CONNECT_TIMEOUT",
                DEFAULT_CONNECT_TIMEOUT,
            )

        if read_timeout is None:
            read_timeout = os.environ.get(
                "SEARCH_GATEWAY_READ_TIMEOUT",
                DEFAULT_READ_TIMEOUT,
            )

        self.connect_timeout = _positive_float(
            connect_timeout,
            "SEARCH_GATEWAY_CONNECT_TIMEOUT",
        )

        self.read_timeout = _positive_float(
            read_timeout,
            "SEARCH_GATEWAY_READ_TIMEOUT",
        )

        self.session = (
            session if session is not None
            else requests.Session()
        )

        # 最关键：
        # 禁止继承训练服务器 HTTP_PROXY / HTTPS_PROXY。
        self.session.trust_env = False

    @classmethod
    def from_env(
        cls,
        *,
        session: Optional[requests.Session] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "SearchGatewayClient":

        env = os.environ if environ is None else environ

        base_url = env.get("SEARCH_GATEWAY_BASE_URL", "")
        token = env.get("SEARCH_GATEWAY_TOKEN", "")

        if not base_url:
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_BASE_URL is required",
                code="missing_gateway_configuration",
            )

        if not token:
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_TOKEN is required",
                code="missing_gateway_configuration",
            )

        return cls(
            base_url=base_url,
            token=token,
            connect_timeout=env.get(
                "SEARCH_GATEWAY_CONNECT_TIMEOUT",
                DEFAULT_CONNECT_TIMEOUT,
            ),
            read_timeout=env.get(
                "SEARCH_GATEWAY_READ_TIMEOUT",
                DEFAULT_READ_TIMEOUT,
            ),
            session=session,
        )

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        value = str(base_url or "").strip().rstrip("/")

        parsed = urlsplit(value)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_BASE_URL must be a valid HTTP(S) URL",
                code="invalid_gateway_configuration",
            )

        return value

    @staticmethod
    def _validate_token(token: str) -> str:
        raw = str(token or "")
        value = raw.strip()

        if (
            not value
            or raw != value
            or any(ch.isspace() for ch in value)
        ):
            raise SearchGatewayConfigurationError(
                "SEARCH_GATEWAY_TOKEN must be non-empty "
                "and contain no whitespace",
                code="invalid_gateway_configuration",
            )

        return value

    def _url(self, path: str) -> str:
        # 不能用 urljoin：
        # 否则 path 以 / 开头时可能丢掉 /agentflow-search。
        return (
            self.base_url.rstrip("/")
            + "/"
            + path.lstrip("/")
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ):
        url = self._url(path)

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "AgentFlow-SearchGatewayClient/1.0",
        }

        kwargs = {
            "headers": headers,
            "timeout": (
                self.connect_timeout,
                self.read_timeout,
            ),
        }

        if payload is not None:
            kwargs["json"] = payload

        try:
            response = self.session.request(
                method,
                url,
                **kwargs,
            )

        except (
            requests.ConnectionError,
            requests.Timeout,
        ) as exc:
            raise SearchGatewayError(
                "Search gateway is unreachable or timed out",
                code="gateway_unreachable",
            ) from exc

        except requests.RequestException as exc:
            raise SearchGatewayError(
                "Search gateway request could not be sent",
                code="gateway_request_error",
            ) from exc

        request_id = response.headers.get("X-Request-ID")

        if not 200 <= response.status_code < 300:
            message = "Search gateway rejected the request"
            code = f"gateway_http_{response.status_code}"

            # 只解析结构化错误。
            # 不直接把整个任意 response body 打进训练日志。
            try:
                data = response.json()

                if isinstance(data, dict):
                    error = data.get("error")
                    detail = data.get("detail")

                    if isinstance(error, dict):
                        if error.get("message"):
                            message = str(error["message"])
                        if error.get("code"):
                            code = str(error["code"])

                    elif isinstance(detail, str):
                        message = detail

                    elif isinstance(detail, dict):
                        upstream = str(detail.get("upstream") or "").strip()
                        upstream_status = detail.get("status")
                        detail_message = detail.get("message")
                        detail_code = detail.get("code")

                        if detail_message:
                            message = str(detail_message)
                        elif upstream and upstream_status is not None:
                            message = (
                                f"{upstream} upstream returned "
                                f"HTTP {upstream_status}"
                            )
                        elif upstream:
                            message = f"{upstream} upstream request failed"

                        if detail_code:
                            code = str(detail_code)
                        elif upstream and upstream_status is not None:
                            code = f"{upstream}_http_{upstream_status}"

            except (TypeError, ValueError):
                pass

            raise SearchGatewayError(
                message,
                code=code,
                status_code=response.status_code,
                request_id=request_id,
            )

        try:
            return response.json()

        except (TypeError, ValueError) as exc:
            raise SearchGatewayError(
                "Search gateway returned invalid JSON",
                code="invalid_gateway_response",
                status_code=response.status_code,
                request_id=request_id,
            ) from exc

    def health(self) -> Dict[str, Any]:
        result = self._request(
            "GET",
            "healthz",
        )

        if not isinstance(result, dict):
            raise SearchGatewayError(
                "Gateway health response must be a JSON object",
                code="invalid_gateway_response",
            )

        return result

    def fetch(self, url: str) -> Dict[str, Any]:
        target = str(url or "").strip()

        parsed = urlsplit(target)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
        ):
            raise SearchGatewayError(
                "fetch URL must be an absolute HTTP(S) URL",
                code="invalid_request",
            )

        result = self._request(
            "POST",
            "v1/fetch",
            payload={
                "url": target,
            },
        )

        if (
            not isinstance(result, dict)
            or not isinstance(result.get("text"), str)
        ):
            raise SearchGatewayError(
                "Gateway fetch response is missing text",
                code="invalid_gateway_response",
            )

        return result

    def wikipedia_search(
        self,
        query: str,
        *,
        max_pages: int = 10,
        max_length: int = 256,
        language: str = "en",
    ) -> Dict[str, Any]:

        query = str(query or "").strip()

        if not query:
            raise SearchGatewayError(
                "Wikipedia query must not be empty",
                code="invalid_request",
            )

        result = self._request(
            "POST",
            "v1/search/wikipedia",
            payload={
                "query": query,
                "max_pages": int(max_pages),
                "max_length": int(max_length),
                "language": str(language or "en"),
            },
        )

        # 兼容当前 Windows Gateway。
        if isinstance(result, list):
            result = {
                "results": result,
            }

        if (
            not isinstance(result, dict)
            or not isinstance(result.get("results"), list)
        ):
            raise SearchGatewayError(
                "Wikipedia gateway response is missing results",
                code="invalid_gateway_response",
            )

        return result

    def brave_search(
        self,
        query: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        # 同时兼容当前：
        # client.brave_search(query="...")
        #
        # 以及早期：
        # client.brave_search(**payload)

        payload = dict(kwargs)

        if query is not None:
            payload["query"] = str(query).strip()

        if not payload.get("query"):
            raise SearchGatewayError(
                "Brave query must not be empty",
                code="invalid_request",
            )

        result = self._request(
            "POST",
            "v1/search/brave",
            payload=payload,
        )

        if not isinstance(result, dict):
            raise SearchGatewayError(
                "Brave gateway response must be a JSON object",
                code="invalid_gateway_response",
            )

        return result

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()