import os
import requests


class SearchGatewayError(RuntimeError):
    pass


class SearchGatewayClient:
    def __init__(self):
        self.base_url = os.environ["SEARCH_GATEWAY_BASE_URL"].rstrip("/")
        self.token = os.environ["SEARCH_GATEWAY_TOKEN"]

        self.session = requests.Session()

        # 关键：
        # 访问 7.150.10.123 时绝不能继承训练服务器原来的
        # http_proxy / https_proxy
        self.session.trust_env = False

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        )

    def _post(self, path, payload):
        url = self.base_url + "/" + path.lstrip("/")

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=(5, 180),
            )
        except requests.RequestException as exc:
            raise SearchGatewayError(
                f"Search Gateway request failed: {exc}"
            ) from exc

        if not response.ok:
            request_id = response.headers.get("X-Request-ID", "")

            try:
                body = response.json()
            except Exception:
                body = response.text[:500]

            raise SearchGatewayError(
                f"gateway status={response.status_code}, "
                f"request_id={request_id}, "
                f"body={body}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise SearchGatewayError(
                "Search Gateway returned invalid JSON"
            ) from exc

    def health(self):
        url = self.base_url + "/healthz"

        try:
            response = self.session.get(
                url,
                timeout=(5, 10),
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise SearchGatewayError(
                f"Search Gateway health check failed: {exc}"
            ) from exc

    def brave_search(self, **payload):
        return self._post(
            "/v1/search/brave",
            payload,
        )

    def wikipedia_search(self, **payload):
        return self._post(
            "/v1/search/wikipedia",
            payload,
        )

    def fetch(self, url):
        return self._post(
            "/v1/fetch",
            {"url": url},
        )