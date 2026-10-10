"""Network-only providers for the shared search service.

Candidate selection, embeddings, RAG summaries and rewards stay on training
machines. These providers cache successful raw retrievals and never cache an
error as content. All outbound HTTP and cache coordination belong to ``service``.
"""

import os
import re
from typing import Any, Dict
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from .core import CACHE_TTL_SECONDS, GatewayFailure


WIKIMEDIA_USER_AGENT = (
    "AgentFlowResearchBot/1.0 "
    "(https://github.com/huangyijieu-cyber/agentflow-v1/issues)"
)
BRAVE_ENDPOINT = "https://yibuapi.com/brave/v1/web/search"
WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _invalid(message):
    return GatewayFailure(message, code="invalid_request", status_code=400)


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool):
        raise _invalid(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid(f"{name} must be an integer") from exc
    if isinstance(value, float) and number != value:
        raise _invalid(f"{name} must be an integer")
    if not minimum <= number <= maximum:
        raise _invalid(f"{name} must be between {minimum} and {maximum}")
    return number


def _query(params):
    value = params.get("query")
    if not isinstance(value, str) or not value.strip():
        raise _invalid("query must be a nonempty string")
    # Keep the actual query, including whitespace, in the cache key. Distinct
    # queries must not become equivalent merely because a normalizer says so.
    if len(value) > 32768:
        raise _invalid("query is too long")
    return value


def _url(value):
    if not isinstance(value, str):
        raise _invalid("url must be an absolute HTTP(S) URL")
    value = value.strip()
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise _invalid("url must be an absolute HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise _invalid("url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise _invalid("url must not contain credentials")
    try:
        parsed.port
    except ValueError as exc:
        raise _invalid("url has an invalid port") from exc
    return value


def prepare(tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a request and return only its effective retrieval parameters."""
    if not isinstance(params, dict):
        raise _invalid("request parameters must be an object")
    if tool == "wikipedia":
        language = params.get("language", "en")
        if not isinstance(language, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", language):
            raise _invalid("invalid Wikipedia language")
        return {
            "query": _query(params),
            "max_pages": _integer(params.get("max_pages", 10), "max_pages", 0, 10),
            "max_length": _integer(params.get("max_length", 256), "max_length", -1, 1000000),
            "language": language,
        }
    if tool == "brave":
        result = {"query": _query(params)}
        # Match the existing tool's effective count (values clamp to 1..20).
        count = _integer(params.get("count", 10), "count", -1000000, 1000000)
        result["count"] = max(1, min(count, 20))
        for name in ("country", "search_lang", "ui_lang", "freshness"):
            value = params.get(name)
            if value:
                if not isinstance(value, str) or len(value) > 256:
                    raise _invalid(f"{name} must be a short string")
                result[name] = value
        return result
    if tool == "fetch":
        # Preserve the existing Web tool's arXiv PDF -> abstract substitution;
        # keep all URL query parameters rather than guessing equivalence.
        return {
            "url": _url(params.get("url")).replace("arxiv.org/pdf", "arxiv.org/abs"),
            "max_length": _integer(params.get("max_length", 1000000), "max_length", 0, 1000000),
        }
    raise _invalid("unknown retrieval tool")


def _tls_verify():
    """Keep the existing proxy's TLS behavior, with an explicit CA override."""
    ca_bundle = os.getenv("SEARCH_SERVICE_CA_BUNDLE")
    if ca_bundle:
        return ca_bundle
    value = os.getenv("SEARCH_SERVICE_VERIFY_TLS", "false").lower()
    if value not in {"true", "false", "1", "0"}:
        raise GatewayFailure("Invalid server setting SEARCH_SERVICE_VERIFY_TLS", code="configuration_error", status_code=500)
    return value in {"true", "1"}


def _json(response, provider):
    try:
        data = response.json()
    except (ValueError, TypeError) as exc:
        raise GatewayFailure(f"{provider} returned invalid JSON", code="invalid_upstream_response") from exc
    finally:
        response.close()
    if not isinstance(data, dict):
        raise GatewayFailure(f"{provider} returned a non-object response", code="invalid_upstream_response")
    return data


def _wiki_request(service, endpoint, params):
    payload = {"action": "query", "format": "json", **params}
    response = service.request(
        "wikipedia", "GET", endpoint,
        params=payload,
        headers={"User-Agent": os.getenv("WIKIMEDIA_USER_AGENT", WIKIMEDIA_USER_AGENT)},
        timeout=20, verify=_tls_verify(),
    )
    data = _json(response, "Wikipedia")
    error = data.get("error")
    if error:
        rate_limited = isinstance(error, dict) and error.get("code") in {"ratelimited", "maxlag"}
        cooldown = 5.0 if rate_limited else None
        if rate_limited:
            # MediaWiki may report throttling in an HTTP 200 body. Publish it
            # to the same shared budget used for transport-level HTTP 429s.
            service.notify_cooldown("wikipedia", endpoint, cooldown)
        raise GatewayFailure(
            "Wikipedia API rejected the request",
            code="rate_limited" if rate_limited else "upstream_error",
            status_code=429 if rate_limited else 502,
            retryable=rate_limited,
            upstream="wikipedia",
            retry_after=cooldown,
        )
    if not isinstance(data.get("query"), dict):
        raise GatewayFailure("Wikipedia response is missing query", code="invalid_upstream_response")
    return data["query"]


def _only_wiki_page(data):
    pages = data.get("pages")
    if not isinstance(pages, dict) or len(pages) != 1:
        raise GatewayFailure("Wikipedia response is missing its page", code="invalid_upstream_response")
    page = next(iter(pages.values()))
    if not isinstance(page, dict) or "missing" in page or "invalid" in page:
        raise GatewayFailure("Wikipedia page is unavailable", code="page_unavailable")
    return page


def _wikipedia(params, service):
    language = params["language"]
    endpoint = f"https://{language}.wikipedia.org/w/api.php"

    def search():
        data = _wiki_request(service, endpoint, {
            "list": "search", "srsearch": params["query"], "srlimit": 10, "srprop": "",
        })
        items = data.get("search")
        if not isinstance(items, list) or any(not isinstance(item, dict) or not isinstance(item.get("title"), str) for item in items):
            raise GatewayFailure("Wikipedia response is missing search results", code="invalid_upstream_response")
        return [item["title"] for item in items]

    titles = service.cached(
        "wikipedia_search", {"endpoint": endpoint, "query": params["query"], "srlimit": 10, "srprop": "", "version": 1},
        search, CACHE_TTL_SECONDS,
    )
    if not titles:
        return {"results": [{
            "title": None, "url": None, "abstract": None,
            "error": f"No results found for query: {params['query']}",
        }]}

    pages = []
    partial = False
    for title in titles[:params["max_pages"]] if params["max_pages"] else titles:
        try:
            def identity(title=title):
                page = _only_wiki_page(_wiki_request(service, endpoint, {
                    "prop": "info|pageprops", "inprop": "url", "ppprop": "disambiguation",
                    "redirects": "", "titles": title,
                }))
                pageprops = page.get("pageprops", {})
                if not isinstance(pageprops, dict):
                    raise GatewayFailure("Wikipedia page metadata is incomplete", code="invalid_upstream_response")
                if "disambiguation" in pageprops:
                    raise GatewayFailure("Wikipedia page is ambiguous", code="page_unavailable")
                if type(page.get("pageid")) is not int or page["pageid"] <= 0 or not isinstance(page.get("fullurl"), str) or not isinstance(page.get("title"), str):
                    raise GatewayFailure("Wikipedia page metadata is incomplete", code="invalid_upstream_response")
                return {"pageid": page["pageid"], "title": page["title"], "url": page["fullurl"]}

            resolved = service.cached(
                "wikipedia_page_identity", {"endpoint": endpoint, "title": title, "redirects": True, "version": 1},
                identity, CACHE_TTL_SECONDS,
            )

            def content():
                # wikipedia 1.4.0 WikipediaPage.content requests explaintext and
                # leaves exsectionformat at the API default. Keep that format;
                # using exsectionformat=plain would change section headings.
                page = _only_wiki_page(_wiki_request(service, endpoint, {
                    "prop": "extracts", "explaintext": "", "pageids": resolved["pageid"],
                }))
                if not isinstance(page.get("extract"), str):
                    raise GatewayFailure("Wikipedia page is missing text", code="invalid_upstream_response")
                return page["extract"]

            text = service.cached(
                "wikipedia_page_text", {"endpoint": endpoint, "pageid": resolved["pageid"], "explaintext": True, "version": 1},
                content, CACHE_TTL_SECONDS,
            )
            limit = params["max_length"]
            if limit != -1 and len(text) > limit:
                text = text[:limit] + "... [truncated]"
            # Keep the searched title and exact original field insertion order.
            pages.append({"title": title, "url": resolved["url"], "abstract": text})
        except GatewayFailure as exc:
            if exc.status_code == 429 or exc.code in {"rate_limited", "queue_full", "deadline_exceeded"}:
                raise
            partial = True
            pages.append({
                "title": title,
                "url": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "abstract": "Please use the URL to get the full text further if needed.",
            })
    result = {"results": pages}
    if partial:
        result["_meta"] = {"partial": True}
    return result


def _brave_results(data):
    if data.get("error") or data.get("errors"):
        raise GatewayFailure("Yibu returned an API error", code="upstream_error")
    for container, name in ((data.get("web"), "results"), (data, "organic_results"), (data.get("discussions"), "results")):
        if isinstance(container, dict) and isinstance(container.get(name), list):
            results = container[name]
            if any(not isinstance(item, dict) for item in results):
                raise GatewayFailure("Yibu returned invalid search entries", code="invalid_upstream_response")
            return results
    # An arbitrary HTTP 200 JSON object is not evidence of a successful empty
    # search (e.g. a provider error/status envelope).
    raise GatewayFailure("Yibu response is missing search results", code="invalid_upstream_response")


def _brave(params, service):
    endpoint = _url(os.getenv("BRAVE_YIBU_BASE_URL", BRAVE_ENDPOINT))
    effective = {"q": params["query"], "count": params["count"]}
    for name in ("country", "search_lang", "ui_lang", "freshness"):
        if name in params:
            effective[name] = params[name]

    def fetch():
        api_key = os.getenv("BRAVE_API_KEY") or os.getenv("YIBU_BRAVE_API_KEY")
        if not api_key:
            raise GatewayFailure("Yibu API key is not configured on the search server", code="configuration_error", status_code=503)
        response = service.request(
            "brave", "GET", endpoint,
            params=effective,
            headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
            timeout=20, verify=_tls_verify(),
        )
        data = _json(response, "Yibu")
        _brave_results(data)
        return data

    data = service.cached("brave_search", {"endpoint": endpoint, "params": effective, "version": 1}, fetch, CACHE_TTL_SECONDS)
    return {"data": data}


def _fetch(params, service):
    url = params["url"]
    hostname = (urlsplit(url).hostname or "").lower()
    headers = dict(WEB_HEADERS)
    is_wiki = hostname == "wikipedia.org" or hostname.endswith(".wikipedia.org")
    if is_wiki:
        headers["User-Agent"] = os.getenv("WIKIMEDIA_USER_AGENT", WIKIMEDIA_USER_AGENT)

    def fetch():
        response = service.request("wikipedia" if is_wiki else f"web:{hostname}", "GET", url, headers=headers, timeout=10, verify=_tls_verify())
        try:
            # Response-size enforcement and HTTP status retries are centralized
            # in service.request. Extract exactly as the existing Web tool does.
            text = BeautifulSoup(response.content, "html.parser").get_text(separator="\n", strip=True)
        except Exception as exc:
            raise GatewayFailure("Could not extract webpage text", code="invalid_upstream_response") from exc
        finally:
            response.close()
        return text[:1000000]

    text = service.cached(
        "web_text", {"url": url, "headers": headers, "parser": "html.parser", "separator": "\n", "strip": True, "max_chars": 1000000, "version": 1},
        fetch, CACHE_TTL_SECONDS,
    )
    return {"text": text[:params["max_length"]]}


def execute(tool: str, params: Dict[str, Any], service) -> Dict[str, Any]:
    """Execute already-prepared parameters without running any language model."""
    if tool == "wikipedia":
        return _wikipedia(params, service)
    if tool == "brave":
        return _brave(params, service)
    if tool == "fetch":
        return _fetch(params, service)
    raise _invalid("unknown retrieval tool")
