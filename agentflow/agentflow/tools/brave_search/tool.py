import argparse
import fcntl
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from agentflow.tools.base import BaseTool
from agentflow.tools.search_gateway import SearchGatewayClient, SearchGatewayError, search_cache_enabled
from agentflow.tools.network_retry import (
    MAX_NETWORK_RETRIES,
    MAX_RETRY_WAIT_SECONDS,
    YIBU_RETRYABLE_HTTP_STATUSES,
    retry_wait_seconds,
)

load_dotenv()

TOOL_NAME = "Yibu_Brave_Search_Tool"
DEFAULT_ENDPOINT = "https://yibuapi.com/brave/v1/web/search"

# NOTE: The remote environment egresses through a corporate proxy (netentsec) that
# terminates TLS with a self-signed certificate, causing SSLCertVerificationError.
# Following the project convention (see wikipedia_search/tool.py), disable cert
# verification for this tool's outbound search requests.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
LIMITATIONS = """
1. This tool is suitable for general web search through Yibu's Brave-compatible API.
2. It returns search result snippets, not full webpage content.
3. It may require Web_RAG_Search_Tool for reading a specific result URL in depth.
4. Search quality depends on Brave/Yibu availability and quota.
"""

BEST_PRACTICES = """
1. Use this tool for up-to-date web search and general information retrieval.
2. Use concise keyword queries for best recall.
3. Use count to request multiple candidates when the answer may require comparison.
4. Follow up with Web_RAG_Search_Tool on specific URLs when detailed page content is needed.
"""


_TOKENIZER_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_WARNED = set()
_CHINA_TZ = timezone(timedelta(hours=8))
_TOKEN_CACHE_SIZE = 1024


def yibu_timestamp():
    return datetime.now(_CHINA_TZ).isoformat(timespec="milliseconds")


def _warn_once(key, message):
    with _WRITE_LOCK:
        if key not in _WARNED:
            _WARNED.add(key)
            print(f"[Yibu Token Usage] {message}", flush=True)


@lru_cache(maxsize=4)
def _load_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
        )
    except Exception as error:
        _warn_once(("tokenizer", path),
                   f"Local tokenizer unavailable ({type(error).__name__}); "
                   "token counts will be null. Set AGENTFLOW_YIBU_TOKENIZER_PATH.")
        return None


class YibuTokenUsageRecorder:
    def __init__(self, tokenizer=None, log_dir=None):
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self._count_tokenizer = None
        self._token_counts = OrderedDict()

    def set_tokenizer(self, tokenizer):
        """Reuse the tokenizer already loaded by the rollout worker."""
        self.tokenizer = tokenizer

    def _measure(self, query, output, response_body):
        # Tokenizers can mutate their internal truncation/padding settings.
        with _TOKENIZER_LOCK:
            tokenizer = self.tokenizer
            path = (os.getenv("AGENTFLOW_YIBU_TOKENIZER_PATH")
                    or os.getenv("AGENTFLOW_AGENT_TOKENIZER_PATH")
                    or os.getenv("BASE_MODEL") or os.getenv("MODEL_PATH"))
            if tokenizer is None:
                tokenizer = _load_tokenizer(path)
            measurements = {
                "token_count_source": "unavailable", "tokenizer": path,
                "input_tokens": None, "output_tokens": None,
                "response_body_tokens": None, "total_tokens": None,
            }
            if tokenizer is None:
                _warn_once("no-tokenizer", "No local tokenizer: token counts are null, "
                           "not character/byte estimates.")
                return measurements

            measurements["tokenizer"] = str(getattr(tokenizer, "name_or_path", None)
                                           or path or type(tokenizer).__name__)
            if self._count_tokenizer is not tokenizer:
                self._token_counts.clear()
                self._count_tokenizer = tokenizer
            cache_hits = cache_misses = 0

            def count(text):
                nonlocal cache_hits, cache_misses
                if text is None:
                    return None
                if not text:
                    return 0
                # Keep only a bounded number of fingerprints/counts in memory,
                # not full query/result strings. Cache is specific to this tokenizer.
                raw = text.encode("utf-8")
                key = (len(raw), hashlib.sha256(raw).digest())
                if key in self._token_counts:
                    cache_hits += 1
                    self._token_counts.move_to_end(key)
                    return self._token_counts[key]
                cache_misses += 1
                tokens = len(tokenizer.encode(text, add_special_tokens=False))
                self._token_counts[key] = tokens
                if len(self._token_counts) > _TOKEN_CACHE_SIZE:
                    self._token_counts.popitem(last=False)
                return tokens

            try:
                input_tokens = count(query)
                # Intermediate retry attempts have no tool output. Final failures
                # include the exact error text returned to the Agent.
                output_tokens = count(output) if output is not None else 0
                measurements.update(
                    token_count_source="local-tokenizer",
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    response_body_tokens=count(response_body),
                    total_tokens=input_tokens + output_tokens,
                    count_cache_hits=cache_hits, count_cache_misses=cache_misses,
                )
            except Exception as error:
                _warn_once(("encode", type(error).__name__),
                           f"Token encoding failed ({type(error).__name__}); "
                           "token counts will be null.")
            return measurements

    def _path(self):
        log_dir = self.log_dir or os.getenv("AGENTFLOW_YIBU_TOKEN_USAGE_DIR")
        if not log_dir:
            log_dir = (str(Path(os.environ["LOG_DIR_PREFIX"]) / "task_logs" / "127.0.0.1_rank0")
                       if os.getenv("LOG_DIR_PREFIX") else "rollout_data")
        return Path(log_dir).expanduser() / "yibu_token.log"

    def record(self, *, call_id, attempt, query, output, response, status,
               elapsed_s, error=None, started_at=None, source="direct",
               event_type="upstream_attempt", http_attempt=True, error_type=None,
               finished_at=None):
        """Record measured traffic; instrumentation never issues a search request."""
        if os.getenv("AGENTFLOW_YIBU_TOKEN_USAGE", "1").lower() in {"0", "false", "no"}:
            return
        try:
            finished = datetime.now(_CHINA_TZ)
            response_body = response.text if response is not None else None
            measurements = self._measure(query, output, response_body)
            record = {
                "event_id": f"{call_id}:{attempt}", "call_id": call_id,
                "attempt": attempt,
                "started_at": started_at or (finished - timedelta(seconds=elapsed_s)).isoformat(
                    timespec="milliseconds"),
                "finished_at": finished_at or finished.isoformat(timespec="milliseconds"),
                "timestamp_utc": finished.astimezone(timezone.utc).isoformat(),
                "tool": "Yibu_Brave_Search_Tool", "status": status,
                "source": source, "event_type": event_type, "http_attempt": http_attempt,
                "http_status": response.status_code if response is not None else None,
                "error_type": type(error).__name__ if error is not None else error_type,
                "elapsed_s": round(elapsed_s, 6),
                "hostname": socket.gethostname(), "pid": os.getpid(),
                "input_text": query, "output_text": output,
                "response_body_text": response_body,
                **measurements,
            }
            # Preserve requested texts, but never authentication headers/API keys.
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            path = self._path()
            with _WRITE_LOCK:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as log:
                    # All rollout processes share one file. Flush before unlocking
                    # so even long records cannot interleave across processes.
                    fcntl.flock(log.fileno(), fcntl.LOCK_EX)
                    try:
                        log.write(line + "\n")
                        log.flush()
                    finally:
                        fcntl.flock(log.fileno(), fcntl.LOCK_UN)
                summary = {key: record[key] for key in (
                    "event_id", "started_at", "status", "input_tokens", "output_tokens",
                )}
                print("[Yibu Token Usage] " + json.dumps(summary, ensure_ascii=False), flush=True)
        except Exception as error:
            _warn_once(("record", type(error).__name__),
                       f"Could not record tokens ({type(error).__name__}); search result is preserved.")


class Brave_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "A web search tool powered by Yibu's Brave-compatible API. "
                "It returns Brave-like web search results with titles, URLs, and snippets."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
                "count": "int - Number of search results to return. Default is 10.",
                "country": "str - Optional country code, such as US, CN, JP.",
                "search_lang": "str - Optional search language, such as en, zh-hans.",
                "ui_lang": "str - Optional UI language.",
                "freshness": "str - Optional freshness filter supported by Brave-compatible API.",
            },
            output_type="str - Formatted web search results with titles, URLs, and descriptions.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Who won the euro 2024?", count=5)',
                    "description": "Search recent public web information."
                },
                {
                    "command": 'execution = tool.execute(query="Physics and Society arXiv August 11 2016", count=10)',
                    "description": "Search for a specific article or web page."
                },
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            },
        )

        self.api_key = os.getenv("BRAVE_API_KEY") or os.getenv("YIBU_BRAVE_API_KEY")
        if not self.api_key and not search_cache_enabled():
            raise Exception(
                "Yibu Brave API key not found. Please set BRAVE_API_KEY "
                "or YIBU_BRAVE_API_KEY."
            )

        self.endpoint = os.getenv("BRAVE_YIBU_BASE_URL", DEFAULT_ENDPOINT)
        self.max_retries = MAX_NETWORK_RETRIES
        self.timeout = 20
        self.token_usage = YibuTokenUsageRecorder()

    def _record_shared_usage(self, *, query, output, events, call_id, elapsed_s,
                             started_at, status):
        """Log logical calls separately from service-reported real HTTP attempts."""
        try:
            for index, event in enumerate(events):
                body = event.get("response_body_text", event.get("raw_response_text"))
                if body is not None and not isinstance(body, str):
                    continue
                error_type = event.get("error_type")
                if not isinstance(error_type, str) or not error_type.isidentifier():
                    error_type = None
                response = SimpleNamespace(text=body, status_code=event.get("http_status"))
                self.token_usage.record(
                    call_id=event["call_id"], attempt=event["attempt"],
                    query=event["input_text"],
                    output=output if index == len(events) - 1 else None,
                    response=response, status=event["status"],
                    elapsed_s=event.get("elapsed_s", event.get("duration_ms", 0) / 1000),
                    started_at=event.get("started_at"), error_type=error_type,
                    finished_at=event.get("finished_at"),
                    source="shared_service", event_type="upstream_attempt", http_attempt=True,
                )
            self.token_usage.record(
                call_id=call_id, attempt=1, query=query, output=output, response=None,
                status=status, elapsed_s=elapsed_s, started_at=started_at,
                source="shared_cache", event_type="tool_call", http_attempt=False,
            )
        except Exception as error:
            _warn_once(("shared-record", type(error).__name__),
                       f"Could not record shared tokens ({type(error).__name__}); "
                       "search result is preserved.")

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _extract_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        web_results = data.get("web", {}).get("results")
        if isinstance(web_results, list):
            return web_results

        organic_results = data.get("organic_results")
        if isinstance(organic_results, list):
            return organic_results

        discussion_results = data.get("discussions", {}).get("results")
        if isinstance(discussion_results, list):
            return discussion_results

        return []

    def _format_results(self, query: str, data: Dict[str, Any], count: int) -> str:
        results = self._extract_results(data)
        if not results:
            return f"No Brave search results found for query: {query}"

        lines = [f"Search results for: {query}", ""]
        for idx, item in enumerate(results[:count], start=1):
            title = self._as_text(item.get("title"))
            url = self._as_text(item.get("url") or item.get("link"))
            description = self._as_text(item.get("description") or item.get("snippet"))

            lines.append(f"[{idx}] {title or 'Untitled'}")
            if url:
                lines.append(f"URL: {url}")
            if description:
                lines.append(f"Description: {description}")

            extra_snippets = item.get("extra_snippets")
            if isinstance(extra_snippets, list) and extra_snippets:
                snippet_text = " ".join(self._as_text(s) for s in extra_snippets if s)
                if snippet_text:
                    lines.append(f"Extra snippets: {snippet_text}")

            lines.append("")

        return "\n".join(lines).strip()

    def _execute_search(
        self,
        query: str,
        count: int = 10,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> str:
        params = {
            "q": query,
            "count": max(1, min(int(count), 20)),
        }
        if country:
            params["country"] = country
        if search_lang:
            params["search_lang"] = search_lang
        if ui_lang:
            params["ui_lang"] = ui_lang
        if freshness:
            params["freshness"] = freshness

        if search_cache_enabled():
            call_id = uuid.uuid4().hex
            started_at = yibu_timestamp()
            started = time.perf_counter()
            events = []
            status = "success"
            try:
                with SearchGatewayClient.from_env() as gateway:
                    data = gateway.brave_search(
                        query=query,
                        **{key: value for key, value in params.items() if key != "q"},
                    )
                    events = gateway.last_usage_events
                output = self._format_results(query, data, params["count"])
            except SearchGatewayError as error:
                events = error.usage_events
                status = "gateway_error"
                output = f"Yibu Brave Search gateway request failed. Last error: {error}"
            self._record_shared_usage(
                query=query, output=output, events=events, call_id=call_id,
                elapsed_s=time.perf_counter() - started, started_at=started_at, status=status,
            )
            return output

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        attempts_made = 0
        call_id = uuid.uuid4().hex
        for attempt in range(self.max_retries + 1):
            attempts_made += 1
            started_at = yibu_timestamp()
            started = time.perf_counter()
            response = None
            output = None
            retry_delay = None
            attempt_error = None
            status = "response_error"
            try:
                response = requests.get(
                    self.endpoint,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    verify=False,
                )
                if response.status_code in YIBU_RETRYABLE_HTTP_STATUSES and attempt < self.max_retries:
                    wait_time = retry_wait_seconds(
                        attempt,
                        status_code=response.status_code,
                        retry_after=response.headers.get("Retry-After"),
                    )
                    if wait_time <= MAX_RETRY_WAIT_SECONDS:
                        retry_delay = wait_time
                        status = "http_retry"
                if retry_delay is None:
                    response.raise_for_status()
                    data = response.json()
                    output = self._format_results(query, data, params["count"])
                    status = "success"
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                attempt_error = e
                status = "network_error"
                if attempt < self.max_retries:
                    retry_delay = retry_wait_seconds(attempt)
                    status = "network_retry"
            except Exception as e:
                last_error = e
                attempt_error = e
                if response is not None and response.status_code >= 400:
                    status = "http_error"
            finally:
                if output is None and retry_delay is None:
                    output = (
                        f"Yibu Brave Search tried {attempts_made} "
                        f"time{'s' if attempts_made != 1 else ''} but failed. "
                        f"Last error: {last_error}"
                    )
                self.token_usage.record(
                    call_id=call_id, attempt=attempt + 1, query=query,
                    output=output, response=response, status=status,
                    elapsed_s=time.perf_counter() - started, error=attempt_error,
                    started_at=started_at,
                )
                if response is not None:
                    response.close()

            if output is not None:
                return output
            if retry_delay is not None:
                reason = (str(response.status_code) if status == "http_retry"
                          else type(attempt_error).__name__)
                label = "HTTP" if status == "http_retry" else "Network"
                print(f"[Yibu Brave {label}] {reason}; retrying in {retry_delay:.1f}s "
                      f"({attempt + 1}/{self.max_retries})")
                time.sleep(retry_delay)
                continue
            break

        return (
            f"Yibu Brave Search tried {attempts_made} "
            f"time{'s' if attempts_made != 1 else ''} but failed. "
            f"Last error: {last_error}"
        )

    def execute(
        self,
        query: str,
        count: int = 10,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        freshness: Optional[str] = None,
    ) -> str:
        return self._execute_search(
            query=query,
            count=count,
            country=country,
            search_lang=search_lang,
            ui_lang=ui_lang,
            freshness=freshness,
        )



TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "response_body_tokens")


def summarize(paths):
    files = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.update(path.rglob("yibu_tokens_*.jsonl"))
            files.update(path.rglob("yibu_token.log"))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(path)
    groups = {}
    seen = set()
    calls = set()
    upstream_calls = set()
    http_events = set()
    logical_events = set()
    duplicates = malformed = 0
    for path in sorted(files):
        with path.open(encoding="utf-8") as log:
            for line in log:
                try:
                    record = json.loads(line)
                    event_id = record["event_id"]
                    event_type = record.get("event_type", "upstream_attempt")
                    source = record.get("source", "direct")
                    http_attempt = record.get("http_attempt", True)
                    if (event_type not in {"upstream_attempt", "tool_call"}
                            or not isinstance(source, str) or type(http_attempt) is not bool
                            or http_attempt != (event_type == "upstream_attempt")):
                        raise ValueError("Invalid traffic category")
                    key = (record["token_count_source"], record["tokenizer"], source, event_type)
                    call_id = record["call_id"]
                    if not isinstance(event_id, str) or not isinstance(call_id, str):
                        raise ValueError("Invalid event identity")
                    status = record["status"]
                    if not isinstance(status, str):
                        raise ValueError("Invalid status")
                    tokens = {field: record[field] for field in TOKEN_FIELDS}
                    if any(value is not None and (type(value) is not int or value < 0)
                           for value in tokens.values()):
                        raise ValueError("Invalid token count")
                    hash(key)
                except (ValueError, KeyError, TypeError):
                    malformed += 1
                    continue
                # A single service HTTP attempt can reach many coalesced clients,
                # including clients using different tokenizers. Count it only once;
                # tokenizer groups remain distinct rather than summing estimates.
                identity = (event_type, event_id, key)
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                if http_attempt:
                    http_events.add(event_id)
                    upstream_calls.add(call_id)
                    if source == "direct":
                        calls.add((source, call_id))
                else:
                    logical_events.add(event_id)
                    calls.add((source, call_id))
                if key not in groups:
                    groups[key] = {
                        "token_count_source": key[0], "tokenizer": key[1],
                        "source": source, "event_type": event_type,
                        "http_attempts": 0, "successful_attempts": 0,
                        "tool_calls": 0,
                        "sums": {field: 0 for field in TOKEN_FIELDS},
                        "unknown_counts": {field: 0 for field in TOKEN_FIELDS},
                    }
                group = groups[key]
                group["http_attempts"] += int(http_attempt)
                group["successful_attempts"] += int(http_attempt and status == "success")
                group["tool_calls"] += int(not http_attempt)
                for field, value in tokens.items():
                    if value is None:
                        group["unknown_counts"][field] += 1
                    else:
                        group["sums"][field] += value
    return {
        "files": len(files), "search_calls": len(calls),
        "http_attempts": len(http_events), "tool_calls": len(logical_events),
        "upstream_calls": len(upstream_calls), "duplicates_ignored": duplicates,
        "malformed_lines": malformed, "groups": list(groups.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yibu search and token-log summary")
    parser.add_argument("--summarize", nargs="+", metavar="PATH",
                        help="Summarize token log files or directories without calling Yibu")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.summarize), ensure_ascii=False, indent=2))
    else:
        tool = Brave_Search_Tool()
        print(tool.execute(query="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.?", count=3))
