#!/usr/bin/env python3
"""Paced, uncached Wiki requests to inspect a User-Agent and HTTP 429 evidence.

Uses the existing environment proxy, never the shared cache or search gateway.
Stops on the first rejection/error, without retries or changing identities.
The client can record its final PreparedRequest UA; it cannot prove what a
proxy forwards or which identity/rate-limit class Wikimedia assigned to it.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

import requests
import urllib3


DEFAULT_USER_AGENT = "AgentFlowResearchBot/1.0"
BODY_LIMIT = 4096
SECRET_NAMES = ("PROXY_TOKEN", "SEARCH_SERVICE_TOKEN", "SEARCH_CACHE_TOKEN",
                "SEARCH_GATEWAY_TOKEN", "GATEWAY_TOKEN", "BRAVE_API_KEY",
                "YIBU_BRAVE_API_KEY", "OPENAI_API_KEY")


def positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a finite positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--requests", type=positive_int, default=120,
                        help="maximum actual HTTP attempts; stop early on any error")
    parser.add_argument("--rpm", type=positive_float, default=60,
                        help="maximum request-start rate; one serial request at a time")
    parser.add_argument("--timeout", type=positive_float, default=10,
                        help="read inactivity timeout in seconds; connect timeout is min(5, timeout)")
    parser.add_argument("--query", default="Moon")
    parser.add_argument("--language", default="en")
    parser.add_argument("--user-agent", default=None,
                        help="default: WIKIMEDIA_USER_AGENT env, otherwise AgentFlowResearchBot/1.0")
    parser.add_argument("--output-dir", default="", help="new report directory; default is under system temp")
    parser.add_argument("--allow-direct", action="store_true",
                        help="explicitly allow running without an environment proxy")
    parser.add_argument("--verify-tls", action="store_true",
                        help="enable TLS verification; default matches the existing proxy route")
    return parser


def _safe_text(value, limit=800):
    value = str(value)
    for name in SECRET_NAMES:
        secret = os.environ.get(name)
        if secret and len(secret) >= 4:
            value = value.replace(secret, "[REDACTED]")
    # Never save proxy URL credentials, even if an error page echoes its URL.
    value = re.sub(r"(https?://)[^\s/@]+(?::[^\s/@]*)?@", r"\1[REDACTED]@", value)
    return " ".join(value.split())[:limit]


def _client_ua(response, error=None):
    prepared = getattr(response, "request", None)
    if prepared is None and error is not None:
        prepared = getattr(error, "request", None)
    return getattr(prepared, "headers", {}).get("User-Agent")


def _response_evidence(response):
    names = {"retry-after", "server", "via", "x-cache", "content-type",
             "x-request-id", "x-ratelimit-limit", "x-ratelimit-remaining",
             "x-ratelimit-reset", "ratelimit-limit", "ratelimit-remaining",
             "ratelimit-reset", "ratelimit-policy"}
    headers = {key.lower(): _safe_text(value, 300)
               for key, value in response.headers.items() if key.lower() in names}
    body = bytearray()
    truncated = False
    read_error = None
    try:
        for chunk in response.iter_content(chunk_size=1024):
            room = BODY_LIMIT - len(body)
            if len(chunk) > room:
                body.extend(chunk[:room])
                truncated = True
                break
            body.extend(chunk)
    except requests.RequestException as exc:
        # Retain the HTTP status/headers already received, e.g. a 429 whose
        # response body stalls. Do not downgrade it to an anonymous timeout.
        read_error = type(exc).__name__
    decoded = body.decode("utf-8", errors="replace")
    api_error = None
    json_valid = False
    if not truncated and read_error is None:
        try:
            payload = json.loads(decoded)
            json_valid = isinstance(payload, dict) and isinstance(payload.get("query"), dict)
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                api_error = _safe_text(payload["error"].get("code", "unknown_api_error"), 100)
        except (ValueError, TypeError):
            pass
    return {"response_headers": headers, "body_excerpt": _safe_text(decoded),
            "body_truncated": truncated, "body_read_error": read_error,
            "api_error_code": api_error, "valid_api_response": json_valid}


def run_probe(args):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", args.language):
        raise ValueError("--language must be a Wikipedia language code")
    ua = args.user_agent if args.user_agent is not None else os.environ.get("WIKIMEDIA_USER_AGENT", DEFAULT_USER_AGENT)
    if not ua or not ua.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in ua):
        raise ValueError("User-Agent must be nonempty printable ASCII")
    if not args.query.strip():
        raise ValueError("--query must not be empty")
    endpoint = f"https://{args.language}.wikipedia.org/w/api.php"
    proxy_configured = bool(requests.utils.get_environ_proxies(endpoint).get("https")
                            or requests.utils.get_environ_proxies(endpoint).get("all"))
    if not proxy_configured and not args.allow_direct:
        raise ValueError("No HTTPS environment proxy found. Source enable_search_proxy.sh, or explicitly use --allow-direct.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(args.output_dir).expanduser() if args.output_dir else (
        Path(tempfile.gettempdir()) / "agentflow-wiki-ua" / f"{stamp}-{uuid.uuid4().hex[:8]}")
    directory.mkdir(parents=True, exist_ok=False)
    events_path, report_path = directory / "events.jsonl", directory / "report.json"
    report = {"schema_version": 1, "started_at": stamp, "endpoint": endpoint,
              "configured_user_agent": ua,
              "ua_has_contact_hint": bool(re.search(r"https?://\S+|[^\s()@]+@[^\s()@]+\.[^\s()@]+", ua)),
              "proxy_configured": proxy_configured, "requested_requests": args.requests,
              "requested_rpm": args.rpm, "concurrency": 1, "automatic_retries": 0,
              "read_timeout_seconds": args.timeout, "ua_classification": "unknown",
              "attempts": 0, "ok_count": 0, "http_status_counts": {}, "error_counts": {},
              "stop_reason": "completed", "elapsed_seconds": 0, "observed_start_rpm": None,
              "events_file": str(events_path.resolve()),
              "interpretation": "This probes observed responses under the current shared traffic. It does not identify an official UA class or an independent RPM quota."}
    previous_ua = os.environ.get("WIKIMEDIA_USER_AGENT")
    # Existing sitecustomize forces this env value into Wiki request headers.
    # An explicit --user-agent must control that path too, rather than merely
    # changing a header that the compatibility shim immediately overwrites.
    os.environ["WIKIMEDIA_USER_AGENT"] = ua
    starts, statuses, errors = [], Counter(), Counter()
    started = time.monotonic()
    verify = os.environ.get("SEARCH_SERVICE_CA_BUNDLE") or args.verify_tls
    print(f"User-Agent: {_safe_text(ua, 1000)}", flush=True)
    print(f"Proxy configured: {proxy_configured}; target {args.rpm:g} RPM; serial; no retries", flush=True)
    print(f"Report directory: {directory.resolve()}", flush=True)
    if not report["ua_has_contact_hint"]:
        print("UA 中未发现网址或联系邮箱；这不是 Wikimedia 身份分类的检测结果。", flush=True)
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        with events_path.open("x", encoding="utf-8") as events, requests.Session() as session:
            session.trust_env = True
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            for index in range(1, args.requests + 1):
                # Never catch up after a slow request by sending a burst.
                if starts:
                    wait = starts[-1] + 60.0 / args.rpm - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                attempt_started = time.monotonic()
                starts.append(attempt_started)
                record = {"index": index, "started_offset_seconds": attempt_started - started,
                          "status_code": None, "client_sent_user_agent": None,
                          "ua_matches_requested": None, "response_headers": {},
                          "body_excerpt": "", "api_error_code": None, "error_type": None}
                response = None
                interrupted = False
                try:
                    response = session.get(
                        endpoint, headers={"User-Agent": ua, "Accept": "application/json"},
                        params={"action": "query", "format": "json", "list": "search",
                                "srsearch": args.query, "srlimit": 1, "srprop": "",
                                "requestid": uuid.uuid4().hex},
                        timeout=(min(5.0, args.timeout), args.timeout),
                        verify=verify, allow_redirects=False, stream=True,
                    )
                    record["status_code"] = response.status_code
                    record["client_sent_user_agent"] = _client_ua(response)
                    record.update(_response_evidence(response))
                except requests.RequestException as exc:
                    record["error_type"] = type(exc).__name__
                    record["client_sent_user_agent"] = _client_ua(response, exc)
                except KeyboardInterrupt:
                    interrupted = True
                    record["error_type"] = "KeyboardInterrupt"
                finally:
                    if response is not None:
                        response.close()
                record["elapsed_seconds"] = time.monotonic() - attempt_started
                sent_ua = record["client_sent_user_agent"]
                if sent_ua is not None:
                    record["ua_matches_requested"] = sent_ua == ua
                    record["client_sent_user_agent"] = _safe_text(sent_ua, 1000)
                if record["status_code"] is not None:
                    statuses[str(record["status_code"])] += 1
                if record["error_type"]:
                    errors[record["error_type"]] += 1
                if record["ua_matches_requested"] is False:
                    report["stop_reason"] = "ua_mismatch"
                elif interrupted:
                    report["stop_reason"] = "interrupted"
                elif record["status_code"] == 429:
                    report["stop_reason"] = "http_429"
                elif record["error_type"]:
                    report["stop_reason"] = "network_error"
                elif record["status_code"] != 200:
                    report["stop_reason"] = "http_error"
                elif record.get("body_read_error"):
                    report["stop_reason"] = "body_read_error"
                elif record["api_error_code"]:
                    report["stop_reason"] = "api_error"
                elif not record.get("valid_api_response"):
                    report["stop_reason"] = "invalid_api_response"
                else:
                    report["ok_count"] += 1
                events.write(json.dumps(record, ensure_ascii=False) + "\n")
                events.flush()
                status = record["status_code"] if record["status_code"] is not None else record["error_type"]
                retry_after = record["response_headers"].get("retry-after", "-")
                print(f"[{index}/{args.requests}] HTTP={status} {record['elapsed_seconds']:.2f}s "
                      f"UA_match={record['ua_matches_requested']} Retry-After={retry_after}", flush=True)
                if report["stop_reason"] != "completed":
                    print("Stopped: " + report["stop_reason"], flush=True)
                    if record["body_excerpt"]:
                        print(record["body_excerpt"], flush=True)
                    break
    except KeyboardInterrupt:
        report["stop_reason"] = "interrupted"
    finally:
        if previous_ua is None:
            os.environ.pop("WIKIMEDIA_USER_AGENT", None)
        else:
            os.environ["WIKIMEDIA_USER_AGENT"] = previous_ua
        report.update(attempts=len(starts), http_status_counts=dict(statuses), error_counts=dict(errors),
                      elapsed_seconds=round(time.monotonic() - started, 6))
        if len(starts) > 1 and starts[-1] > starts[0]:
            report["observed_start_rpm"] = 60 * (len(starts) - 1) / (starts[-1] - starts[0])
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[REPORT] {report_path.resolve()}", flush=True)
    print("结果只反映本次实际负载；UA分类仍未知，背景训练会影响结果。", flush=True)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_probe(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Wiki UA probe setup failed: {_safe_text(exc)}\n")
    if result["stop_reason"] == "interrupted":
        return 130
    return 0 if result["stop_reason"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
