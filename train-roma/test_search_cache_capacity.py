#!/usr/bin/env python3
"""Measure configured boundaries and the highest tested stable network load.

No deployed service, database, environment file or training process is changed.
Real probes require --mode upstream/all; --mode local never contacts Internet.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_cache_capacity_local import run_local_suite
from search_cache_capacity_upstream import ProbeOutcome, UpstreamProbe, probe_stage


ARTICLE_TITLES = [
    "Albert Einstein", "Ada Lovelace", "Alan Turing", "Marie Curie",
    "Isaac Newton", "Charles Darwin", "Leonardo da Vinci", "Nikola Tesla",
    "Galileo Galilei", "Ludwig van Beethoven", "Johann Sebastian Bach",
    "Alexander Fleming", "Rosalind Franklin", "Katherine Johnson",
    "James Clerk Maxwell", "Grace Hopper", "Louis Pasteur", "Richard Feynman",
]
WIKI_KINDS = ("wiki_api", "wiki_identity", "wiki_extract", "wiki_html")


def number_grid(value, *, integer=False):
    try:
        values = [int(part) if integer else float(part) for part in value.split(",")]
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("use a comma-separated list of positive numbers") from None
    if not values or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError("all grid values must be finite and positive")
    return sorted(set(values))


def load_records(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def load_queries(path=None, key=None):
    if not path:
        return ARTICLE_TITLES[:]
    result = []
    for record in load_records(path):
        if isinstance(record, str):
            query = record
        elif isinstance(record, dict):
            query = record
            if key:
                for part in key.split("."):
                    query = query.get(part) if isinstance(query, dict) else None
            else:
                query = next((record.get(name) for name in ("query", "question", "problem")
                              if isinstance(record.get(name), str)), None)
        else:
            query = None
        if isinstance(query, str) and query.strip() and len(query) <= 32768:
            result.append(query)
    if not result:
        raise ValueError("query file has no valid query/question strings; use --query-key for a nested field")
    return list(dict.fromkeys(result))


def extract_urls(value):
    if isinstance(value, str):
        parts = urlsplit(value.strip())
        if parts.scheme in {"http", "https"} and parts.hostname and not parts.username and not parts.password:
            yield value.strip()
    elif isinstance(value, dict):
        for child in value.values():
            yield from extract_urls(child)
    elif isinstance(value, list):
        for child in value:
            yield from extract_urls(child)


def load_url_groups(path=None, queries_path=None):
    groups = {}
    source = path or (queries_path if queries_path and Path(queries_path).suffix.lower() in {".json", ".jsonl"} else None)
    if source:
        for record in load_records(source):
            for url in extract_urls(record):
                host = (urlsplit(url).hostname or "").lower()
                # Wikipedia HTML is measured under the shared Wiki bucket.
                if host == "wikipedia.org" or host.endswith(".wikipedia.org"):
                    continue
                groups.setdefault(host, [])
                if url not in groups[host]:
                    groups[host].append(url)
    return groups


def assess_stage(stage, *, duration, min_samples, short=False):
    """Never promote a truncated stage or an unachieved offered rate."""
    reasons = []
    if stage.get("sent", 0) < min_samples:
        reasons.append("insufficient_samples")
    if not short and stage.get("dispatch_seconds", 0) < duration * .95:
        reasons.append("incomplete_window")
    if stage.get("rate_limited"):
        reasons.append("rate_limited")
    if stage.get("success_rate", 0) < .98:
        reasons.append("errors")
    if stage.get("latency_p95", math.inf) > stage.get("latency_limit", 0):
        reasons.append("latency")
    if not short and not stage.get("target_reached", False):
        reasons.append("target_not_reached")
    if stage.get("aborted_reason"):
        reasons.append(stage["aborted_reason"])
    return {"passed": not reasons, "reasons": reasons}


def select_rate_result(stages, grid):
    passed = [row for row in stages if row.get("assessment", {}).get("passed")]
    if not passed:
        return {"highest_stable_rpm": None, "smallest_passing_concurrency_cap": None,
                "observed_peak_at_that_rate": None, "status": "no_stable_baseline"}
    best = max(passed, key=lambda row: (row["offered_rpm"], -row["target_concurrency"]))
    failed_higher = [row for row in stages if row["offered_rpm"] > best["offered_rpm"]
                     and not row.get("assessment", {}).get("passed")]
    failed = min(failed_higher, key=lambda row: (row["offered_rpm"], -row["target_concurrency"]), default=None)
    # An unsuccessful capacity cap at the same RPM is not an RPM upper bound.
    ceiling_reached = best["offered_rpm"] >= max(grid)
    return {
        "highest_stable_rpm": best["offered_rpm"],
        "smallest_passing_concurrency_cap": best["target_concurrency"],
        "observed_peak_at_that_rate": best["peak_inflight"],
        "first_failed_higher_rpm": failed["offered_rpm"] if failed else None,
        "failure_reasons": failed.get("assessment", {}).get("reasons", []) if failed else [],
        "status": "tested_ceiling_passed_upper_bound_not_found" if ceiling_reached else
                  ("failure_boundary_observed" if failed else "stopped_before_upper_bound"),
    }


class RunControl:
    def __init__(self, max_seconds, request_budget):
        self.began = time.monotonic()
        self.deadline = self.began + max_seconds
        self.remaining_requests = request_budget
        self.used_requests = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def enough_time(self, seconds):
        return time.monotonic() + seconds <= self.deadline

    def allowance(self, desired):
        with self.lock:
            return min(desired, self.remaining_requests)

    def wrap(self, call):
        def counted(index):
            with self.lock:
                if self.remaining_requests <= 0:
                    self.stop_event.set()
                    return ProbeOutcome(False, error_code="benchmark_request_budget")
                if time.monotonic() >= self.deadline:
                    self.stop_event.set()
                    return ProbeOutcome(False, error_code="benchmark_runtime_budget")
                self.remaining_requests -= 1
                self.used_requests += 1
            return call(index)
        return counted


def say(message):
    print(message, flush=True)


def stage_line(label, phase, row):
    state = "PASS" if row.get("assessment", {}).get("passed") else "FAIL"
    say(f"[{state}] {label}/{phase}: offered={row['offered_rpm']:g} RPM "
        f"cap={row['target_concurrency']} actual_peak={row['peak_inflight']} "
        f"sent={row['sent']} success={row['success_rate']:.1%} "
        f"achieved={row['achieved_rpm']:.1f} RPM p95={row['latency_p95']:.2f}s "
        f"errors={row['error_codes']}")


def rest_for_window(control, seconds, label):
    if not control.enough_time(seconds):
        return False
    if seconds > 0:
        say(f"[WAIT] {label}: {seconds:.0f}s to separate quota windows")
        time.sleep(seconds)
    return True


def rate_sweep(probe, label, args, control, *, rpm_ceiling=None, progress=None):
    rows = []
    grid = [rpm for rpm in args.rpm_steps if rpm_ceiling is None or rpm <= rpm_ceiling]
    if rpm_ceiling is not None and rpm_ceiling not in grid:
        grid.append(rpm_ceiling)
        grid.sort()
    for rpm in grid:
        for concurrency in args.concurrency_steps:
            desired = math.ceil(rpm * args.duration / 60) + 1
            allowance = control.allowance(desired)
            if allowance < desired or not control.enough_time(args.duration + args.request_timeout):
                return {"stages": rows, "limit": select_rate_result(rows, grid), "stopped": "test_budget"}
            say(f"[RUN] {label}: {rpm:g} RPM / concurrency cap {concurrency} / {args.duration:g}s")
            row = probe_stage(control.wrap(probe.call), duration=args.duration, concurrency=concurrency,
                              rpm=rpm, max_requests=allowance, latency_limit=args.latency_limit,
                              stop_event=control.stop_event)
            row["phase"] = "rate"
            row["assessment"] = assess_stage(row, duration=args.duration, min_samples=args.min_samples)
            rows.append(row)
            if progress:
                progress(rows, grid)
            stage_line(label, "rate", row)
            if row.get("aborted_reason") == "interrupted":
                raise KeyboardInterrupt
            if row["assessment"]["passed"]:
                break
            if row["rate_limited"]:
                cooldown = max(args.cooldown, row.get("retry_after_seconds") or 0)
                rest_for_window(control, cooldown, label)
                return {"stages": rows, "limit": select_rate_result(rows, grid), "stopped": "rate_limited"}
            # Increase concurrent-call capacity only when successful requests
            # cannot deliver the offered rate. Errors/latency need diagnosis.
            capacity_only = set(row["assessment"]["reasons"]) <= {"target_not_reached", "insufficient_samples"}
            if not capacity_only:
                return {"stages": rows, "limit": select_rate_result(rows, grid), "stopped": "unstable"}
        else:
            return {"stages": rows, "limit": select_rate_result(rows, grid), "stopped": "concurrency_ceiling"}
    return {"stages": rows, "limit": select_rate_result(rows, grid), "stopped": "grid_exhausted"}


def wiki_dispatch(probes):
    # Cold original Wiki flow: 1 search + 10 identities + 10 extracts,
    # followed by up to 3 HTML fetches. This tests aggregate network load;
    # it does not pretend to run selection/embedding/summary model calls.
    schedule = ["wiki_api"] + ["wiki_identity"] * 10 + ["wiki_extract"] * 10 + ["wiki_html"] * 3
    def call(index):
        return probes[schedule[index % len(schedule)]].call(index)
    return call


def burst_sweep(call, label, args, control, sustainable_rpm, *, progress=None):
    rows = []
    maximum = None
    for concurrency in args.concurrency_steps:
        # A burst larger than the validated minute allowance would conflate
        # RPM and concurrent-call capacity. Leave that region unmeasured.
        if concurrency > sustainable_rpm:
            return {"stages": rows, "highest_stable_burst_concurrency": maximum,
                    "status": "minute_allowance_prevents_larger_burst"}
        if control.allowance(concurrency * args.burst_rounds) < concurrency * args.burst_rounds:
            break
        reached = True
        passed = True
        for round_index in range(args.burst_rounds):
            if not rest_for_window(control, args.cooldown, label) or not control.enough_time(args.request_timeout + 5):
                return {"stages": rows, "highest_stable_burst_concurrency": maximum, "status": "test_budget"}
            row = probe_stage(control.wrap(call), duration=5, concurrency=concurrency,
                              rpm=1e6, max_requests=concurrency, latency_limit=args.latency_limit,
                              stop_event=control.stop_event)
            row["phase"] = "burst"
            row["round"] = round_index + 1
            row["assessment"] = assess_stage(row, duration=5, min_samples=concurrency, short=True)
            if not row.get("concurrency_reached"):
                row["assessment"]["passed"] = False
                row["assessment"]["reasons"].append("concurrency_not_exercised")
                reached = False
            rows.append(row)
            if progress:
                progress(rows, maximum)
            stage_line(label, "burst", row)
            if row.get("aborted_reason") == "interrupted":
                raise KeyboardInterrupt
            passed = passed and row["assessment"]["passed"]
            if not row["assessment"]["passed"]:
                if row["rate_limited"]:
                    rest_for_window(control, max(args.cooldown, row.get("retry_after_seconds") or 0), label)
                break
        if passed:
            maximum = concurrency
        else:
            return {"stages": rows, "highest_stable_burst_concurrency": maximum,
                    "first_failed_burst_concurrency": concurrency,
                    "status": "failure_boundary_observed" if reached else "concurrency_not_exercised"}
    return {"stages": rows, "highest_stable_burst_concurrency": maximum,
            "status": "tested_ceiling_passed_upper_bound_not_found" if maximum == max(args.concurrency_steps) else "test_budget"}


def lower_rate(value, grid):
    lower = [candidate for candidate in grid if candidate < value]
    return max(lower) if lower else None


def mixed_confirmation(plans, args, control, *, progress=None):
    rounds = []
    current = {name: {"rpm": plan["rpm"], "concurrency": plan["concurrency"]} for name, plan in plans.items()}
    for attempt in range(args.confirmation_rounds):
        desired = sum(math.ceil(plan["rpm"] * args.confirmation_duration / 60) + 1 for plan in current.values())
        if control.allowance(desired) < desired or not control.enough_time(args.cooldown + args.confirmation_duration + args.request_timeout):
            return {"rounds": rounds, "confirmed": {}, "stopped": "test_budget"}
        if not rest_for_window(control, args.cooldown, "joint confirmation"):
            return {"rounds": rounds, "confirmed": {}, "stopped": "test_budget"}
        say(f"[RUN] Joint proxy load: {len(current)} buckets simultaneously, {args.confirmation_duration:g}s")
        result = {}
        pool = ThreadPoolExecutor(max_workers=len(current))
        futures = {}
        try:
            for name, plan in current.items():
                cap = math.ceil(plan["rpm"] * args.confirmation_duration / 60) + 1
                future = pool.submit(probe_stage, control.wrap(plans[name]["call"]),
                                     duration=args.confirmation_duration, concurrency=plan["concurrency"],
                                     rpm=plan["rpm"], max_requests=cap, latency_limit=args.latency_limit,
                                     stop_event=control.stop_event)
                futures[future] = name
            for future in as_completed(futures):
                name = futures[future]
                row = future.result()
                row["assessment"] = assess_stage(row, duration=args.confirmation_duration,
                                                  min_samples=args.min_samples)
                result[name] = row
                stage_line(name, "joint", row)
        except KeyboardInterrupt:
            control.stop_event.set()
            for future, name in futures.items():
                row = future.result()
                row["assessment"] = assess_stage(row, duration=args.confirmation_duration,
                                                  min_samples=args.min_samples)
                result[name] = row
            rounds.append({"attempt": attempt + 1, "settings": current, "results": result})
            if progress:
                progress(rounds)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        rounds.append({"attempt": attempt + 1, "settings": current, "results": result})
        if progress:
            progress(rounds)
        if any(row.get("aborted_reason") == "interrupted" for row in result.values()):
            raise KeyboardInterrupt
        if all(row["assessment"]["passed"] for row in result.values()):
            return {"rounds": rounds, "confirmed": current, "stopped": "passed"}
        # Shared exit capacity couples all buckets. If the aggregate fails,
        # lower all offered rates rather than advertising independent maxima.
        next_settings = {}
        for name, plan in current.items():
            rpm = lower_rate(plan["rpm"], args.rpm_steps)
            if rpm is None:
                return {"rounds": rounds, "confirmed": {}, "stopped": "no_lower_rate_in_grid"}
            next_settings[name] = dict(plan, rpm=rpm)
        current = next_settings
    return {"rounds": rounds, "confirmed": {}, "stopped": "confirmation_failed"}


def run_upstream(args, control, report, checkpoint):
    queries = load_queries(args.queries_file, args.query_key)
    groups = load_url_groups(args.urls_file, args.queries_file)
    input_web_hosts = sorted(groups)
    selected = set(args.tools.split(","))
    if selected - {"wiki", "brave", "web"}:
        raise ValueError("--tools accepts wiki,brave,web")
    if "web" in selected and not groups:
        report["unmeasured"].append("web: no non-Wikipedia URLs; provide --urls-file (one absolute URL per line)")
    if len(groups) > args.max_web_hosts:
        unselected = sorted(groups)[args.max_web_hosts:]
        report["unmeasured"].append(f"web: {len(unselected)} hostnames exceed --max-web-hosts")
        groups = {host: groups[host] for host in sorted(groups)[:args.max_web_hosts]}
    kinds = list(WIKI_KINDS) if "wiki" in selected else []
    if "brave" in selected:
        kinds.append("brave")
    if "web" in selected:
        kinds.extend("web:" + host for host in sorted(groups))
    probes = {}
    plans = {}
    rate_results = report.setdefault("upstream", {})
    with ExitStack() as stack:
        wiki_api_ceiling = None
        for label in kinds:
            kind = "web" if label.startswith("web:") else label
            urls = groups[label[4:]] if kind == "web" else []
            # InfoSeek questions are suitable search queries, not canonical
            # article titles. Identity/extract use a representative title set.
            query_set = ARTICLE_TITLES if kind in {"wiki_identity", "wiki_extract"} else queries
            try:
                probe = stack.enter_context(UpstreamProbe(kind, query_set, urls,
                            timeout=args.request_timeout, max_concurrency=max(args.concurrency_steps)))
            except ValueError:
                report["unmeasured"].append(f"{label}: invalid/missing inputs or API key; check service env")
                checkpoint()
                continue
            probes[label] = probe
            # All Wiki API varieties use the same upstream quota. Once search
            # finds a failure boundary, do not exceed its validated rate again.
            ceiling = wiki_api_ceiling if kind in {"wiki_identity", "wiki_extract", "wiki_html"} else None
            def save_rate(rows, grid):
                rate_results[label] = {"stages": list(rows), "limit": select_rate_result(rows, grid),
                                      "stopped": "running", "transport_metrics": probe.metrics()}
                checkpoint()
            outcome = rate_sweep(probe, label, args, control, rpm_ceiling=ceiling, progress=save_rate)
            outcome["scope"] = "uncached raw HTTP; manual service limits bypassed; retries=0"
            outcome["transport_metrics"] = probe.metrics()
            rate_results[label] = outcome
            if outcome["limit"].get("highest_stable_rpm") is None:
                report["unmeasured"].append(f"{label}: no stable baseline; see rate-stage errors")
            if kind == "wiki_api":
                wiki_api_ceiling = outcome["limit"].get("highest_stable_rpm")
            checkpoint()
        if "wiki" in selected:
            values = [rate_results.get(kind, {}).get("limit", {}) for kind in WIKI_KINDS]
            if all(value.get("highest_stable_rpm") for value in values):
                plans["wiki"] = {"call": wiki_dispatch(probes),
                    "rpm": min(value["highest_stable_rpm"] for value in values),
                    "concurrency": max(value["smallest_passing_concurrency_cap"] for value in values)}
            else:
                report["unmeasured"].append("wiki aggregate: all four HTTP varieties need a stable baseline")
        for label in kinds:
            if label in WIKI_KINDS:
                continue
            limit = rate_results.get(label, {}).get("limit", {})
            if limit.get("highest_stable_rpm"):
                plans[label] = {"call": probes[label].call, "rpm": limit["highest_stable_rpm"],
                                "concurrency": limit["smallest_passing_concurrency_cap"]}
        bursts = report.setdefault("burst_concurrency", {})
        for label, plan in plans.items():
            if args.skip_bursts:
                bursts[label] = {"status": "skipped", "highest_stable_burst_concurrency": None}
            else:
                def save_burst(rows, maximum):
                    bursts[label] = {"stages": list(rows), "highest_stable_burst_concurrency": maximum, "status": "running"}
                    checkpoint()
                bursts[label] = burst_sweep(plan["call"], label, args, control, plan["rpm"], progress=save_burst)
                observed = bursts[label].get("highest_stable_burst_concurrency")
                if observed is not None:
                    plan["concurrency"] = max(plan["concurrency"], observed)
            checkpoint()
        # SEARCH_WEB_* is one pair of settings applied to every hostname. Test
        # that common pair, not different per-host caps later combined without
        # validation. A slow domain may force the joint test to reduce RPM.
        web_plans = [plan for label, plan in plans.items() if label.startswith("web:")]
        if web_plans:
            common_rpm = min(plan["rpm"] for plan in web_plans)
            common_concurrency = min(plan["concurrency"] for plan in web_plans)
            for plan in web_plans:
                plan.update(rpm=common_rpm, concurrency=common_concurrency)
        if plans and not args.skip_confirmation:
            def save_joint(rounds):
                report["joint_confirmation"] = {"rounds": list(rounds), "confirmed": {}, "stopped": "running"}
                checkpoint()
            report["joint_confirmation"] = mixed_confirmation(plans, args, control, progress=save_joint)
        else:
            report["joint_confirmation"] = {"rounds": [], "confirmed": {},
                                              "stopped": "skipped" if plans else "no_stable_baselines"}
        report["transport_metrics_final"] = {label: probe.metrics() for label, probe in probes.items()}
        confirmed = report["joint_confirmation"]["confirmed"]
        recommendation = {}
        if "wiki" in confirmed:
            recommendation.update(SEARCH_WIKI_RPM=confirmed["wiki"]["rpm"],
                                  SEARCH_WIKI_CONCURRENCY=confirmed["wiki"]["concurrency"])
        if "brave" in confirmed:
            recommendation.update(SEARCH_BRAVE_RPM=confirmed["brave"]["rpm"],
                                  SEARCH_BRAVE_CONCURRENCY=confirmed["brave"]["concurrency"])
        web = [value for name, value in confirmed.items() if name.startswith("web:")]
        if web:
            # The current service has one Web configuration applied per host.
            recommendation.update(SEARCH_WEB_RPM=min(value["rpm"] for value in web),
                                  SEARCH_WEB_CONCURRENCY=min(value["concurrency"] for value in web))
        if args.duration >= 60 and args.confirmation_duration >= 120:
            report["confirmed_network_settings"] = recommendation
        else:
            report["smoke_network_settings"] = recommendation
            report["confirmed_network_settings"] = {}
        report["request_budget"] = {"logical_calls_used": control.used_requests,
                                    "logical_calls_remaining": control.remaining_requests,
                                    "redirects_are_additional_http_attempts": True}
        requested_buckets = (["wiki"] if "wiki" in selected else []) + (["brave"] if "brave" in selected else [])
        if "web" in selected:
            requested_buckets += ["web:" + host for host in input_web_hosts] if input_web_hosts else ["web:urls_not_provided"]
        missing_buckets = sorted(set(requested_buckets) - set(confirmed))
        report["coverage"] = {"requested_tools": sorted(selected), "requested_buckets": requested_buckets,
                              "confirmed_buckets": sorted(confirmed), "missing_buckets": missing_buckets,
                              "input_web_hostnames": input_web_hosts, "tested_web_hostnames": sorted(groups),
                              "all_requested_buckets_confirmed": not missing_buckets and bool(confirmed)}
        checkpoint()


def render_summary(report):
    lines = ["# Search cache capacity report", "",
             "These are the highest tested stable values, not official quotas or proven physical maxima.",
             "No deployed service or environment file was changed.", "",
             f"Status: {report.get('status', 'running')}",
             f"Mode: {report['arguments']['mode']}", ""]
    if "local" in report:
        lines.extend(["## Local synthetic load and configured boundaries", "",
                      "Warm/cold/single-flight tests include real HTTP and SQLite, with a simulated upstream.",
                      "A supported worker count here does not establish production mixed-network capacity.", "",
                      "```json", json.dumps(report["local"].get("boundaries", {}), ensure_ascii=False, indent=2), "```", "",
                      "```json", json.dumps(report["local"].get("recommendations", {}), ensure_ascii=False, indent=2), "```", ""])
        lines += ["| Workers cap | Actual worker peak | Workload | Successful requests/s | p95 seconds | Status |",
                  "|---:|---:|---|---:|---:|---|"]
        for row in report["local"].get("stages", []):
            lines.append(f"| {row.get('workers')} | {row.get('peak_executing_workers')} | {row.get('mode')} | {row.get('throughput_rps')} | "
                         f"{row.get('p95_seconds')} | {row.get('status')} |")
        lines.append("")
    if "upstream" in report:
        lines += ["## Independent upstream rate tests", "",
                  "| Probe | Highest stable RPM | Smallest passing cap tested | Actual peak | Status |",
                  "|---|---:|---:|---:|---|"]
        for name, result in report["upstream"].items():
            limit = result["limit"]
            lines.append(f"| {name} | {limit.get('highest_stable_rpm')} | "
                         f"{limit.get('smallest_passing_concurrency_cap')} | {limit.get('observed_peak_at_that_rate')} | {limit['status']} |")
        lines += ["", "## Burst concurrency", "",
                  "| Bucket | Highest exercised passing concurrency | Status |", "|---|---:|---|"]
        for name, result in report.get("burst_concurrency", {}).items():
            lines.append(f"| {name} | {result.get('highest_stable_burst_concurrency')} | {result['status']} |")
        lines += ["", "## Settings passing simultaneous proxy load", "", "```ini"]
        for key, value in report.get("confirmed_network_settings", {}).items():
            lines.append(f"{key}={value:g}")
        lines += ["```", "",
                  "An empty settings block means confirmation failed, was incomplete/skipped, or used smoke-test windows.",
                  "Concurrency settings are passing caps; only burst tests require that actual overlap reaches the cap.",
                  "Web values apply only to the tested hostnames. HTTP limits exclude LLM/embedding stages.", "",
                  "```json", json.dumps(report.get("coverage", {}), ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Unmeasured or incomplete", ""]
    lines.extend("- " + item for item in report.get("unmeasured", []))
    lines += ["", "## Interpretation", "",
              "- tested_ceiling_passed_upper_bound_not_found: at least this value passed; raise the test ceiling to search further.",
              "- failure_boundary_observed: a higher tested load failed; inspect reasons, not every failure is quota exhaustion.",
              "- concurrency_not_exercised: requests completed too quickly to actually overlap at the requested concurrency.",
              "- no_stable_baseline: diagnose authentication, proxy/network or invalid workload first.",
              "- queue length has no fixed performance maximum; adding slots does not increase throughput.",
              "- Connection capacity remains workers + queue + 16; admission capacity remains workers + queue.",
              "- Minute-scale tests are necessary for RPM. A short successful run is only a smoke test.",
              "- Results share the current proxy exit and API account with other running traffic.", ""]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=("local", "upstream", "all"), default="local")
    parser.add_argument("--tools", default="wiki,brave,web")
    parser.add_argument("--queries-file", help="txt/json/jsonl search queries; optional")
    parser.add_argument("--query-key", help="dot-separated query field, e.g. data.query")
    parser.add_argument("--urls-file", help="real training URLs, one per line or json/jsonl; grouped by hostname")
    parser.add_argument("--max-web-hosts", type=int, default=3)
    parser.add_argument("--workers", type=lambda value: number_grid(value, integer=True), default=number_grid("8,16,32,64", integer=True))
    parser.add_argument("--clients", type=int, default=256)
    parser.add_argument("--local-duration", type=float, default=3)
    parser.add_argument("--payload-bytes", type=int, default=16384)
    parser.add_argument("--producer-delay", type=float, default=.05)
    parser.add_argument("--skip-connections", action="store_true")
    parser.add_argument("--rpm-steps", type=number_grid, default=number_grid("30,60,120,150,180,200,240,360,480,600"))
    parser.add_argument("--concurrency-steps", type=lambda value: number_grid(value, integer=True), default=number_grid("1,2,4,8,16,32,64", integer=True))
    parser.add_argument("--duration", type=float, default=60, help="per upstream RPM stage, in seconds")
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--latency-limit", type=float, default=20, help="maximum passing request p95, seconds")
    parser.add_argument("--request-timeout", type=float, default=30, help="raw upstream probe deadline, seconds")
    parser.add_argument("--cooldown", type=float, default=65, help="quiet interval before bursts/confirmation and after throttling")
    parser.add_argument("--burst-rounds", type=int, default=2)
    parser.add_argument("--skip-bursts", action="store_true")
    parser.add_argument("--confirmation-duration", type=float, default=120)
    parser.add_argument("--confirmation-rounds", type=int, default=3)
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--request-budget", type=int, default=12000, help="maximum real logical calls; redirects may add HTTP attempts")
    parser.add_argument("--max-runtime", type=float, default=7200, help="budget for scheduling new upstream stages; active calls drain")
    parser.add_argument("--output-dir", help="report directory; default in local temporary directory")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("clients", "max_web_hosts", "payload_bytes", "min_samples", "burst_rounds",
                 "confirmation_rounds", "request_budget"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("local_duration", "duration", "latency_limit", "request_timeout", "cooldown",
                 "confirmation_duration", "max_runtime"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.producer_delay) or args.producer_delay < 0:
        parser.error("--producer-delay must be finite and nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or Path(tempfile.gettempdir()) / "agentflow-search-capacity" / f"{stamp}-{os.getpid()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "started_at": stamp, "status": "running", "arguments": vars(args),
              "unmeasured": [], "scope": "network cache service; no LLM/embedding or full agent training"}
    if args.duration < 60 or args.confirmation_duration < 120:
        report["unmeasured"].append("short windows selected: results are smoke tests, not sustained RPM ceilings")
    if args.skip_bursts:
        report["unmeasured"].append("burst concurrency sweep skipped")
    if args.skip_confirmation:
        report["unmeasured"].append("simultaneous proxy confirmation skipped")
    def checkpoint():
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "report.json")
        (output / "summary.md").write_text(render_summary(report), encoding="utf-8")
    say(f"[OUTPUT] {output}")
    say("[SCOPE] New temporary services/databases only. The running :8091 service is not altered.")
    checkpoint()
    exit_code = 0
    try:
        if args.mode in {"local", "all"}:
            say("[RUN] Local HTTP/SQLite capacity and configured-boundary checks")
            report["local"] = run_local_suite(workers=args.workers, burst_size=args.clients,
                duration=args.local_duration, latency_limit=args.latency_limit,
                payload_bytes=args.payload_bytes, producer_delay=args.producer_delay,
                include_connections=not args.skip_connections)
            checkpoint()
        if args.mode in {"upstream", "all"}:
            say(f"[LOAD] Real proxy traffic; <= {args.request_budget} logical calls; no retries or persistent cache.")
            control = RunControl(args.max_runtime, args.request_budget)
            run_upstream(args, control, report, checkpoint)
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        exit_code = 130
    except Exception as error:
        # Exception texts may contain URLs, proxy credentials or input queries.
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
        say(f"[ERROR] {type(error).__name__}; see partial report and verify inputs/dependencies.")
        exit_code = 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        checkpoint()
    say(f"[REPORT] {output / 'report.json'}")
    say(f"[SUMMARY] {output / 'summary.md'}")
    settings = report.get("confirmed_network_settings", {})
    if settings:
        say("[CONFIRMED] Highest jointly passing tested network settings (no headroom discount):")
        for key, value in settings.items():
            say(f"{key}={value:g}")
    elif args.mode != "local":
        say("[UNCONFIRMED] No jointly confirmed network configuration; inspect independent stage results.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
