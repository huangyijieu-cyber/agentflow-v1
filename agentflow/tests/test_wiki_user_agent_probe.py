"""Offline regressions for the direct Wiki User-Agent diagnostic script."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests


SCRIPT = Path(__file__).resolve().parents[2] / "train-roma" / "test_wiki_user_agent.py"
SPEC = importlib.util.spec_from_file_location("wiki_user_agent_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class FakeClock:
    def __init__(self):
        self.value = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def response(status=200, body=None, headers=None, sent_ua=None, latency=0):
    value = requests.Response()
    value.status_code = status
    value.headers.update(headers or {})
    value._content = json.dumps(body if body is not None else {"query": {"search": []}}).encode()
    value._content_consumed = True
    value.encoding = "utf-8"
    value.probe_sent_ua = sent_ua
    value.probe_latency = latency
    return value


class FakeSession(requests.Session):
    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = list(outcomes)
        self.calls = []
        self.closed_for_test = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs, os.environ.get("WIKIMEDIA_USER_AGENT")))
        if not self.outcomes:
            raise AssertionError("Unexpected extra request (possibly a retry)")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.clock.value += outcome.probe_latency
        requested_ua = kwargs.get("headers", {}).get("User-Agent")
        actual_ua = outcome.probe_sent_ua if outcome.probe_sent_ua is not None else requested_ua
        outcome.request = requests.Request("GET", url, headers={"User-Agent": actual_ua}).prepare()
        outcome.url = outcome.request.url
        return outcome

    def close(self):
        self.closed_for_test = True
        super().close()


class WikiUserAgentProbeTests(unittest.TestCase):
    def run_fake(self, outcomes, argv=(), env=None):
        session = FakeSession(outcomes)
        clock = FakeClock()
        session.clock = clock
        configured_env = {"HTTPS_PROXY": "http://proxy.example:18090"}
        configured_env.update(env or {})
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "run"
            with patch.dict(os.environ, configured_env, clear=True), \
                 patch.object(probe.requests, "Session", return_value=session), \
                 patch.object(probe.time, "monotonic", side_effect=clock.monotonic), \
                 patch.object(probe.time, "sleep", side_effect=clock.sleep), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                before = os.environ.get("WIKIMEDIA_USER_AGENT")
                code = probe.main(["--requests", "3", "--output-dir", str(output), *argv])
                self.assertEqual(os.environ.get("WIKIMEDIA_USER_AGENT"), before)
            report = json.loads((output / "report.json").read_text())
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        return code, report, events, session, clock

    def test_success_preserves_proxy_ua_and_paces_without_hidden_retries(self):
        ua = "ResearchProbe/1.0 (https://example.org/contact)"
        code, report, events, session, clock = self.run_fake(
            [response(), response(), response()], env={"WIKIMEDIA_USER_AGENT": ua})
        self.assertEqual(code, 0)
        self.assertEqual(report["attempts"], 3)
        self.assertEqual(report["ok_count"], 3)
        self.assertEqual(report["configured_user_agent"], ua)
        self.assertEqual(report["ua_classification"], "unknown")
        self.assertTrue(session.trust_env)
        self.assertTrue(session.closed_for_test)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(session.get_adapter("https://en.wikipedia.org").max_retries.total, 0)
        self.assertEqual(len(events), 3)
        self.assertEqual(len(clock.sleeps), 2)
        self.assertTrue(all(value >= 1 for value in clock.sleeps))
        for url, kwargs, env_ua in session.calls:
            self.assertEqual(url, "https://en.wikipedia.org/w/api.php")
            self.assertEqual(kwargs["headers"]["User-Agent"], ua)
            self.assertEqual(env_ua, ua)
            self.assertFalse(kwargs["allow_redirects"])
            self.assertTrue(kwargs["stream"])
            self.assertEqual(kwargs["params"]["list"], "search")
            self.assertEqual(kwargs["params"]["srlimit"], 1)
        self.assertEqual(len({call[1]["params"]["requestid"] for call in session.calls}), 3)
        self.assertTrue(all(event["ua_matches_requested"] for event in events))

    def test_429_records_retry_after_and_stops_without_sleep_or_retry(self):
        code, report, events, session, clock = self.run_fake([
            response(429, {"error": "rate limited"}, {"Retry-After": "120"}),
            response(),
        ])
        self.assertEqual(code, 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["ok_count"], 0)
        self.assertEqual(report["ua_classification"], "unknown")
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(events[0]["status_code"], 429)
        self.assertEqual(events[0]["response_headers"]["retry-after"], "120")
        self.assertIn("rate limited", events[0]["body_excerpt"])

    def test_http_200_api_error_is_not_success(self):
        code, report, events, session, _ = self.run_fake([
            response(200, {"error": {"code": "ratelimited", "info": "Try later"}}),
            response(),
        ])
        self.assertEqual(code, 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["ok_count"], 0)
        self.assertEqual(events[0]["api_error_code"], "ratelimited")
        self.assertEqual(len(session.calls), 1)

    def test_explicit_ua_overrides_env_and_actual_mismatch_is_detected(self):
        code, report, events, session, _ = self.run_fake(
            [response(sent_ua="UnexpectedUA/1.0"), response()],
            ["--user-agent", "ExplicitProbe/1.0"],
            {"WIKIMEDIA_USER_AGENT": "EnvironmentProbe/1.0"},
        )
        self.assertEqual(code, 1)
        self.assertEqual(report["configured_user_agent"], "ExplicitProbe/1.0")
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0][2], "ExplicitProbe/1.0")
        self.assertFalse(events[0]["ua_matches_requested"])
        self.assertEqual(events[0]["client_sent_user_agent"], "UnexpectedUA/1.0")

    def test_network_error_is_reported_without_retry(self):
        code, report, events, session, clock = self.run_fake([
            requests.exceptions.ProxyError("proxy connection failed"), response(),
        ])
        self.assertEqual(code, 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(clock.sleeps, [])
        self.assertTrue(events[0]["error_type"])
        self.assertTrue(report["stop_reason"])

    def test_interrupt_saves_partial_report(self):
        code, report, events, session, _ = self.run_fake([response(), KeyboardInterrupt()])
        self.assertEqual(code, 130)
        self.assertEqual(report["ok_count"], 1)
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(session.closed_for_test)

    def test_slow_requests_do_not_trigger_catch_up_bursts(self):
        code, _, events, _, clock = self.run_fake([
            response(latency=2.5), response(latency=0.1), response(latency=0.1),
        ])
        self.assertEqual(code, 0)
        starts = [event["started_offset_seconds"] for event in events]
        self.assertGreaterEqual(starts[1] - starts[0], 2.5)
        self.assertGreaterEqual(starts[2] - starts[1], 1)
        self.assertEqual(len(clock.sleeps), 1)

    def test_missing_proxy_requires_explicit_direct_opt_in_before_any_http(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "run"
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(probe.requests.utils, "get_environ_proxies", return_value={}), \
                 patch.object(probe.requests, "Session") as factory, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    probe.main(["--requests", "1", "--output-dir", str(output)])
            self.assertEqual(caught.exception.code, 2)
            factory.assert_not_called()
            self.assertFalse(output.exists())

    def test_other_http_errors_and_redirects_stop_without_following(self):
        for status in (302, 403, 502):
            with self.subTest(status=status):
                code, report, _, session, _ = self.run_fake([
                    response(status), response(),
                ])
                self.assertEqual(code, 1)
                self.assertEqual(report["ok_count"], 0)
                self.assertEqual(len(session.calls), 1)
                self.assertFalse(session.calls[0][1]["allow_redirects"])

    def test_body_read_failure_preserves_original_429_evidence(self):
        item = response(429, headers={"Retry-After": "30"})
        item.iter_content = lambda **kwargs: self.failed_body()
        code, report, events, _, _ = self.run_fake([item])
        self.assertEqual(code, 1)
        self.assertEqual(report["stop_reason"], "http_429")
        self.assertEqual(events[0]["status_code"], 429)
        self.assertEqual(events[0]["response_headers"]["retry-after"], "30")
        self.assertEqual(events[0]["body_read_error"], "ConnectionError")

    @staticmethod
    def failed_body():
        yield b"rate limited"
        raise requests.exceptions.ConnectionError("response stalled")

    def test_parser_rejects_invalid_request_counts_rates_and_timeouts(self):
        for flag, values in {
            "--requests": ["0", "-1", "1.2"],
            "--rpm": ["0", "-1", "nan", "inf"],
            "--timeout": ["0", "-1", "nan", "inf"],
        }.items():
            for value in values:
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        probe.build_parser().parse_args([flag, value])


if __name__ == "__main__":
    unittest.main()
