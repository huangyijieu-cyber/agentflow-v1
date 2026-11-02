"""Verify read-only capture of HCCL port conflicts without NPU dependencies."""

import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hccl_port_diagnostics_test", ROOT / "train-roma/diagnose_hccl_ports.py")
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def address(ip, port, family=socket.AF_INET):
    packed = socket.inet_pton(family, ip)
    words = [packed[i:i + 4] for i in range(0, len(packed), 4)]
    return "".join(f"{int.from_bytes(word, sys.byteorder):08X}" for word in words) + f":{port:04X}"


def row(local, remote, state, inode):
    return f"0: {local} {remote} {state} 00000000:00000000 00:00000000 00000000 0 0 {inode}"


class PortDiagnosticsTests(unittest.TestCase):
    def test_local_source_ports_and_time_wait_are_not_limited_to_listeners(self):
        local = address("172.16.67.120", 60006)
        remote = address("172.16.67.120", 19998)
        table = "\n".join([
            "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode",
            row(local, address("0.0.0.0", 0), "0A", "101"),
            row(local, remote, "01", "102"),
            row(local, remote, "06", "0"),
            # A remote port of 60006 does not occupy local port 60006.
            row(address("172.16.67.120", 45678), address("1.2.3.4", 60006), "01", "103"),
        ])
        found = diagnostics.parse_tcp_table(table, socket.AF_INET, 60006)
        self.assertEqual([r["state"] for r in found], ["LISTEN", "ESTABLISHED", "TIME_WAIT"])
        self.assertEqual(found[0]["local"], {"address": "172.16.67.120", "port": 60006})
        self.assertEqual(found[1]["remote"], {"address": "172.16.67.120", "port": 19998})
        self.assertEqual(found[2]["inode"], "0")
        full_range = diagnostics.parse_tcp_table(table, socket.AF_INET, range(60000, 60032))
        self.assertEqual(full_range, found)

    def test_ipv6_wildcard_listener_can_be_captured(self):
        table = row(address("::", 60006, socket.AF_INET6),
                    address("::", 0, socket.AF_INET6), "0A", "204")
        found = diagnostics.parse_tcp_table(table, socket.AF_INET6, 60006)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["family"], "tcp6")
        self.assertEqual(found[0]["local"], {"address": "::", "port": 60006})

    def test_port_range_validates_before_monitoring(self):
        self.assertEqual(list(diagnostics._port_range("60000-60002")), [60000, 60001, 60002])
        for value in ("60006", "0-10", "60031-60000", "60000-65536"):
            with self.subTest(value=value), self.assertRaises(diagnostics.argparse.ArgumentTypeError):
                diagnostics._port_range(value)

    def test_process_lookup_records_only_socket_owners_and_allowed_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = root / "123"
            (process / "fd").mkdir(parents=True)
            (process / "fd/7").symlink_to("socket:[102]")
            (process / "comm").write_text("ray::WorkerDict\n")
            # /proc stat fields 4 (ppid) and 22 (starttime).
            (process / "stat").write_text("123 (worker with spaces) S 11 " + "0 " * 17 + "99\n")
            (process / "environ").write_bytes(
                b"RANK=6\0LOCAL_RANK=0\0ASCEND_RT_VISIBLE_DEVICES=6\0SECRET_TOKEN=never-log-this\0")
            owners = diagnostics.find_socket_owners({"102", "0", "999"}, root)
        encoded = json.dumps(owners)
        self.assertEqual(len(owners["102"]), 1)
        self.assertEqual(owners["102"][0]["pid"], 123)
        self.assertIn("ray::WorkerDict", encoded)
        self.assertIn('"RANK": "6"', encoded)
        self.assertIn('"ASCEND_RT_VISIBLE_DEVICES": "6"', encoded)
        self.assertNotIn("SECRET_TOKEN", encoded)
        self.assertNotIn("never-log-this", encoded)
        self.assertEqual(owners.get("0", []), [])
        self.assertEqual(owners.get("999", []), [])

    def test_monitor_retains_live_owner_and_time_wait_after_the_port_disappears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "net").mkdir()
            tcp = root / "net/tcp"
            local, remote = address("172.16.67.120", 60006), address("172.16.67.120", 8080)
            tcp.write_text(row(local, remote, "01", "102"))
            process = root / "123"
            (process / "fd").mkdir(parents=True)
            (process / "fd/7").symlink_to("socket:[102]")
            (process / "comm").write_text("rollout-worker\n")
            output = root / "capture.jsonl"
            remaining = iter((row(local, remote, "06", "0"), ""))

            def advance(_):
                try:
                    tcp.write_text(next(remaining))
                except StopIteration:
                    raise KeyboardInterrupt

            with mock.patch.object(diagnostics.time, "sleep", side_effect=advance), \
                 mock.patch.object(diagnostics.socket, "socket", side_effect=AssertionError("no network sockets")), \
                 mock.patch.object(diagnostics, "_metadata", return_value={"port": 60006}):
                diagnostics.monitor(60006, 0.1, 30, output, root)
            events = [json.loads(line) for line in output.read_text().splitlines()]
            present = [event for event in events if event["event"] == "socket_present"]
            self.assertEqual([event["state"] for event in present], ["ESTABLISHED", "TIME_WAIT"])
            self.assertEqual(present[0]["owners"][0]["pid"], 123)
            self.assertEqual(present[1]["owners"], [])
            self.assertEqual(len([event for event in events if event["event"] == "socket_closed"]), 2)
            self.assertEqual(events[-1]["event"], "summary")
            self.assertEqual(events[-1]["stop_reason"], "interrupted")
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                diagnostics.monitor(60006, 0.1, 1, output, root)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
