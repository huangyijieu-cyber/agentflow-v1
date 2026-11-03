#!/usr/bin/env python3
"""Observe one HCCL host port through Linux /proc without changing training.

Run once in a separate terminal before the normal training command. JSONL events
record socket_present, socket_changed and socket_closed observations, including
TIME_WAIT sockets which have no owning process. No sockets are opened or bound.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import time


STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV",
    "04": "FIN_WAIT1", "05": "FIN_WAIT2", "06": "TIME_WAIT",
    "07": "CLOSE", "08": "CLOSE_WAIT", "09": "LAST_ACK",
    "0A": "LISTEN", "0B": "CLOSING", "0C": "NEW_SYN_RECV",
}
ENV_KEYS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "ASCEND_RT_VISIBLE_DEVICES",
    "HCCL_IF_BASE_PORT", "HCCL_HOST_SOCKET_PORT_RANGE",
)


def _endpoint(value, family):
    address, port = value.split(":")
    # /proc represents each 32-bit address word in native byte order.
    raw = b"".join(struct.pack("=I", int(address[i:i + 8], 16))
                   for i in range(0, len(address), 8))
    return {"address": socket.inet_ntop(family, raw), "port": int(port, 16)}


def parse_tcp_table(text, family, port):
    """Return entries whose LOCAL port matches, including IPv6 and TIME_WAIT."""
    family = socket.AF_INET6 if family in (socket.AF_INET6, "tcp6") else socket.AF_INET
    ports = {port} if isinstance(port, int) else set(port)
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10 or ":" not in parts[0] or parts[0] == "sl":
            continue
        try:
            local = _endpoint(parts[1], family)
            if local["port"] not in ports:
                continue
            rows.append({
                "family": "tcp6" if family == socket.AF_INET6 else "tcp",
                "local": local, "remote": _endpoint(parts[2], family),
                "state": STATES.get(parts[3].upper(), parts[3]), "inode": parts[9],
            })
        except (ValueError, OSError, struct.error):
            continue
    return rows


def _read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return None


def find_socket_owners(inodes, proc_root=Path("/proc")):
    """Resolve visible socket FDs; read only explicitly permitted process data."""
    wanted = {str(inode) for inode in inodes if str(inode) != "0"}
    result = {inode: [] for inode in wanted}
    if not wanted:
        return result
    for process in Path(proc_root).iterdir():
        if not process.name.isdigit():
            continue
        matched = set()
        try:
            for fd in (process / "fd").iterdir():
                try:
                    link = os.readlink(fd)
                    if link.startswith("socket:[") and link.endswith("]") and link[8:-1] in wanted:
                        matched.add(link[8:-1])
                except OSError:
                    pass
        except OSError:
            continue
        if not matched:
            continue
        owner = {"pid": int(process.name), "name": _read(process / "comm")}
        stat = _read(process / "stat")
        if stat and ")" in stat:
            fields = stat.rsplit(")", 1)[1].split()
            try:
                owner.update(ppid=int(fields[1]), start_time_ticks=int(fields[19]))
            except (IndexError, ValueError):
                pass
        try:
            environment = dict(entry.split("=", 1) for entry in
                               (process / "environ").read_bytes().decode(errors="replace").split("\0")
                               if "=" in entry)
            owner["env"] = {key: environment[key] for key in ENV_KEYS if key in environment}
        except OSError:
            owner["env"] = None
        for inode in matched:
            result[inode].append(owner)
    return result


def _identity(row):
    return (row["family"], row["local"]["address"], row["local"]["port"],
            row["remote"]["address"], row["remote"]["port"], row["inode"])


def _metadata(proc_root, port):
    versions = {}
    for package in ("torch", "torch-npu", "verl", "vllm", "vllm-ascend"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "ports": [port] if isinstance(port, int) else sorted(port),
        "pid": os.getpid(), "python": sys.version.split()[0],
        "versions": versions, "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
        "boot_time": next((line.split()[1] for line in (_read(proc_root / "stat") or "").splitlines()
                           if line.startswith("btime ")), None),
        "ip_local_port_range": _read(proc_root / "sys/net/ipv4/ip_local_port_range"),
        "ip_local_reserved_ports": _read(proc_root / "sys/net/ipv4/ip_local_reserved_ports"),
        "env": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
    }


def monitor(port, interval, duration, output, proc_root=Path("/proc")):
    """Write changed observations until the duration expires or Ctrl+C."""
    proc_root, output = Path(proc_root), Path(output)
    if not (proc_root / "net/tcp").is_file():
        raise RuntimeError("Linux /proc/net/tcp is required; run this on the training server")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous, owner_cache = {}, {}
    started = time.monotonic()
    samples = events = 0
    reason = "duration"
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        def emit(event, **details):
            stream.write(json.dumps({"event": event,
                                     "timestamp": datetime.now(timezone.utc).isoformat(),
                                     "elapsed_seconds": round(time.monotonic() - started, 6),
                                     **details}, ensure_ascii=True) + "\n")

        emit("metadata", **_metadata(proc_root, port))
        print(f"Read-only monitor started: port={port}, output={output}", flush=True)
        print("Start training normally in another terminal; Ctrl+C stops only this monitor.", flush=True)
        try:
            while time.monotonic() - started < duration:
                rows = parse_tcp_table((proc_root / "net/tcp").read_text(), "tcp", port)
                tcp6 = _read(proc_root / "net/tcp6")
                rows += parse_tcp_table(tcp6 or "", "tcp6", port)
                inodes = {row["inode"] for row in rows}
                unresolved = {inode for inode in inodes if not owner_cache.get(inode)}
                owner_cache.update(find_socket_owners(unresolved, proc_root))
                owner_cache = {inode: owners for inode, owners in owner_cache.items() if inode in inodes}
                current = {_identity(row): dict(row, owners=owner_cache.get(row["inode"], [])) for row in rows}
                for key, row in current.items():
                    if previous.get(key) != row:
                        emit("socket_changed" if key in previous else "socket_present", **row)
                        events += 1
                for key, row in previous.items():
                    if key not in current:
                        emit("socket_closed", **row)
                        events += 1
                previous = current
                samples += 1
                time.sleep(min(interval, max(0, duration - (time.monotonic() - started))))
        except KeyboardInterrupt:
            reason = "interrupted"
        except OSError as error:
            reason = "read_error"
            emit("read_error", error_type=type(error).__name__, message=str(error))
            raise
        finally:
            emit("summary", samples=samples, socket_events=events, stop_reason=reason,
                 note="No observation does not prove the port was always free; short events may fall between samples.")
    print(f"Monitor stopped: samples={samples}, socket_events={events}, output={output}", flush=True)


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _port_range(value):
    try:
        start, end = map(int, value.split("-"))
        if 1 <= start <= end <= 65535:
            return range(start, end + 1)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected START-END within 1..65535")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ports = parser.add_mutually_exclusive_group()
    ports.add_argument("--port", type=int, default=60006)
    ports.add_argument("--port-range", type=_port_range, help="capture all ports in START-END")
    parser.add_argument("--interval", type=_positive, default=0.1)
    parser.add_argument("--duration", type=_positive, default=900)
    parser.add_argument("--output", type=Path, default=Path("/tmp/agentflow-hccl-port-diagnosis.jsonl"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        monitor(args.port_range if args.port_range is not None else args.port,
                args.interval, args.duration, args.output)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Diagnostic failed: {error}\n")


if __name__ == "__main__":
    main()
