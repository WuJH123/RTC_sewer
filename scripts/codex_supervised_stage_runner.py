"""Small durable launcher for long Project6 stages.

It owns one child process, persists a heartbeat/status JSON, and never retries
or restarts the child.  A later Codex turn can inspect the same status file and
continue monitoring the original PID.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", type=Path, required=True)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--stdout", type=Path, required=True)
    ap.add_argument("--stderr", type=Path, required=True)
    ap.add_argument("--heartbeat-sec", type=float, default=30.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        ap.error("a child command is required after --")

    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status: dict[str, object] = {
        "status": "starting",
        "command": command,
        "cwd": str(args.cwd),
        "launcher_pid": os.getpid(),
        "child_pid": None,
        "started_at": _now(),
        "last_heartbeat": _now(),
        "elapsed_sec": 0.0,
        "exit_code": None,
    }
    _write(args.status, status)
    with args.stdout.open("ab") as stdout, args.stderr.open("ab") as stderr:
        child = subprocess.Popen(command, cwd=str(args.cwd), stdout=stdout, stderr=stderr)
        status["child_pid"] = child.pid
        status["status"] = "running"
        _write(args.status, status)
        while child.poll() is None:
            status["last_heartbeat"] = _now()
            status["elapsed_sec"] = round(time.time() - started, 3)
            status["stdout_bytes"] = args.stdout.stat().st_size
            status["stderr_bytes"] = args.stderr.stat().st_size
            _write(args.status, status)
            time.sleep(max(1.0, args.heartbeat_sec))
        status["exit_code"] = child.returncode
        status["status"] = "completed" if child.returncode == 0 else "failed"
        status["finished_at"] = _now()
        status["last_heartbeat"] = _now()
        status["elapsed_sec"] = round(time.time() - started, 3)
        status["stdout_bytes"] = args.stdout.stat().st_size
        status["stderr_bytes"] = args.stderr.stat().st_size
        _write(args.status, status)
    return int(child.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
