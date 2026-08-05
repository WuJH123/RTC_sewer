"""Durable stdlib supervisor for one long-running Project6 stage."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    child = subprocess.Popen(
        command,
        cwd=str(args.cwd),
        stdout=args.stdout.open("w", encoding="utf-8", buffering=1),
        stderr=args.stderr.open("w", encoding="utf-8", buffering=1),
        text=True,
    )
    payload = {
        "state": "running",
        "supervisor_pid": __import__("os").getpid(),
        "child_pid": child.pid,
        "command": command,
        "started_utc": _now(),
        "heartbeat_utc": _now(),
    }
    _write(args.status, payload)
    while child.poll() is None:
        payload["heartbeat_utc"] = _now()
        payload["elapsed_s"] = round(time.time() - started, 1)
        _write(args.status, payload)
        time.sleep(30)
    payload.update(
        {
            "state": "completed" if child.returncode == 0 else "failed",
            "exit_code": child.returncode,
            "heartbeat_utc": _now(),
            "elapsed_s": round(time.time() - started, 1),
        }
    )
    _write(args.status, payload)
    return int(child.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
