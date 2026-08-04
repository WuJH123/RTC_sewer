"""Low-rate Windows/GPU telemetry for long Formal stages."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


GPU_QUERY = (
    "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,"
    "clocks.sm,clocks.mem"
)
GPU_NAMES = [
    "gpu_index",
    "gpu_util_percent",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_power_w",
    "gpu_temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
]


def _gpu() -> dict[str, object]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        values = [x.strip() for x in out.splitlines()[0].split(",")]
        result: dict[str, object] = dict(zip(GPU_NAMES, values))
        for key in GPU_NAMES[1:]:
            result[key] = float(result[key])
        result["gpu_index"] = int(float(result["gpu_index"]))
        return result
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {key: None for key in GPU_NAMES}


def _status(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--flush-sec", type=float, default=30.0)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "pid", "stage", "epoch", "batch", "windows_seen", "windows_per_sec",
        "gpu_util_percent", "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_power_w",
        "gpu_temperature_c", "sm_clock_mhz", "memory_clock_mhz", "total_cpu_percent",
        "per_core_cpu_percent", "rss_process_mb", "total_ram_used_gb", "available_ram_gb",
        "pagefile_used_gb", "disk_read_MBps", "disk_write_MBps",
    ]
    exists = args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        last = time.monotonic()
        last_io = None
        last_flush = last
        while True:
            now = time.monotonic()
            try:
                cpu = psutil.cpu_percent(interval=None)
                cores = psutil.cpu_percent(interval=None, percpu=True)
                vm = psutil.virtual_memory()
                io = process.io_counters()
                if last_io is None:
                    read_rate = write_rate = 0.0
                else:
                    delta = max(now - last, 1e-6)
                    read_rate = (io.read_bytes - last_io[0]) / delta / 1e6
                    write_rate = (io.write_bytes - last_io[1]) / delta / 1e6
                last_io = (io.read_bytes, io.write_bytes)
                gpu = _gpu()
                status = _status(args.status_file)
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pid": args.pid,
                    "stage": status.get("stage", ""),
                    "epoch": status.get("epoch", ""),
                    "batch": status.get("batch", ""),
                    "windows_seen": status.get("windows_seen", ""),
                    "windows_per_sec": status.get("windows_per_sec", ""),
                    **gpu,
                    "total_cpu_percent": cpu,
                    "per_core_cpu_percent": json.dumps(cores, separators=(",", ":")),
                    "rss_process_mb": process.memory_info().rss / 1e6,
                    "total_ram_used_gb": vm.used / 1e9,
                    "available_ram_gb": vm.available / 1e9,
                    "pagefile_used_gb": psutil.swap_memory().used / 1e9,
                    "disk_read_MBps": read_rate,
                    "disk_write_MBps": write_rate,
                }
                writer.writerow(row)
                if now - last_flush >= args.flush_sec:
                    handle.flush()
                    last_flush = now
                print(json.dumps(row, ensure_ascii=False), flush=True)
            except psutil.NoSuchProcess:
                handle.flush()
                return 0
            time.sleep(max(0.1, args.interval_sec))
            last = now


if __name__ == "__main__":
    raise SystemExit(main())
