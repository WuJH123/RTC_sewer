"""Low-rate Windows/GPU telemetry for long Formal stages."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from collections import deque
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


def _classify(rows: list[dict[str, object]]) -> str:
    """Classify the current limiting resource from the recent sample window."""
    if not rows:
        return "UNKNOWN"

    def values(key: str) -> list[float]:
        result = []
        for row in rows:
            value = row.get(key)
            if value is not None:
                try:
                    result.append(float(value))
                except (TypeError, ValueError):
                    pass
        return result

    available = values("available_ram_gb")
    used = values("total_ram_used_gb")
    pagefile = values("pagefile_used_gb")
    if (
        (available and min(available) < 2.5)
        or (used and max(used) > 13.0)
        or (len(pagefile) >= 2 and pagefile[-1] - pagefile[0] > 0.25)
    ):
        return "RAM_BOUND"

    gpu = values("gpu_util_percent")
    gpu_mem = values("gpu_memory_used_mb")
    gpu_total = values("gpu_memory_total_mb")
    cpu = values("total_cpu_percent")
    read = values("disk_read_MBps")
    write = values("disk_write_MBps")
    gpu_median = statistics.median(gpu) if gpu else 0.0
    memory_headroom = True
    if gpu_mem and gpu_total:
        memory_headroom = statistics.median(gpu_mem) < 0.90 * statistics.median(gpu_total)
    if gpu_median >= 85.0:
        return "GPU_COMPUTE_BOUND"
    if gpu_median < 65.0 and cpu and statistics.median(cpu) >= 70.0:
        return "CPU_BOUND"
    if gpu_median < 65.0 and (read or write):
        io_median = statistics.median(read or [0.0]) + statistics.median(write or [0.0])
        if io_median >= 5.0 and (not cpu or statistics.median(cpu) < 70.0):
            return "IO_BOUND"
    if gpu_median < 65.0 and memory_headroom:
        return "GPU_STARVED"
    return "UNKNOWN"


def _write_runtime_status(path: Path, row: dict[str, object], samples: int) -> None:
    value = {
        "timestamp": row.get("timestamp"),
        "pid": row.get("pid"),
        "stage": row.get("stage", ""),
        "epoch": row.get("epoch", ""),
        "batch": row.get("batch", ""),
        "windows_seen": row.get("windows_seen", ""),
        "windows_per_sec": row.get("windows_per_sec", ""),
        "bottleneck": row.get("bottleneck", "UNKNOWN"),
        "telemetry_samples": samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--runtime-status-output", type=Path, default=None)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--flush-sec", type=float, default=30.0)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "pid", "stage", "epoch", "batch", "windows_seen", "windows_per_sec",
        "gpu_index",
        "gpu_util_percent", "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_power_w",
        "gpu_temperature_c", "sm_clock_mhz", "memory_clock_mhz", "total_cpu_percent",
        "per_core_cpu_percent", "rss_process_mb", "total_ram_used_gb", "available_ram_gb",
        "pagefile_used_gb", "disk_read_MBps", "disk_write_MBps", "process_cpu_percent",
        "bottleneck",
    ]
    exists = args.output.exists()
    if exists:
        with args.output.open("r", encoding="utf-8") as existing:
            existing_fields = next(csv.reader(existing), [])
        if existing_fields:
            # Preserve append compatibility for pre-autotuner telemetry files.
            fieldnames = existing_fields
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        last = time.monotonic()
        last_io = None
        last_flush = last
        recent: deque[dict[str, object]] = deque(maxlen=24)
        runtime_status = args.runtime_status_output or args.output.with_name("FORMAL_RUNTIME_STATUS.json")
        while True:
            now = time.monotonic()
            try:
                cpu = psutil.cpu_percent(interval=None)
                cores = psutil.cpu_percent(interval=None, percpu=True)
                process_cpu = process.cpu_percent(interval=None)
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
                    "process_cpu_percent": process_cpu,
                }
                recent.append(row)
                row["bottleneck"] = _classify(list(recent))
                writer.writerow({key: row.get(key, "") for key in fieldnames})
                if now - last_flush >= args.flush_sec:
                    handle.flush()
                    _write_runtime_status(runtime_status, row, len(recent))
                    last_flush = now
                print(json.dumps(row, ensure_ascii=False), flush=True)
            except psutil.NoSuchProcess:
                handle.flush()
                return 0
            time.sleep(max(0.1, args.interval_sec))
            last = now


if __name__ == "__main__":
    raise SystemExit(main())
