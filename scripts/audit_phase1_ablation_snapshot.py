from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["event_id", "phase", "actuator_id", "action_direction", "action_delta"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"E:\RTC_sewer\Project6")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "outputs" / "ablation_all109"
    dataset_path = out_dir / "exact_no_control_action_effect_dataset.csv"
    results_path = out_dir / "single_actuator_ablation_results.csv"
    config_path = root / "configs" / "wuhan_project6.yaml"

    command = (
        r'"E:\RTC_sewer\Project6\.venv\Scripts\python.exe" '
        r'scripts\76_generate_no_control_single_actuator_ablation.py '
        r'--config E:\RTC_sewer\Project6\configs\wuhan_project6.yaml '
        r'--event-ids T20_D150_chicago_center,T30_D210_chicago_late,T50_D240_block,'
        r'T75_D210_chicago_center,T75_D300_chicago_late,T100_D240_double_peak,'
        r'T100_D300_block,T50_D300_double_peak,T20_D300_chicago_center,'
        r'T20_D300_chicago_early,T20_D300_chicago_late,T20_D300_block,'
        r'T20_D300_double_peak,T30_D300_chicago_center,T30_D300_chicago_early,'
        r'T30_D300_chicago_late,T30_D300_block,T30_D300_double_peak,'
        r'T50_D300_chicago_center,T50_D300_chicago_early,T50_D300_chicago_late,'
        r'T50_D300_block,T50_D240_double_peak,T75_D300_chicago_center,'
        r'T75_D300_chicago_early,T75_D240_chicago_late,T75_D300_block,'
        r'T75_D300_double_peak,T100_D300_chicago_center,T100_D300_chicago_early '
        r'--max-events 10 --max-actuators 0 --samples-per-phase 1 --delta 0.05 '
        r'--hold-steps 2 --workers 16 --resume '
        r'--out-dir E:\RTC_sewer\Project6\outputs\ablation_all109'
    )
    requested_events = command.split("--event-ids ", 1)[1].split(" --max-events", 1)[0].split(",")
    selected_events = requested_events[:10]

    parse_error = ""
    try:
        data = pd.read_csv(dataset_path, low_memory=False)
    except Exception as exc:
        data = pd.DataFrame()
        parse_error = repr(exc)
    duplicate_case_ids = int(data.duplicated("case_id", keep=False).sum()) if "case_id" in data else None
    available_key = [column for column in KEY_COLUMNS if column in data]
    duplicate_semantic_keys = int(data.duplicated(available_key, keep=False).sum()) if available_key else None
    numeric = data.select_dtypes(include="number")
    nan_counts = {column: int(value) for column, value in numeric.isna().sum().items() if value}
    inf_counts = {
        column: int((series == float("inf")).sum() + (series == float("-inf")).sum())
        for column, series in numeric.items()
        if ((series == float("inf")) | (series == float("-inf"))).any()
    }
    phases = sorted(data.get("phase", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    event_counts = data.get("event_id", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    actuator_count = int(data.get("actuator_id", pd.Series(dtype=str)).nunique())
    direction_counts = data.get("action_direction", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    expected_cases = 10 * 3 * 109 * 2
    completed = int(data.get("case_id", pd.Series(dtype=str)).nunique())
    failures_path = out_dir / "failures.csv"
    failed = 0
    if failures_path.exists() and failures_path.stat().st_size:
        try:
            failed = int(len(pd.read_csv(failures_path)))
        except Exception:
            failed = -1

    source_paths = [
        root / "scripts" / "76_generate_no_control_single_actuator_ablation.py",
        root / "sewerrtc" / "simulation" / "pyswmm_runner.py",
        root / "sewerrtc" / "control" / "horizon_rollout.py",
        root / "sewerrtc" / "control" / "horizon_action_features.py",
        root / "sewerrtc" / "simulation" / "kpi_metrics.py",
    ]
    latest_artifact = max(
        [path for path in out_dir.rglob("*") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        default=dataset_path,
    )
    integrity = {
        "parse_error": parse_error,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "duplicate_case_id_rows": duplicate_case_ids,
        "duplicate_semantic_key_rows": duplicate_semantic_keys,
        "semantic_key_columns_available": available_key,
        "required_key_gaps": ["direction column is named action_direction", "hold_steps absent from exact dataset"],
        "nan_counts_numeric_nonzero": nan_counts,
        "inf_counts_numeric_nonzero": inf_counts,
        "event_counts": event_counts,
        "phase_values": phases,
        "actuator_count": actuator_count,
        "direction_counts": direction_counts,
        "expected_case_count": expected_cases,
        "completed_case_count": completed,
        "coverage_fraction": completed / expected_cases,
        "complete": completed == expected_cases and not parse_error,
        "resume_key": "case_id=sha1(event_id|actuator_id|elapsed_min|delta|hold_steps)[:16]",
        "resume_duplicate_risk": "low for existing case_id rows; interrupted details without exact rows will be rerun",
        "units": {
            "PFV_H": "m3; sum(priority flooding rate in m3/s) * dt_sec over future horizon",
            "TFV_H": "m3; sum(all-node flooding rate in m3/s) * dt_sec over future horizon",
            "peak_TFV_rate_H": "m3/s; maximum all-node flooding rate over future horizon",
            "effect_*": "candidate minus same-time no_control reference; negative is improvement",
        },
    }
    manifest = {
        "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
        "process_status": "not_running_observed",
        "pid": 37096,
        "worker_launcher_pid": 34840,
        "start_time_local": "2026-07-12T02:57:09",
        "end_time": None,
        "end_time_evidence": f"process absent; latest artifact mtime={datetime.fromtimestamp(latest_artifact.stat().st_mtime).isoformat()}",
        "python_executable": r"E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
        "command_line": command,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_sha256_snapshot_after_process_exit": {str(path): sha256(path) for path in source_paths},
        "source_hash_caveat": "Hashes describe files at Phase-1 audit time, not a guaranteed copy of modules loaded at process start.",
        "requested_event_ids": requested_events,
        "selected_event_ids_by_max_events": selected_events,
        "max_events": 10,
        "actuator_count": 109,
        "delta": 0.05,
        "hold_steps": 2,
        "expected_case_count": expected_cases,
        "completed_case_count": completed,
        "failed_case_count": failed,
        "duplicate_case_count": duplicate_case_ids,
        "current_dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "integrity": integrity,
    }
    manifest_path = out_dir / "run_manifest_phase0.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit_dir = root / "outputs" / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "ablation_dataset_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "integrity": integrity}, indent=2))


if __name__ == "__main__":
    main()
