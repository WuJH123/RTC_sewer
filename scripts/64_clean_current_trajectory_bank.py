from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _detail_event_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_detail"):
        stem = stem[: -len("_detail")]
    event_id, sep, _policy = stem.rpartition("__")
    return event_id if sep else stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--apply", action="store_true", help="Write filtered summary and quarantine stale detail files.")
    ap.add_argument("--quarantine-stale-details", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = ensure_dir(cfg_path(cfg, "outputs.data_bank_train"))
    rain_table_path = cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"
    rain = pd.read_csv(rain_table_path)
    allowed_events = set(rain["event_id"].astype(str)) if "event_id" in rain else set()

    summary_path = out_root / "summary.csv"
    summary_rows_before = 0
    summary_rows_after = 0
    stale_summary_path = out_root / "summary_stale_excluded.csv"
    if summary_path.exists() and summary_path.stat().st_size > 0:
        summary = pd.read_csv(summary_path)
        summary_rows_before = int(len(summary))
        if "event_id" in summary:
            keep = summary["event_id"].astype(str).isin(allowed_events)
            stale = summary.loc[~keep].copy()
            current = summary.loc[keep].copy()
            summary_rows_after = int(len(current))
            if args.apply:
                stale.to_csv(stale_summary_path, index=False)
                current.to_csv(summary_path, index=False)

    traj_dir = out_root / "trajectories"
    stale_files = []
    if traj_dir.exists():
        for path in sorted(traj_dir.glob("*_detail.csv")):
            if _detail_event_id(path) not in allowed_events:
                stale_files.append(path)

    quarantine_dir = out_root / "trajectories_stale_excluded"
    moved = 0
    if args.apply and args.quarantine_stale_details and stale_files:
        ensure_dir(quarantine_dir)
        for path in stale_files:
            target = quarantine_dir / path.name
            if target.exists():
                path.unlink()
            else:
                shutil.move(str(path), str(target))
            moved += 1

    manifest_path = out_root / "stale_trajectory_manifest.csv"
    manifest = pd.DataFrame(
        {
            "detail_file": [str(p) for p in stale_files],
            "event_id": [_detail_event_id(p) for p in stale_files],
        }
    )
    if args.apply:
        manifest.to_csv(manifest_path, index=False)

    report = {
        "rainfall_event_table": str(rain_table_path),
        "allowed_event_count": int(len(allowed_events)),
        "summary_rows_before": summary_rows_before,
        "summary_rows_after": summary_rows_after,
        "summary_stale_rows": int(summary_rows_before - summary_rows_after),
        "stale_detail_files": int(len(stale_files)),
        "stale_detail_files_moved": int(moved),
        "quarantine_dir": str(quarantine_dir),
        "manifest": str(manifest_path),
        "applied": bool(args.apply),
    }
    (out_root / "trajectory_bank_clean_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
