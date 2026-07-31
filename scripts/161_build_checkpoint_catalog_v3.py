#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import OUT_ROOT, read_csv, sha256_file, write_csv, write_json, utc_now


def _exists(path: str) -> bool:
    return bool(path) and Path(path).exists() and Path(path).is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build checkpoint catalog from real baseline hot-start and controller-memory artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory-root", default=str(OUT_ROOT / "baseline_trajectories"))
    parser.add_argument("--out-dir", default=str(OUT_ROOT / "checkpoint_catalog"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    root = Path(args.trajectory_root)
    audit = root / "baseline_checkpoint_audit.csv"
    if not audit.exists():
        report = {"status": "blocked", "failure_reason": "baseline_checkpoint_audit_missing", "required": str(audit)}
        write_json(out_dir / "checkpoint_catalog_report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3
    rows = []
    hash_rows = []
    source_rows = read_csv(audit)
    for src in source_rows:
        hotstart = src.get("hotstart_path", "")
        memory = src.get("controller_memory_path", "")
        hotstart_exists = _exists(hotstart)
        memory_exists = _exists(memory)
        eligible = (
            hotstart_exists
            and memory_exists
            and str(src.get("history_60min_available", "")).lower() == "true"
            and str(src.get("future_120min_available", "")).lower() == "true"
        )
        row = {
            "checkpoint_id": src.get("checkpoint_id", ""),
            "trajectory_id": src.get("trajectory_id", ""),
            "event_id": src.get("event_id", ""),
            "policy_id": src.get("policy_id", ""),
            "storm_family_id": src.get("storm_family_id", ""),
            "event_time": src.get("checkpoint_elapsed_min", ""),
            "phase": src.get("phase", "unknown"),
            "state_source": "project6_retrofit_baseline_hotstart",
            "state_clone_source": hotstart,
            "hotstart_path": hotstart,
            "hotstart_sha256": sha256_file(Path(hotstart)) if hotstart_exists else "",
            "controller_memory_path": memory,
            "controller_memory_hash": sha256_file(Path(memory)) if memory_exists else "",
            "node_state_hash": src.get("hotstart_sha256", ""),
            "link_state_hash": src.get("hotstart_sha256", ""),
            "storage_state_hash": src.get("hotstart_sha256", ""),
            "state_clone_hash": sha256_file(Path(hotstart)) if hotstart_exists else "",
            "rainfall_state_hash": src.get("rainfall_state_hash", ""),
            "network_sha256": src.get("network_sha256", ""),
            "history_60min_available": src.get("history_60min_available", ""),
            "future_120min_available": src.get("future_120min_available", ""),
            "split": src.get("split", "action_effect_fit"),
            "eligible_for_effect_training": "false",
            "eligible_for_state_clone": str(eligible).lower(),
            "exclusion_reason": "" if eligible else "missing_hotstart_or_controller_memory_or_temporal_support",
        }
        rows.append(row)
        hash_rows.append(
            {
                "checkpoint_id": row["checkpoint_id"],
                "hotstart_exists": str(hotstart_exists).lower(),
                "controller_memory_exists": str(memory_exists).lower(),
                "state_clone_hash": row["state_clone_hash"],
                "controller_memory_hash": row["controller_memory_hash"],
                "eligible_for_state_clone": row["eligible_for_state_clone"],
            }
        )
    eligible_count = sum(1 for row in rows if row["eligible_for_state_clone"] == "true")
    status = "completed" if rows and eligible_count > 0 else "blocked"
    files = [
        write_csv(out_dir / "checkpoint_catalog.csv", rows),
        write_csv(out_dir / "checkpoint_state_hash_audit.csv", hash_rows),
        write_csv(out_dir / "checkpoint_near_duplicate_audit.csv", []),
        write_csv(out_dir / "checkpoint_split_audit.csv", []),
        write_json(
            out_dir / "checkpoint_catalog_report.json",
            {
                "status": status,
                "created_at": utc_now(),
                "checkpoint_count": len(rows),
                "runtime_clone_eligible_checkpoint_count": eligible_count,
                "source_audit": str(audit),
                "source_audit_sha256": sha256_file(audit),
            },
        ),
    ]
    print(json.dumps({"status": status, "checkpoint_count": len(rows), "runtime_clone_eligible_checkpoint_count": eligible_count, "outputs": [str(p) for p in files]}, indent=2, ensure_ascii=False))
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())

