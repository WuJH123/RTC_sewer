"""Adapt NEW Formal F2 Calibration SWMM cases into causal Step1/Step2 manifests.

This script does not create or simulate events. It is the strict bridge used
after the local authoritative SWMM generator has executed the frozen F2
Calibration plan. The case manifest must identify rainfall SHA, checkpoint,
case_id, and a same-event full causal history detail. The four-reference outcome
is independently re-admitted by materialize_v42_formal_step2_f2.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, canonical_rain_group, checkpoint_of, read_table, sha256_file, text
from sewerrtc.v4.v42_step1_dataset import _build_usecols, _detail_extract_window, load_graph_assets


def _history_path(row: dict) -> str:
    for key in ("history_detail_path", "prefix_detail_path", "source_detail_path_history", "whole_event_detail_path"):
        value = text(row.get(key, ""))
        if value:
            return value
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--case-manifest", type=Path, required=True)
    ap.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_EVENT_LEDGER.csv",
    )
    ap.add_argument(
        "--step1-model-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step1/seed_42",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration",
    )
    ap.add_argument("--min-groups", type=int, default=8)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    args = ap.parse_args()

    cases = read_table(args.case_manifest)
    ledger = read_table(args.ledger)
    if cases.empty:
        raise ValueError("F2 Calibration case manifest is empty")
    allowed = set(ledger.loc[ledger["formal_f2_role"].astype(str).eq("calibration"), "rainfall_group_key"].astype(str))
    graph = load_graph_assets(args.project_root)
    required = _build_usecols(graph.node_ids, graph.facility_ids)
    step1_rows: list[dict] = []
    metadata_rows: list[dict] = []
    failures: list[dict] = []
    seen_windows: set[tuple[str, float, str]] = set()

    for pos, row in enumerate(cases.to_dict("records")):
        rain = canonical_rain_group(row)
        event = text(row.get("event_id", row.get("rainfall_event_id", "")))
        case_id = text(row.get("case_id", row.get("candidate_id", "")))
        checkpoint = checkpoint_of(row)
        history_raw = _history_path(row)
        try:
            if rain not in allowed:
                raise RuntimeError(f"rainfall not frozen as new F2 Calibration: {rain}")
            if not case_id or not np.isfinite(checkpoint):
                raise ValueError("calibration row missing case_id/checkpoint")
            history = Path(history_raw)
            if not history.exists():
                raise FileNotFoundError(f"history detail missing: {history}")
            header = pd.read_csv(history, nrows=0)
            missing = [c for c in required if c not in set(map(str, header.columns))]
            if missing:
                raise KeyError(f"history detail lacks Step1 columns: {missing[:8]}")
            detail = pd.read_csv(history, usecols=required, low_memory=False).loc[:, required]
            item = _detail_extract_window(detail, float(checkpoint), graph.node_ids, graph.facility_ids)
            if item is None:
                raise ValueError("history detail does not provide exact 13x5min Step1 window at checkpoint")
            physical_sha = sha256_file(history)
            wkey = (rain, float(checkpoint), physical_sha)
            if wkey not in seen_windows:
                seen_windows.add(wkey)
                step1_rows.append(
                    {
                        "formal_generation_id": FORMAL_GENERATION_ID,
                        "formal_split": "calibration",
                        "step1_domain_role": "target_formal_calibration",
                        "split_group_key": rain,
                        "rainfall_sha256": rain,
                        "event_id": event,
                        "detail_path": str(history.resolve()),
                        "anchor_min": float(checkpoint),
                        "physical_identity_sha256": physical_sha,
                        "source_dataset": "formal_f2_new_calibration",
                    }
                )
            metadata_rows.append(
                {
                    "formal_generation_id": FORMAL_GENERATION_ID,
                    "source_id": "formal_f2_new_calibration",
                    "source_manifest": str(args.case_manifest.resolve()),
                    "source_manifest_sha256": sha256_file(args.case_manifest),
                    "source_row_number": int(pos),
                    "case_id": case_id,
                    "event_id": event,
                    "rainfall_group_key": rain,
                    "checkpoint_min": float(checkpoint),
                    "state_key": "",
                    "action_key": "",
                    "formal_step1_allowed": True,
                    "formal_step2_allowed": True,
                    "step2_accepted_from_manifest": False,
                    "raw_readmission_required": True,
                    "raw_readmission_pending": True,
                    "formal_f2_role": "calibration",
                    "training_admission_authorized": False,
                }
            )
        except Exception as exc:
            failures.append({"source_row_number": pos, "case_id": case_id, "rainfall_group": rain, "error": f"{type(exc).__name__}: {exc}"})

    step1 = pd.DataFrame(step1_rows)
    metadata = pd.DataFrame(metadata_rows)
    groups = set(metadata.get("rainfall_group_key", pd.Series(dtype=str)).astype(str))
    if len(groups) < args.min_groups:
        raise RuntimeError(f"only {len(groups)} valid new F2 Calibration groups; require {args.min_groups}; examples={failures[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step1_path = args.output_dir / "FORMAL_F2_CALIBRATION_STEP1_WINDOW_MANIFEST.parquet"
    metadata_path = args.output_dir / "FORMAL_F2_CALIBRATION_STEP2_METADATA_POOL.parquet"
    step1.to_parquet(step1_path, index=False)
    metadata.to_parquet(metadata_path, index=False)

    py = str(Path(sys.executable))
    raw_path = args.output_dir / "FORMAL_F2_CALIBRATION_RAW_MANIFEST.parquet"
    gat_path = args.output_dir / "FORMAL_F2_CALIBRATION_GAT_MANIFEST.parquet"
    subprocess.run(
        [py, "-u", str(args.project_root / "scripts/materialize_v42_formal_step2_f2.py"),
         "--project-root", str(args.project_root), "--metadata-pool", str(metadata_path),
         "--output-manifest", str(raw_path), "--min-rainfall-groups", str(args.min_groups)],
        cwd=str(args.project_root), check=True,
    )
    subprocess.run(
        [py, "-u", str(args.project_root / "scripts/materialize_v42_formal_gat_history_f2.py"),
         "--project-root", str(args.project_root), "--input-manifest", str(raw_path),
         "--step1-window-manifest", str(step1_path), "--step1-model-dir", str(args.step1_model_dir),
         "--output-manifest", str(gat_path), "--min-rainfall-groups", str(args.min_groups),
         "--sensor-layout-seed", str(args.sensor_layout_seed)],
        cwd=str(args.project_root), check=True,
    )
    audit = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_new_calibration_data_bridge",
        "status": "pass",
        "case_manifest": str(args.case_manifest),
        "case_manifest_sha256": sha256_file(args.case_manifest),
        "valid_rainfall_groups": len(groups),
        "step1_windows": len(step1),
        "step2_metadata_rows": len(metadata),
        "failed_rows": len(failures),
        "failure_examples": failures[:100],
        "step1_manifest": str(step1_path),
        "step2_raw_manifest": str(raw_path),
        "step2_gat_manifest": str(gat_path),
        "uses_new_f2_calibration_only": True,
    }
    (args.output_dir / "FORMAL_F2_CALIBRATION_DATA_BRIDGE_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
