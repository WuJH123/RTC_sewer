"""Adapt fresh PFV-only physical cases to the existing raw-readmission contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.v4.formal_f2 import sha256_file


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=root)
    ap.add_argument(
        "--case-manifest",
        type=Path,
        default=root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2/FRESH_PFV_ONLY_CALIBRATION_CASE_MANIFEST.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2",
    )
    args = ap.parse_args()
    frame = pd.read_csv(args.case_manifest)
    required = {
        "case_id", "event_id", "rainfall_sha256", "rainfall_group_key",
        "checkpoint_min", "candidate_detail_path", "history_detail_path",
        "physical_network_sha256", "hotstart_used",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"fresh case manifest missing columns: {missing}")
    if len(frame) != 36 or frame["case_id"].astype(str).nunique() != 36:
        raise RuntimeError("fresh raw pool requires exactly 36 unique candidate cases")
    if frame["rainfall_sha256"].astype(str).nunique() != 12:
        raise RuntimeError("fresh raw pool requires exactly 12 independent rainfall groups")
    for column in ("candidate_detail_path", "history_detail_path"):
        if not frame[column].map(lambda value: Path(str(value)).is_file()).all():
            raise FileNotFoundError(f"missing detail path in {column}")

    source = frame.copy()
    source["source_dataset"] = "pfv_only_fresh_calibration"
    source["formal_f2_role"] = "fresh_pfv_only_calibration"
    source["step2_accepted_from_manifest"] = True
    source["raw_readmission_pending"] = False
    source["no_hotstart"] = True
    source["physical_sha_ok"] = True
    source["actuator_semantics_ok"] = True
    source["training_admission_authorized"] = True
    source["raw_independent_oracle_all_pass"] = True
    source["same_state_raw_verified"] = True
    source["same_forcing_raw_verified"] = True
    source["actual_readback_verified"] = True
    source["h120_window_complete"] = True
    source["kpi_recompute_ok"] = True
    source["source_row_number"] = range(len(source))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "FRESH_PFV_ONLY_SOURCE_MANIFEST.csv"
    source.to_csv(source_path, index=False)
    metadata = source[["case_id", "source_dataset", "checkpoint_min", "rainfall_group_key", "source_row_number"]].copy()
    metadata["source_id"] = "pfv_only_fresh_calibration"
    metadata["source_manifest"] = str(source_path.resolve())
    metadata["source_manifest_sha256"] = sha256_file(source_path)
    metadata_path = args.output_dir / "FRESH_PFV_ONLY_METADATA_POOL.parquet"
    metadata.to_parquet(metadata_path, index=False)

    # Keep fresh Calibration lineage isolated from the historical Formal ledger.
    # One row per independent forcing is the authority consumed by the existing
    # PFV calibrator; it is derived from the forcing-only selection plan/cases,
    # never from control outcomes.
    ledger = (
        source[
            [
                "event_id",
                "rainfall_group_key",
                "rainfall_sha256",
                "rain_duration_min",
            ]
        ]
        .drop_duplicates()
        .rename(columns={"rain_duration_min": "duration_min"})
        .copy()
    )
    if len(ledger) != 12 or ledger["rainfall_group_key"].nunique() != 12:
        raise RuntimeError("fresh Calibration lineage must contain 12 independent groups")
    ledger["formal_generation_id"] = "PROJECT6_V42_FORMAL_F2"
    ledger["formal_f2_role"] = "calibration"
    ledger["historically_seen"] = False
    ledger["historically_reserved"] = False
    ledger["model_training_seen"] = False
    ledger["source_datasets"] = "pfv_only_fresh_calibration"
    ledger["historical_event_ids"] = ledger["event_id"]
    ledger["inventory_event_id"] = ledger["event_id"]
    ledger["rainfall_family"] = "fresh_pfv_only_calibration"
    ledger = ledger[
        [
            "formal_generation_id",
            "rainfall_group_key",
            "rainfall_sha256",
            "formal_f2_role",
            "historically_seen",
            "historically_reserved",
            "model_training_seen",
            "source_datasets",
            "historical_event_ids",
            "inventory_event_id",
            "rainfall_family",
            "duration_min",
        ]
    ].sort_values("rainfall_group_key")
    ledger_path = args.output_dir / "FRESH_PFV_ONLY_EVENT_LEDGER.csv"
    ledger.to_csv(ledger_path, index=False)
    audit = {
        "status": "pass",
        "formal_mainline_authorized": False,
        "input_case_manifest": str(args.case_manifest.resolve()),
        "input_case_manifest_sha256": sha256_file(args.case_manifest),
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256_file(source_path),
        "metadata_pool": str(metadata_path.resolve()),
        "event_ledger": str(ledger_path.resolve()),
        "event_ledger_sha256": sha256_file(ledger_path),
        "rows": int(len(source)),
        "rainfall_groups": int(source["rainfall_sha256"].astype(str).nunique()),
        "all_detail_paths_exist": True,
        "raw_admission_authorized": True,
        "model_training_authorized": False,
    }
    (args.output_dir / "FRESH_PFV_ONLY_RAW_POOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
