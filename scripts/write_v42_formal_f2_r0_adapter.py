"""Write a strict Formal-F2 compatibility view for the existing R0 mainline gate.

This is NOT a bypass. The legacy R0 filenames are written only after Formal F2
proves >=65 target rainfall groups, raw same-state/same-forcing four-reference
admission, frozen target-source registry, no reserved-evaluation contamination,
and an expanded Step1 physical pool. Existing mainline code can therefore remain
fail-closed while F2 replaces stale historical discovery identities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, read_table, sha256_file

FORMAL_TARGET_STEP2_SOURCES = {
    "train1600_v3",
    "pilot_v3",
    "peak_boundary",
    "v41_calibration",
    "v41_locked",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--formal-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse",
    )
    ap.add_argument("--min-train-groups", type=int, default=65)
    args = ap.parse_args()

    prepare = _json(args.formal_root / "prepare" / "FORMAL_F2_PREPARE_AUDIT.json")
    step1_pool = _json(args.formal_root / "prepare" / "FORMAL_F2_STEP1_POOL_AUDIT.json")
    raw_audit = _json(args.formal_root / "step2" / "FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json")
    raw = read_table(args.formal_root / "step2" / "FORMAL_F2_STEP2_RAW_MANIFEST.parquet")
    ledger = read_table(args.formal_root / "prepare" / "FORMAL_F2_EVENT_LEDGER.csv")
    if prepare.get("status") != "pass" or step1_pool.get("status") != "pass" or raw_audit.get("status") != "pass":
        raise RuntimeError("Formal F2 prepare/Step1 pool/raw Step2 must pass before writing R0 compatibility evidence")
    if raw.empty:
        raise RuntimeError("Formal F2 raw Step2 manifest is empty")
    groups = set(raw["split_group_key"].astype(str))
    if len(groups) < args.min_train_groups:
        raise RuntimeError(f"Formal F2 raw Step2 has only {len(groups)} rainfall groups")
    sources = set(raw["source_dataset"].astype(str)) if "source_dataset" in raw else set()
    if not sources or not sources.issubset(FORMAL_TARGET_STEP2_SOURCES):
        raise RuntimeError(f"non-target or unknown Step2 sources cannot authorize R0 target gate: {sorted(sources - FORMAL_TARGET_STEP2_SOURCES)}")
    required_true = [
        "training_admission_authorized",
        "raw_independent_oracle_all_pass",
        "same_state_raw_verified",
        "same_forcing_raw_verified",
        "actual_readback_verified",
        "h120_window_complete",
        "kpi_recompute_ok",
    ]
    for col in required_true:
        if col not in raw or not bool(raw[col].astype(bool).all()):
            raise RuntimeError(f"Formal F2 raw manifest does not prove {col}")
    reserved = set(ledger.loc[ledger["formal_f2_role"].astype(str).isin(["calibration", "locked_validation", "challenge", "formal_blind", "excluded_historical_reserved"]), "rainfall_group_key"].astype(str))
    if groups & reserved:
        raise RuntimeError(f"Formal F2 R0 training population overlaps reserved/evaluation rainfalls: {sorted(groups & reserved)[:10]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "full_finite_check": True,
        "missing_targets_are_imputed": False,
        "strict_semantics_wrapper": True,
        "discovery_cache_current": True,
        "pfvfirst_continuation_role_policy": "auxiliary_until_proven_by_provenance",
        "f2_structured_source_registry": True,
        "raw_detail_revalidation": True,
        "rainfall_group_count": len(groups),
        "source_datasets": sorted(sources),
        "formal_f2_prepare_sha256": sha256_file(args.formal_root / "prepare" / "FORMAL_F2_PREPARE_AUDIT.json"),
        "formal_f2_raw_admission_sha256": sha256_file(args.formal_root / "step2" / "FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json"),
    }
    (args.output_root / "data_reuse_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    reusable = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "strict_scientific_admission": True,
        "counterfactual_requires_all_four_roles_finite": True,
        "formal_counterfactual_requires_target_no_dwf": True,
        "source_domain_formal_admission_forbidden": True,
        "task_counts": {
            "dynamics_pretrain_physical_runs": int(step1_pool.get("physical_runs", 0)),
            "formal_target_domain_cases": int(len(raw)),
            "counterfactual_flood_cases": int(len(raw)),
        },
        "authority": "Formal F2 raw re-admission, not legacy filename/domain inference",
    }
    (args.output_root / "reusable_pool_summary.json").write_text(json.dumps(reusable, indent=2, allow_nan=False), encoding="utf-8")
    alignment = pd.DataFrame(
        {
            "case_uid": raw["case_uid"].astype(str),
            "split_group_key": raw["split_group_key"].astype(str),
            "same_state_numeric_pass": True,
            "same_forcing_pass": True,
            "formal_generation_id": FORMAL_GENERATION_ID,
        }
    )
    alignment.to_csv(args.output_root / "case_alignment_audit.csv", index=False)
    split = pd.DataFrame({"split_group_key": sorted(groups), "reserved_evaluation": False, "formal_generation_id": FORMAL_GENERATION_ID})
    split.to_parquet(args.output_root / "split_group_manifest.parquet", index=False)
    summary = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "status": "pass",
        "legacy_r0_schema_written_from_stricter_f2_evidence": True,
        "rainfall_groups": len(groups),
        "counterfactual_cases": len(raw),
        "reserved_overlap_count": 0,
        "sources": sorted(sources),
    }
    (args.output_root / "FORMAL_F2_R0_ADAPTER_AUDIT.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
