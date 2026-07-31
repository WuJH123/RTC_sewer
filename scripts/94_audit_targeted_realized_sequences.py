from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--dataset", default="outputs/project6_36_temporal_joint_v1/effect_dataset/same_state_raw_joint_36.npz")
    parser.add_argument("--case-dir", default="outputs/project6_36_temporal_joint_v2/paired_cases")
    parser.add_argument("--manifest", default="outputs/project6_36_temporal_joint_v2/joint_data_plan/targeted_informative_paired_manifest.csv")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v2/joint_data_plan_alignment_fix")
    parser.add_argument("--max-correction-cases", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    dataset_path = root / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    case_dir = root / args.case_dir if not Path(args.case_dir).is_absolute() else Path(args.case_dir)
    manifest_path = root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    action_ids = np.load(dataset_path, allow_pickle=True)["action_ids"].astype(str).tolist()
    results = pd.read_csv(case_dir / "paired_candidate_results.csv")
    records = []
    for row in results.itertuples(index=False):
        detail = pd.read_csv(row.candidate_detail)
        elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)
        start = int(np.searchsorted(elapsed, float(row.override_start_min), side="left"))
        window = detail.iloc[start : start + 6]
        realized = window[[f"a:{actuator_id}" for actuator_id in action_ids]].to_numpy(np.float32)
        planned = np.asarray(json.loads(str(row.materialized_candidate_action_sequence)), dtype=np.float32)
        difference = realized - planned
        changed = np.abs(difference) > 1.0e-6
        positions = np.argwhere(changed)
        mismatch_actuator_ids = [action_ids[index] for index in np.flatnonzero(changed.any(axis=0))]
        records.append({
            "case_id": str(row.case_id),
            "pair_id": str(row.pair_id),
            "event_id": str(row.event_id),
            "phase": str(row.phase),
            "candidate_family": str(row.candidate_family),
            "override_start_min": float(row.override_start_min),
            "first_realized_elapsed_min": float(window["elapsed_min"].iloc[0]),
            "sequence_aligned": not bool(len(positions)),
            "mismatch_cell_count": int(changed.sum()),
            "mismatch_actuator_count": int(changed.any(axis=0).sum()),
            "mismatch_actuator_ids": ",".join(mismatch_actuator_ids),
            "max_abs_setting_error": float(np.abs(difference).max(initial=0.0)),
            "planned_sequence_sha256": _hash_array(planned),
            "realized_sequence_sha256": _hash_array(realized),
            "mismatch_positions": json.dumps(positions.astype(int).tolist()),
        })
    audit = pd.DataFrame(records)
    audit.to_csv(out / "realized_action_sequence_alignment_audit.csv", index=False)
    mismatch_pairs = set(audit.loc[~audit["sequence_aligned"], "pair_id"].astype(str))
    manifest = pd.read_csv(manifest_path)
    correction = manifest[manifest["pair_id"].astype(str).isin(mismatch_pairs)].copy()
    correction.loc[correction["branch"].astype(str).eq("B"), "status"] = "alignment_fix_pending"
    correction_path = out / "targeted_alignment_correction_manifest.csv"
    correction.to_csv(correction_path, index=False)
    candidate_count = int(correction["branch"].astype(str).eq("B").sum())
    report = {
        "passed": candidate_count <= int(args.max_correction_cases),
        "audited_candidate_cases": int(len(audit)),
        "aligned_candidate_cases": int(audit["sequence_aligned"].sum()),
        "mismatched_candidate_cases": int((~audit["sequence_aligned"]).sum()),
        "mismatched_pair_count": len(mismatch_pairs),
        "correction_candidate_cases": candidate_count,
        "correction_physical_cases": candidate_count + int(correction["checkpoint_id"].nunique()),
        "correction_budget": int(args.max_correction_cases),
        "correction_manifest": str(correction_path),
        "original_results_untouched": True,
        "failure_reason": "off_grid_rounding_skipped_temporal_tokens",
    }
    (out / "alignment_correction_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("alignment correction exceeds approved candidate budget")


if __name__ == "__main__":
    main()
