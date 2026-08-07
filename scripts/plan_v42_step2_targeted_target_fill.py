"""Freeze a small development-only target-fill plan from existing candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_v42_step2_formal_f2 import _rank
from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256


def arr(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--experience-bank", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-candidates", type=int, default=128)
    args = ap.parse_args()

    columns = [
        "state_key", "event_id", "rainfall_sha256", "split_group_key", "checkpoint_min",
        "action_candidate_readback", "action_hold_previous_readback",
        "source_detail_path_no_control", "source_detail_path_hold_previous",
        "candidate_action_sha256", "pfv_delta", "tfv_delta",
        "trajectory_storage_volume_candidate_available",
        "trajectory_facility_flow_candidate_available",
        "control_core_target_coverage_complete", "candidate_expansion_family",
        "candidate_expansion_round",
    ]
    manifest = pd.read_parquet(args.manifest, columns=columns)
    groups = _rank(sorted(manifest.split_group_key.astype(str).unique()), 42)
    train_groups = set(groups[16:])
    source = manifest[
        manifest.split_group_key.astype(str).isin(train_groups)
        & (~manifest.trajectory_storage_volume_candidate_available.fillna(False).astype(bool)
           | ~manifest.trajectory_facility_flow_candidate_available.fillna(False).astype(bool))
    ].copy()
    if source.empty:
        raise RuntimeError("no missing CONTROL_CORE targets remain in development groups")

    bank = pd.read_parquet(
        args.experience_bank,
        columns=["state_key", "canonical_candidate_action_sha256", "pfv_budget_metric_m3", "tfv_candidate_m3", "tfv_internal_m3"],
    )
    source["canonical_action_sha256"] = source.action_candidate_readback.map(lambda value: action_sha256(arr(value)))
    bank = bank.drop_duplicates(["state_key", "canonical_candidate_action_sha256"], keep="last")
    source = source.merge(
        bank,
        left_on=["state_key", "canonical_action_sha256"],
        right_on=["state_key", "canonical_candidate_action_sha256"],
        how="left",
    )
    source["pfv_safe_truth"] = source.pfv_budget_metric_m3 <= 100.0
    source["tfv_improving_truth"] = source.tfv_candidate_m3 < source.tfv_internal_m3
    source["action_magnitude"] = source.action_candidate_readback.map(lambda value: float(np.abs(arr(value)[0] - arr(value)[-1]).sum()))
    source["information_score"] = (
        source.pfv_safe_truth.astype(float) * 4.0
        + source.tfv_improving_truth.astype(float) * 3.0
        + source.pfv_budget_metric_m3.fillna(1.0e9).abs().rank(pct=True) * 0.5
        + source.action_magnitude.rank(pct=True) * 0.5
    )
    # Two high-information actions per state, then fill by score.  This keeps
    # the SWMM budget state-diverse without pretending to estimate uncertainty.
    ranked = source.sort_values(["state_key", "information_score", "canonical_action_sha256"], ascending=[True, False, True], kind="stable")
    chosen = ranked.groupby("state_key", sort=True, group_keys=False).head(2).copy()
    if len(chosen) < int(args.max_candidates):
        remaining = ranked.loc[~ranked.index.isin(chosen.index)].sort_values(["information_score", "canonical_action_sha256"], ascending=[False, True], kind="stable")
        chosen = pd.concat([chosen, remaining.head(int(args.max_candidates) - len(chosen))], ignore_index=True)
    chosen = chosen.head(int(args.max_candidates)).copy()
    if chosen["canonical_action_sha256"].duplicated().any():
        raise RuntimeError("target-fill plan contains duplicate canonical actions")

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    plan_columns = [
        "state_key", "event_id", "rainfall_sha256", "split_group_key", "checkpoint_min",
        "canonical_action_sha256", "action_candidate_readback", "action_hold_previous_readback",
        "source_detail_path_no_control", "source_detail_path_hold_previous", "candidate_action_sha256",
        "pfv_budget_metric_m3", "tfv_candidate_m3", "tfv_internal_m3", "pfv_safe_truth",
        "tfv_improving_truth", "information_score", "candidate_expansion_family", "candidate_expansion_round",
    ]
    plan = chosen[plan_columns].copy()
    plan_path = out / "TARGETED_STEP2_TARGET_FILL_PLAN.csv"
    plan.to_csv(plan_path, index=False)
    lock = {
        "lock_id": "V42_STEP2_TARGETED_TARGET_FILL_PLAN_LOCK_V1",
        "development_only": True,
        "formal_mainline_authorized": False,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "experience_bank": str(args.experience_bank.resolve()),
        "experience_bank_sha256": hashlib.sha256(args.experience_bank.read_bytes()).hexdigest(),
        "train_group_count": len(train_groups),
        "planned_rows": int(len(plan)),
        "planned_states": int(plan.state_key.nunique()),
        "max_candidates": int(args.max_candidates),
        "new_swmm_budget_hard_max": int(args.max_candidates),
        "uncertainty_available": False,
        "selection_basis": "truth labels, PFV/TFV Pareto relevance, action novelty, state diversity; no future SWMM truth used for online control",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    }
    (out / "TARGETED_STEP2_TARGET_FILL_PLAN_LOCK.json").write_text(json.dumps(lock, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"planned_rows": len(plan), "states": int(plan.state_key.nunique()), "train_groups": len(train_groups), "plan": str(plan_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
