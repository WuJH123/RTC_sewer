from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.experiments.targeted_joint_pairs import (
    action_window,
    event_pattern,
    event_return_period,
    materialize_candidate,
    sequence_diagnostics,
)
from sewerrtc.experiments.informative_effect_supplement import build_boundary_v4_specifications
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


EVENT_SPLIT = {
    "T20_D150_chicago_center": "train",
    "T20_D240_block": "train",
    "T50_D105_chicago_late": "train",
    "T50_D150_block": "train",
    "T100_D105_block": "train",
    "T100_D150_chicago_early": "train",
    "T100_D240_chicago_late": "train",
    "T20_D300_chicago_late": "validation",
    "T50_D240_chicago_center": "validation",
    "T100_D300_chicago_center": "validation",
}


def _hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _profile(delta: float, phase: str) -> list[float]:
    if phase == "peak":
        mask = [0, 1, 1, 1, 1, 0]
    else:
        mask = [1, 1, 1, 1, 1, 0]
    return [float(delta) * value for value in mask]


def _specifications(phase: str, offset: int) -> list[dict]:
    broad = ["ADD424.1", "ADD424.2", "ADD424.3", "cc006.1", "dwxh.2", "Zhongyi-2.2"]
    focused = ["ADD424.1", "ADD424.3", "RTC_IN_02", "jichangheTank.2"]
    target = focused[offset % len(focused)]
    return [
        {
            "family": "pfv_unsafe_boundary",
            "kind": "strong_counterfactual",
            "mode": "six_regulator_restrict_0p50",
            "actuators": broad,
            "signed_profiles": {actuator_id: _profile(-0.50, phase) for actuator_id in broad},
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
            "intended_evidence_role": "pfv_unsafe_rejection",
        },
        {
            "family": "pfv_unsafe_boundary",
            "kind": "strong_counterfactual",
            "mode": "six_regulator_restrict_0p80",
            "actuators": broad,
            "signed_profiles": {actuator_id: _profile(-0.80, phase) for actuator_id in broad},
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
            "intended_evidence_role": "pfv_unsafe_rejection",
        },
        {
            "family": "pfv_unsafe_boundary",
            "kind": "strong_single_or_pair",
            "mode": f"{target}_restrict_0p80",
            "actuators": [target],
            "signed_profiles": {target: _profile(-0.80, phase)},
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
            "intended_evidence_role": "pfv_boundary_identity",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--base-manifest", default="outputs/project6_36_temporal_joint_v2/joint_data_plan/targeted_informative_paired_manifest.csv")
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v2/joint_data_plan_pfv_unsafe")
    parser.add_argument("--max-candidate-cases", type=int, default=60)
    parser.add_argument("--spec-version", choices=("v3", "v4"), default="v3")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    base_manifest_path = root / args.base_manifest if not Path(args.base_manifest).is_absolute() else Path(args.base_manifest)
    base_manifest = pd.read_csv(base_manifest_path)
    action_ids = np.load(
        root / "outputs/project6_36_temporal_joint_v1/effect_dataset/same_state_raw_joint_36.npz",
        allow_pickle=True,
    )["action_ids"].astype(str).tolist()
    rain = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    canonical_hash = str(base_manifest["canonical_action_order_hash"].iloc[0])
    semantics_hash = str(base_manifest["actuator_semantics_hash"].iloc[0])
    schema = f"pfv_unsafe_{args.spec_version}"
    rows = []
    for event_offset, (event_id, split) in enumerate(EVENT_SPLIT.items()):
        event = rain.loc[event_id]
        detail_path = reference_bank / f"{event_id}__no_control_detail.csv"
        detail = pd.read_csv(detail_path)
        for phase in ("peak", "recession"):
            start_min = float(event.duration_min) * 0.55 if phase == "peak" else float(event.duration_min) + 30.0
            reference = action_window(detail, action_ids=action_ids, start_min=start_min, horizon_steps=6)
            checkpoint_id = f"{event_id}|{phase}|{start_min:.1f}"
            reference_execution_id = _hash({"event_id": event_id, "checkpoint": checkpoint_id, "branch": "reference", "schema": schema})
            specifications = (
                build_boundary_v4_specifications(phase)
                if args.spec_version == "v4"
                else _specifications(phase, event_offset)
            )
            for specification in specifications:
                candidate = materialize_candidate(reference, action_ids=action_ids, specification=specification)
                diagnostic = sequence_diagnostics(
                    candidate,
                    reference,
                    action_ids=action_ids,
                    binary_pump_ids={"ADD301.2", "ADD301.3"},
                )
                simultaneous_limit = 8 if args.spec_version == "v4" else 6
                if not diagnostic["valid"] or int(diagnostic["max_simultaneous_changes"]) > simultaneous_limit:
                    raise ValueError(f"invalid supplemental candidate {event_id}/{phase}: {diagnostic}")
                pair_id = _hash({"checkpoint": checkpoint_id, "specification": specification, "schema": schema})
                candidate_execution_id = _hash({"event_id": event_id, "checkpoint": checkpoint_id, "specification": specification, "schema": schema})
                common = {
                    "pair_id": pair_id,
                    "event_id": event_id,
                    "phase": phase,
                    "split": split,
                    "checkpoint_id": checkpoint_id,
                    "override_start_min": start_min,
                    "split_timestamp_fraction": start_min / float(event.duration_min),
                    "reference_policy": "no_control",
                    "candidate_action_sequence": json.dumps(specification, sort_keys=True),
                    "canonical_action_order_hash": canonical_hash,
                    "actuator_semantics_hash": semantics_hash,
                    "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "requires_same_state_branching": True,
                    "candidate_kind": specification["kind"],
                    "candidate_family": specification["family"],
                    "intended_evidence_role": specification["intended_evidence_role"],
                    "selection_rank": 0,
                    "selection_required_family": "pfv_unsafe_boundary",
                    "ensemble_disagreement_score": np.nan,
                    "information_score": np.nan,
                    "ensemble_mean_delta_PFV": np.nan,
                    "ensemble_mean_delta_TFV": np.nan,
                    "ensemble_mean_delta_peak": np.nan,
                    "materialized_reference_action_sequence": json.dumps(reference.astype(float).tolist()),
                    "materialized_candidate_action_sequence": json.dumps(candidate.astype(float).tolist()),
                    **{key: value for key, value in diagnostic.items() if key not in {"actual_delta_after_clipping", "changed_actuator_ids"}},
                    "changed_actuator_ids": ",".join(diagnostic["changed_actuator_ids"]),
                    "actual_delta_after_clipping": json.dumps(diagnostic["actual_delta_after_clipping"], sort_keys=True),
                    "return_period": event_return_period(event_id),
                    "rain_pattern": event_pattern(event_id),
                }
                rows.append({
                    **common,
                    "case_id": _hash({"pair": pair_id, "branch": "A"}),
                    "execution_case_id": reference_execution_id,
                    "branch": "A",
                    "executed_action_sequence": json.dumps({"mode": "default_no_control", "horizon_steps": 6}, sort_keys=True),
                    "status": "reference_reused",
                })
                rows.append({
                    **common,
                    "case_id": _hash({"pair": pair_id, "branch": "B"}),
                    "execution_case_id": candidate_execution_id,
                    "branch": "B",
                    "executed_action_sequence": json.dumps(specification, sort_keys=True),
                    "status": "preflight_validated_not_started",
                })
    supplement = pd.DataFrame(rows)
    candidates = supplement[supplement["branch"].astype(str).eq("B")]
    combined = pd.concat([base_manifest, supplement], ignore_index=True)
    train_events = set(candidates.loc[candidates["split"].eq("train"), "event_id"])
    validation_events = set(candidates.loc[candidates["split"].eq("validation"), "event_id"])
    checks = {
        "exact_candidate_budget": len(candidates) == int(args.max_candidate_cases),
        "combined_candidate_budget_240": int(combined["branch"].astype(str).eq("B").sum()) == 240,
        "no_noops": not bool(candidates["is_noop"].astype(bool).any()),
        "event_group_split_disjoint": not bool(train_events & validation_events),
        "validation_events_at_least_3": len(validation_events) >= 3,
        "same_checkpoint_reference": candidates["checkpoint_id"].notna().all(),
        "canonical_shape_H36": candidates["materialized_candidate_action_sequence"].map(lambda value: np.asarray(json.loads(value)).shape == (6, 36)).all(),
        "simultaneous_limit": int(candidates["max_simultaneous_changes"].max()) <= (8 if args.spec_version == "v4" else 6),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    supplement_path = out / "pfv_unsafe_supplement_manifest.csv"
    combined_path = out / "targeted_informative_combined_240_manifest.csv"
    supplement.to_csv(supplement_path, index=False)
    combined.to_csv(combined_path, index=False)
    report = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "supplement_candidate_cases": int(len(candidates)),
        "combined_candidate_cases": int(combined["branch"].astype(str).eq("B").sum()),
        "reference_checkpoints_reused": int(candidates["checkpoint_id"].nunique()),
        "train_events": sorted(train_events),
        "validation_events": sorted(validation_events),
        "candidate_modes": candidates["candidate_action_sequence"].map(lambda value: json.loads(value)["mode"]).value_counts().to_dict(),
        "spec_version": args.spec_version,
        "supplement_manifest": str(supplement_path),
        "combined_manifest": str(combined_path),
        "formal_noninferiority_margin_unchanged": {"absolute_m3": 100.0, "relative": 0.005},
        "swmm_started": False,
    }
    (out / "pfv_unsafe_supplement_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("PFV unsafe supplement preflight failed")


if __name__ == "__main__":
    main()
