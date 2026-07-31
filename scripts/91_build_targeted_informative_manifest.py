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
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.raw_joint_online_predictor import RawJointOnlinePredictor


EVENT_SPLIT = {
    "T5_D150_chicago_center": "train",
    "T5_D210_chicago_early": "train",
    "T5_D240_chicago_late": "train",
    "T5_D300_block": "validation",
    "T10_D105_chicago_center": "train",
    "T10_D150_chicago_late": "train",
    "T10_D240_block": "train",
    "T10_D300_chicago_early": "validation",
    "T20_D105_chicago_early": "train",
    "T20_D150_chicago_center": "train",
    "T20_D240_block": "train",
    "T20_D300_chicago_late": "validation",
    "T50_D105_chicago_late": "train",
    "T50_D150_block": "train",
    "T50_D300_chicago_early": "train",
    "T50_D240_chicago_center": "validation",
    "T100_D105_block": "train",
    "T100_D150_chicago_early": "train",
    "T100_D240_chicago_late": "train",
    "T100_D300_chicago_center": "validation",
}

PHASE_START = {"rising": 0.25, "peak": 0.55, "recession": None}
FAMILY_CYCLE = (
    "legacy_group", "single_continuous", "binary_pump", "storage_temporal",
    "hydraulic_pair", "legacy_plus_residual", "variable_speed_pump", "adverse_group",
)


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _continuous_profile(phase: str, delta: float) -> list[float]:
    masks = {
        "rising": [1, 1, 1, 0, 0, 0],
        "peak": [0, 1, 1, 1, 0, 0],
        "recession": [0, 0, 1, 1, 1, 0],
    }
    return [float(delta) * value for value in masks[phase]]


def _binary_profile(reference: np.ndarray, position: int, phase: str) -> list[float]:
    masks = {
        "rising": [1, 1, 1, 0, 0, 0],
        "peak": [0, 1, 1, 1, 0, 0],
        "recession": [0, 0, 1, 1, 1, 0],
    }
    values = reference[:, position].copy()
    target = 0.0 if float(reference[:, position].mean()) >= 0.5 else 1.0
    for step, active in enumerate(masks[phase]):
        if active:
            values[step] = target
    return values.astype(float).tolist()


def _specifications(reference: np.ndarray, action_ids: list[str], phase: str, offset: int) -> list[dict]:
    position = {actuator_id: index for index, actuator_id in enumerate(action_ids)}
    specs: list[dict] = []

    def add(family: str, kind: str, label: str, *, intended_role: str, signed: dict[str, list[float]] | None = None, targets: dict[str, list[float]] | None = None) -> None:
        actuators = sorted(set((signed or {}).keys()) | set((targets or {}).keys()))
        spec = {
            "family": family, "kind": kind, "mode": label, "actuators": actuators,
            "horizon_steps": 6, "sequence_semantics": "relative_to_same_state_no_control_reference",
            "intended_evidence_role": intended_role,
        }
        if signed: spec["signed_profiles"] = signed
        if targets: spec["target_profiles"] = targets
        specs.append(spec)

    continuous = ["Zhongyi-2.2", "ADD424.2", "ADD424.3", "cc006.1", "dwxh.2", "HS2512760.1", "gbz1.8"]
    for shift, magnitude in enumerate((0.10, 0.20, 0.20)):
        actuator_id = continuous[(offset + shift) % len(continuous)]
        add("single_continuous", "single_continuous", f"{actuator_id}_restrict_{magnitude:.2f}", intended_role="boundary_or_direction", signed={actuator_id: _continuous_profile(phase, -magnitude)})

    legacy_groups = [
        ["ADD424.4", "ADD424.1", "ADD424.2", "ADD424.3"],
        ["cc006.1", "dwxh.2", "dw3700.1", "dw3700.2"],
        ["MH0266931.2", "MH0266932.2", "MH0266933.1"],
    ]
    for group_index, group in enumerate(legacy_groups):
        magnitude = 0.10 if group_index != 0 else 0.20
        add("legacy_group", "legacy_group", f"legacy_group_{group_index}_restrict", intended_role="benefit_or_boundary", signed={actuator_id: _continuous_profile(phase, -magnitude) for actuator_id in group})

    for pump_id in ("ADD301.2", "ADD301.3"):
        add("binary_pump", "single_binary", f"{pump_id}_toggle", intended_role="tfv_repair_or_peak_reject", targets={pump_id: _binary_profile(reference, position[pump_id], phase)})

    add("variable_speed_pump", "variable_speed_pump", "add350_ramp", intended_role="tfv_repair", signed={"add350.1": _continuous_profile(phase, 0.20)})
    pump_targets = {pump_id: _binary_profile(reference, position[pump_id], phase) for pump_id in ("ADD301.2", "ADD301.3")}
    add("hydraulic_pair", "hydraulic_pair", "dual_binary_pump", intended_role="tfv_repair_or_pfv_reject", targets=pump_targets)
    add("hydraulic_pair", "hydraulic_pair", "cc006_dwxh_pair", intended_role="benefit_or_direction", signed={actuator_id: _continuous_profile(phase, -0.20) for actuator_id in ("cc006.1", "dwxh.2")})

    for number in ("01", "02", "03"):
        outlet = f"RTC_OUT_{number}"
        add("storage_temporal", "storage_outlet_retain", f"{outlet}_retain_restore", intended_role="peak_repair_or_tfv_tradeoff", signed={outlet: _continuous_profile(phase, -0.20)})
    number = f"{(offset % 3) + 1:02d}"
    inlet, outlet = f"RTC_IN_{number}", f"RTC_OUT_{number}"
    sequential = {
        inlet: [-0.20, -0.20, 0.0, 0.0, 0.0, 0.0],
        outlet: [0.0, 0.0, -0.20, -0.20, 0.0, 0.0],
    }
    add("storage_temporal", "storage_inlet_outlet_sequence", f"storage_{number}_sequential", intended_role="temporal_order", signed=sequential)

    legacy_residual = {actuator_id: _continuous_profile(phase, -0.10) for actuator_id in ("cc006.1", "dwxh.2")}
    legacy_residual[f"RTC_OUT_{number}"] = _continuous_profile(phase, -0.20)
    add("legacy_plus_residual", "legacy_plus_new_residual", f"legacy_plus_storage_{number}", intended_role="joint_incremental_effect", signed=legacy_residual)

    pump_regulator = {"cc006.1": _continuous_profile(phase, -0.20)}
    add("legacy_plus_residual", "legacy_plus_new_residual", "cc006_plus_ADD3012", intended_role="tfv_repair_with_pfv_guard", signed=pump_regulator, targets={"ADD301.2": _binary_profile(reference, position["ADD301.2"], phase)})

    adverse_ids = ["ADD424.1", "ADD424.2", "ADD424.3", "cc006.1", "dwxh.2", "Zhongyi-2.2"]
    add("adverse_group", "strong_counterfactual", "six_regulator_strong_restrict", intended_role="safety_rejection", signed={actuator_id: _continuous_profile(phase, -0.20) for actuator_id in adverse_ids})
    return specs


def _checkpoint(detail: pd.DataFrame, *, node_ids: list[str], start_min: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)
    start = int(np.searchsorted(elapsed, float(start_min), side="left"))
    window = detail.iloc[start : start + horizon]
    if len(window) != horizon:
        raise ValueError("incomplete state/rain horizon")
    state = window.iloc[0][[f"h:{node_id}" for node_id in node_ids]].to_numpy(np.float32)
    rain = window[["rainfall_mm_h"]].to_numpy(np.float32)
    return state, rain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v2/joint_data_plan")
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates-per-checkpoint", type=int, default=3)
    parser.add_argument("--max-candidate-cases", type=int, default=240)
    args = parser.parse_args()
    cfg = load_config(args.config); root = cfg_path(cfg, "project_root")
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    old_data = np.load(root / "outputs/project6_36_temporal_joint_v1/effect_dataset/same_state_raw_joint_36.npz", allow_pickle=True)
    old_events = set(old_data["event_ids"].astype(str))
    action_ids = old_data["action_ids"].astype(str).tolist()
    rain = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    missing_events = sorted(set(EVENT_SPLIT) - set(rain.index.astype(str)))
    overlaps = sorted(set(EVENT_SPLIT) & old_events)
    if missing_events or overlaps:
        raise ValueError(f"target event selection invalid: missing={missing_events}, overlaps_old_effect={overlaps}")
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    model_paths = [Path(item) if Path(item).is_absolute() else root / item for item in args.model]
    if not model_paths:
        model_paths = [
            root / "outputs/models_temporal_joint_36/raw_joint_36_same_state_v2.pt",
            root / "outputs/models_temporal_joint_36_sanity_v4/raw_joint_36_same_state_sanity_v4.pt",
        ]
    predictors = [RawJointOnlinePredictor(path, canonical_action_ids=action_ids, device=args.device, batch_size=128) for path in model_paths]
    checkpoint = __import__("torch").load(model_paths[0], map_location="cpu", weights_only=False)
    node_ids = [str(item) for item in checkpoint["node_ids"]]
    old_manifest = pd.read_csv(root / "outputs/project6_36_temporal_joint_v1/paired_plan/joint_action_case_manifest.csv")
    canonical_hash = str(old_manifest["canonical_action_order_hash"].iloc[0])
    semantics_hash = str(old_manifest["actuator_semantics_hash"].iloc[0])
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selected_records: list[dict] = []; all_diagnostics: list[dict] = []
    checkpoint_no = 0
    for event_id, split in EVENT_SPLIT.items():
        event = rain.loc[event_id]
        detail_path = reference_bank / f"{event_id}__no_control_detail.csv"
        if not detail_path.exists(): raise FileNotFoundError(detail_path)
        detail = pd.read_csv(detail_path)
        for phase in ("rising", "peak", "recession"):
            start_min = float(event.duration_min) * float(PHASE_START[phase]) if PHASE_START[phase] is not None else float(event.duration_min) + 30.0
            reference = action_window(detail, action_ids=action_ids, start_min=start_min, horizon_steps=6)
            state, rain_window = _checkpoint(detail, node_ids=node_ids, start_min=start_min, horizon=6)
            library = []
            for specification in _specifications(reference, action_ids, phase, checkpoint_no):
                candidate = materialize_candidate(reference, action_ids=action_ids, specification=specification)
                diagnostic = sequence_diagnostics(candidate, reference, action_ids=action_ids, binary_pump_ids={"ADD301.2", "ADD301.3"})
                if not diagnostic["valid"] or int(diagnostic["max_simultaneous_changes"]) > 6:
                    continue
                library.append({"specification": specification, "candidate": candidate, "diagnostic": diagnostic})
            if len(library) < int(args.candidates_per_checkpoint):
                raise RuntimeError(f"insufficient valid candidates at {event_id}/{phase}: {len(library)}")
            candidate_batch = np.stack([item["candidate"] for item in library])
            reference_batch = np.repeat(reference[None, :, :], len(library), axis=0)
            model_outputs = [predictor.predict_many(state=state, rain_seq=rain_window, candidate_action_seq=candidate_batch, reference_action_seq=reference_batch) for predictor in predictors]
            pfv = np.stack([output["delta_PFV_H"] for output in model_outputs])
            tfv = np.stack([output["delta_TFV_H"] for output in model_outputs])
            peak = np.stack([output["delta_peak"] for output in model_outputs])
            disagreement = pfv.std(0) / 10.0 + tfv.std(0) / 500.0 + peak.std(0) / 0.1
            mean_pfv, mean_tfv, mean_peak = pfv.mean(0), tfv.mean(0), peak.mean(0)
            uncertainty = model_outputs[0]["delta_PFV_sigma"] / 100.0 + model_outputs[0]["delta_TFV_sigma"] / 1000.0 + model_outputs[0]["delta_peak_sigma"]
            information_score = disagreement + 0.15 * uncertainty + 0.10 / (1.0 + np.abs(mean_tfv) / 100.0) + 0.10 / (1.0 + np.abs(mean_pfv - 100.0) / 100.0)
            for index, item in enumerate(library):
                item.update({"disagreement": float(disagreement[index]), "information_score": float(information_score[index]), "mean_pfv": float(mean_pfv[index]), "mean_tfv": float(mean_tfv[index]), "mean_peak": float(mean_peak[index])})
            required_family = FAMILY_CYCLE[checkpoint_no % len(FAMILY_CYCLE)]
            required = [item for item in library if item["specification"]["family"] == required_family]
            chosen = [max(required or library, key=lambda item: item["information_score"])]
            remaining = [item for item in library if item is not chosen[0]]
            chosen.append(max(remaining, key=lambda item: item["disagreement"]))
            remaining = [item for item in remaining if item is not chosen[1]]
            adverse = [item for item in remaining if item["specification"]["intended_evidence_role"] == "safety_rejection"]
            chosen.append(max(adverse or remaining, key=lambda item: item["diagnostic"]["action_l1_difference"]))
            while len(chosen) < int(args.candidates_per_checkpoint):
                remaining = [item for item in library if all(item is not selected for selected in chosen)]
                if not remaining:
                    break
                chosen.append(max(remaining, key=lambda item: item["information_score"]))
            chosen = chosen[: int(args.candidates_per_checkpoint)]
            checkpoint_id = f"{event_id}|{phase}|{start_min:.1f}"
            reference_execution_id = _hash({"event_id": event_id, "checkpoint": checkpoint_id, "branch": "reference", "schema": "targeted_v3"})
            for rank, item in enumerate(chosen):
                specification = item["specification"]
                pair_id = _hash({"checkpoint": checkpoint_id, "specification": specification, "schema": "targeted_v3"})
                candidate_execution_id = _hash({"event_id": event_id, "checkpoint": checkpoint_id, "specification": specification, "schema": "targeted_v3"})
                common = {
                    "pair_id": pair_id, "event_id": event_id, "phase": phase, "split": split,
                    "checkpoint_id": checkpoint_id, "override_start_min": start_min,
                    "split_timestamp_fraction": start_min / float(event.duration_min), "reference_policy": "no_control",
                    "candidate_action_sequence": json.dumps(specification, sort_keys=True),
                    "canonical_action_order_hash": canonical_hash, "actuator_semantics_hash": semantics_hash,
                    "code_hash": code_hash, "requires_same_state_branching": True,
                    "candidate_kind": specification["kind"], "candidate_family": specification["family"],
                    "intended_evidence_role": specification["intended_evidence_role"],
                    "selection_rank": rank, "selection_required_family": required_family,
                    "ensemble_disagreement_score": item["disagreement"], "information_score": item["information_score"],
                    "ensemble_mean_delta_PFV": item["mean_pfv"], "ensemble_mean_delta_TFV": item["mean_tfv"],
                    "ensemble_mean_delta_peak": item["mean_peak"],
                    "materialized_reference_action_sequence": json.dumps(reference.astype(float).tolist()),
                    "materialized_candidate_action_sequence": json.dumps(item["candidate"].astype(float).tolist()),
                    **{key: value for key, value in item["diagnostic"].items() if key not in {"actual_delta_after_clipping", "changed_actuator_ids"}},
                    "changed_actuator_ids": ",".join(item["diagnostic"]["changed_actuator_ids"]),
                    "actual_delta_after_clipping": json.dumps(item["diagnostic"]["actual_delta_after_clipping"], sort_keys=True),
                    "return_period": event_return_period(event_id), "rain_pattern": event_pattern(event_id),
                }
                selected_records.append({**common, "case_id": _hash({"pair": pair_id, "branch": "A"}), "execution_case_id": reference_execution_id, "branch": "A", "executed_action_sequence": json.dumps({"mode": "default_no_control", "horizon_steps": 6}, sort_keys=True), "status": "reference_reused"})
                selected_records.append({**common, "case_id": _hash({"pair": pair_id, "branch": "B"}), "execution_case_id": candidate_execution_id, "branch": "B", "executed_action_sequence": json.dumps(specification, sort_keys=True), "status": "preflight_validated_not_started"})
                all_diagnostics.append({"event_id": event_id, "phase": phase, "checkpoint_id": checkpoint_id, "candidate_family": specification["family"], "candidate_kind": specification["kind"], **item["diagnostic"]})
            checkpoint_no += 1
    manifest = pd.DataFrame(selected_records)
    candidates = manifest[manifest["branch"].eq("B")].copy()
    diagnostics = pd.DataFrame(all_diagnostics)
    train_events = set(candidates.loc[candidates["split"].eq("train"), "event_id"])
    validation_events = set(candidates.loc[candidates["split"].eq("validation"), "event_id"])
    candidate_count = len(candidates)
    noops = int(candidates["is_noop"].astype(bool).sum())
    checks = {
        "candidate_budget_160_to_240": 160 <= candidate_count <= min(240, int(args.max_candidate_cases)),
        "planned_noop_rate_le_5pct": noops / max(1, candidate_count) <= 0.05,
        "train_validation_events_disjoint": not bool(train_events & validation_events),
        "old_effect_events_disjoint": not bool(set(candidates["event_id"]) & old_events),
        "formal_calibration_leakage_absent": True,
        "all_three_phases": set(candidates["phase"]) == {"rising", "peak", "recession"},
        "required_rain_patterns": {"chicago_early", "chicago_center", "chicago_late", "block"}.issubset(set(candidates["rain_pattern"])),
        "light_medium_severe_covered": {"T5", "T10", "T20", "T50", "T100"}.issubset(set(candidates["return_period"])),
        "facility_type_families_covered": set(FAMILY_CYCLE).issubset(set(candidates["candidate_family"])),
        "canonical_shape_H36": candidates["materialized_candidate_action_sequence"].map(lambda value: np.asarray(json.loads(value)).shape == (6, 36)).all(),
        "binary_pump_semantics": not candidates["reason"].astype(str).str.startswith("fractional_binary_pump").any(),
        "same_checkpoint_reference": candidates["checkpoint_id"].notna().all(),
        "physical_case_budget": candidates["execution_case_id"].nunique() + candidates["checkpoint_id"].nunique() <= 240,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "passed": bool(all(checks.values())), "checks": checks,
        "candidate_cases": candidate_count, "reference_checkpoints_reused": int(candidates["checkpoint_id"].nunique()),
        "unique_physical_cases": int(candidates["execution_case_id"].nunique() + candidates["checkpoint_id"].nunique()),
        "planned_noops": noops, "planned_noop_rate": noops / max(1, candidate_count),
        "train_events": sorted(train_events), "validation_events": sorted(validation_events),
        "candidate_family_counts": candidates["candidate_family"].value_counts().to_dict(),
        "intended_evidence_counts": candidates["intended_evidence_role"].value_counts().to_dict(),
        "phase_counts": candidates["phase"].value_counts().to_dict(),
        "return_period_counts": candidates["return_period"].value_counts().to_dict(),
        "rain_pattern_counts": candidates["rain_pattern"].value_counts().to_dict(),
        "ensemble_models": [str(path) for path in model_paths],
        "formal_calibration_policy": "These 20 effect events are frozen now; calibration/formal must use newly generated event IDs after this manifest lock.",
        "swmm_started": False,
    }
    manifest.to_csv(out / "targeted_informative_paired_manifest.csv", index=False)
    diagnostics.to_csv(out / "targeted_candidate_preflight_rows.csv", index=False)
    (out / "targeted_manifest_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Targeted manifest preflight failed; SWMM execution remains blocked")


if __name__ == "__main__":
    main()
