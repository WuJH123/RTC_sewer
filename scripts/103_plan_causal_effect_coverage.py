from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.experiments.causal_effect_coverage import (
    build_phase_profile_library,
    build_coverage_gaps,
    summarize_effect_coverage,
)
from sewerrtc.experiments.targeted_joint_pairs import (
    action_window,
    event_pattern,
    event_return_period,
    materialize_candidate,
    sequence_diagnostics,
)
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary_profile(reference: np.ndarray, target: float, phase: str) -> list[float]:
    base = (np.asarray(reference, dtype=np.float32) >= 0.5).astype(np.float32)
    if phase == "rising":
        base[1:5] = float(target)
    elif phase == "peak":
        base[:5] = float(target)
    elif phase == "recession":
        base[:3] = float(target)
    else:
        raise ValueError(f"unknown phase: {phase}")
    return base.astype(float).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_causal_effect_v3.yaml")
    parser.add_argument(
        "--dataset",
        default="outputs/project6_36_temporal_joint_peakfixed_v1/effect_dataset/same_state_raw_joint_36_peakfixed_v1.npz",
    )
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--out-dir", default="outputs/project6_36_causal_effect_v3/paired_plan")
    parser.add_argument("--min-train-events", type=int, default=5)
    parser.add_argument("--min-validation-events", type=int, default=3)
    parser.add_argument("--max-candidate-cases", type=int, default=800)
    parser.add_argument("--minimum-effective-delta", type=float, default=0.05)
    parser.add_argument("--magnitude-levels", default="0.05,0.10,0.20")
    parser.add_argument("--profiles-per-context", type=int, default=2)
    parser.add_argument("--joint-case-fraction", type=float, default=0.15)
    args = parser.parse_args()

    magnitude_levels = tuple(
        sorted({abs(float(value)) for value in str(args.magnitude_levels).split(",") if str(value).strip()})
    )
    if not magnitude_levels:
        raise ValueError("at least one positive magnitude level is required")
    if any(value <= 0.0 or value > 1.0 for value in magnitude_levels):
        raise ValueError("magnitude levels must be in (0, 1]")
    if int(args.profiles_per_context) < 1:
        raise ValueError("profiles-per-context must be at least 1")
    if not 0.0 <= float(args.joint_case_fraction) < 0.5:
        raise ValueError("joint-case-fraction must be in [0, 0.5)")

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    dataset_path = root / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    data = np.load(dataset_path, allow_pickle=True)
    action_ids = data["action_ids"].astype(str).tolist()
    if len(action_ids) != 36 or data["candidate_action_seq"].shape[1:] != (6, 36):
        raise ValueError("coverage planning requires the strict canonical [N,6,36] dataset")
    event_ids = data["event_ids"].astype(str)
    splits = data["split"].astype(str)
    phases = data["phase"].astype(str)
    if any(set(event_ids[splits == split]) & set(event_ids[splits != split]) for split in set(splits)):
        raise ValueError("existing dataset contains event leakage across splits")

    coverage = summarize_effect_coverage(
        event_ids=event_ids,
        splits=splits,
        phases=phases,
        candidate_action_seq=data["candidate_action_seq"],
        reference_action_seq=data["reference_action_seq"],
        action_ids=action_ids,
    )
    coverage.to_csv(out / "facility_direction_phase_event_coverage.csv", index=False)
    gaps = build_coverage_gaps(
        coverage,
        action_ids=action_ids,
        min_train_events=int(args.min_train_events),
        min_validation_events=int(args.min_validation_events),
    )
    gap_frame = pd.DataFrame(gaps)
    gap_frame.to_csv(out / "facility_direction_phase_event_gaps.csv", index=False)
    reference_risk = data["reference_risk_rate_seq"].astype(np.float64)
    delta_risk = data["delta_risk_rate_seq"].astype(np.float64)
    joint_rows = []
    for row_index, row_residual in enumerate(data["candidate_action_seq"] - data["reference_action_seq"]):
        changed = np.flatnonzero(np.any(np.abs(row_residual) > 1.0e-7, axis=0))
        if len(changed) < 2:
            continue
        candidate_tfv = reference_risk[row_index, :, 1] + delta_risk[row_index, :, 1]
        joint_rows.append({
            "actuator_signature": "|".join(action_ids[index] for index in changed),
            "event_id": str(event_ids[row_index]),
            "phase": str(phases[row_index]),
            "delta_PFV_m3": float(delta_risk[row_index, :, 0].sum() * 300.0),
            "delta_TFV_m3": float(delta_risk[row_index, :, 1].sum() * 300.0),
            "delta_peak": float(candidate_tfv.max() - reference_risk[row_index, :, 1].max()),
        })
    if joint_rows:
        joint_evidence = (
            pd.DataFrame(joint_rows)
            .groupby("actuator_signature", as_index=False)
            .agg(
                rows=("event_id", "size"),
                independent_events=("event_id", "nunique"),
                phases=("phase", lambda values: "|".join(sorted(set(values)))),
                median_delta_PFV_m3=("delta_PFV_m3", "median"),
                median_delta_TFV_m3=("delta_TFV_m3", "median"),
                median_delta_peak=("delta_peak", "median"),
            )
            .sort_values(["independent_events", "rows"], ascending=False)
        )
    else:
        joint_evidence = pd.DataFrame()
    joint_evidence.to_csv(out / "joint_action_evidence.csv", index=False)

    temporal = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}))
    binary_pumps = set((temporal.get("candidate_search", {}) or {}).get("binary_pump_ids", []))
    rainfall = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    event_split = {}
    for event_id, split in zip(event_ids, splits):
        event_split.setdefault(str(event_id), str(split))
    residual = data["candidate_action_seq"] - data["reference_action_seq"]
    existing_events: dict[tuple[str, str, str, str], set[str]] = {}
    for row_index, event_id in enumerate(event_ids):
        for action_index, actuator_id in enumerate(action_ids):
            values = residual[row_index, :, action_index]
            for direction, active in (
                ("increase", np.any(values > 1.0e-7)),
                ("decrease", np.any(values < -1.0e-7)),
            ):
                if active:
                    existing_events.setdefault(
                        (actuator_id, direction, str(phases[row_index]), str(splits[row_index])), set()
                    ).add(str(event_id))

    audit_table = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    canonical_hash = _hash(action_ids)
    semantics_hash = _file_hash(audit_table)
    phase_fraction = {"rising": 0.25, "peak": 0.50, "recession": 0.80}
    event_candidates = {
        split: sorted(
            [
                event for event, event_label in event_split.items()
                if event_label == split and event in rainfall.index and (reference_bank / f"{event}__no_control_detail.csv").exists()
            ],
            key=lambda event: (event_pattern(event), event_return_period(event), event),
        )
        for split in ("train", "validation")
    }
    reference_cache: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []
    unplanned: list[dict[str, object]] = []
    planned_signatures: set[str] = set()
    planned_contexts: list[dict[str, object]] = []
    stage_counts = {"independent_event_coverage": 0, "magnitude_timing_richness": 0, "joint_action_evidence": 0}
    max_cases = int(args.max_candidate_cases)
    joint_case_target = int(round(max_cases * float(args.joint_case_fraction)))
    single_case_budget = max_cases - joint_case_target
    planned_candidates = 0

    def reference_for(event_id: str, phase: str) -> tuple[np.ndarray, float]:
        event = rainfall.loc[event_id]
        start_min = float(event.duration_min) * phase_fraction[str(phase)]
        if event_id not in reference_cache:
            reference_cache[event_id] = pd.read_csv(reference_bank / f"{event_id}__no_control_detail.csv")
        return (
            action_window(reference_cache[event_id], action_ids=action_ids, start_min=start_min, horizon_steps=6),
            start_min,
        )

    def append_case(
        *,
        event_id: str,
        phase: str,
        split: str,
        spec: dict[str, object],
        reference: np.ndarray,
        start_min: float,
        direction: str,
        magnitude: float,
        plan_stage: str,
        profile_variant: str,
    ) -> bool:
        nonlocal planned_candidates
        candidate = materialize_candidate(reference, action_ids=action_ids, specification=spec)
        diagnostics = sequence_diagnostics(
            candidate,
            reference,
            action_ids=action_ids,
            binary_pump_ids=binary_pumps,
            minimum_effective_delta=float(args.minimum_effective_delta),
        )
        if not diagnostics["valid"]:
            return False
        sequence_key = _hash({
            "event": event_id,
            "phase": phase,
            "sequence": np.round(candidate, 6).astype(float).tolist(),
        })
        if sequence_key in planned_signatures:
            return False
        planned_signatures.add(sequence_key)
        actuators = [str(value) for value in spec.get("actuators", [])]
        pair_id = _hash({"event": event_id, "phase": phase, "spec": spec, "schema": "causal_effect_v3_coverage_v2"})
        reference_spec = {"mode": "default_no_control", "horizon_steps": 6}
        common = {
            "pair_id": pair_id,
            "event_id": event_id,
            "phase": phase,
            "split": split,
            "split_timestamp_fraction": phase_fraction[phase],
            "override_start_min": start_min,
            "reference_policy": "no_control",
            "candidate_action_sequence": json.dumps(spec, sort_keys=True),
            "canonical_action_order_hash": canonical_hash,
            "actuator_semantics_hash": semantics_hash,
            "requires_same_state_branching": True,
            "status": "validated_plan_not_started",
            "materialized_candidate_action_sequence": json.dumps(candidate.astype(float).tolist()),
            "planned_direction": direction,
            "planned_magnitude": float(magnitude),
            "profile_variant": profile_variant,
            "plan_stage": plan_stage,
            "candidate_kind": str(spec.get("kind", "unknown")),
            "actuator_signature": "|".join(actuators),
            "changed_actuator_count": int(diagnostics["changed_actuator_count"]),
            "changed_time_step_count": int(diagnostics["changed_time_step_count"]),
            "rain_pattern": event_pattern(event_id),
            "return_period": event_return_period(event_id),
        }
        for branch, executed in (("A", reference_spec), ("B", spec)):
            payload = {
                "event": event_id,
                "start": start_min,
                "branch": branch,
                "executed": executed,
                "schema": "causal_effect_v3_coverage_v2",
            }
            manifest_rows.append({
                **common,
                "case_id": _hash(payload),
                "execution_case_id": _hash(payload),
                "branch": branch,
                "executed_action_sequence": json.dumps(executed, sort_keys=True),
                "source_case_id": pair_id,
            })
        stage_counts[plan_stage] += 1
        planned_candidates += 1
        return True

    for gap in sorted(gaps, key=lambda row: (-int(row["missing_events"]), row["split"], row["phase"], row["actuator_id"], row["direction"])):
        missing = int(gap["missing_events"])
        if missing <= 0:
            continue
        if planned_candidates >= single_case_budget:
            unplanned.append({**gap, "reason": "case_budget_exhausted"})
            continue
        cell = (str(gap["actuator_id"]), str(gap["direction"]), str(gap["phase"]), str(gap["split"]))
        excluded = existing_events.get(cell, set())
        selected = 0
        # Rotate the event pool deterministically per coverage cell. This avoids
        # repeatedly selecting the first rain pattern/return period for every
        # actuator while keeping the manifest reproducible.
        ordered_events = sorted(
            event_candidates[str(gap["split"])],
            key=lambda event: _hash({"cell": cell, "event": event}),
        )
        for event_id in ordered_events:
            if event_id in excluded or selected >= missing or planned_candidates >= single_case_budget:
                continue
            reference, start_min = reference_for(event_id, str(gap["phase"]))
            action_index = action_ids.index(str(gap["actuator_id"]))
            if str(gap["actuator_id"]) in binary_pumps:
                target = 1.0 if str(gap["direction"]) == "increase" else 0.0
                spec = {
                    "family": "causal_coverage_binary_pump",
                    "kind": "single_binary",
                    "mode": f"{gap['direction']}_{gap['phase']}",
                    "actuators": [str(gap["actuator_id"])],
                    "target_profile": _binary_profile(reference[:, action_index], target, str(gap["phase"])),
                    "horizon_steps": 6,
                    "sequence_semantics": "relative_to_same_state_no_control_reference",
                }
                accepted = append_case(
                    event_id=event_id,
                    phase=str(gap["phase"]),
                    split=str(gap["split"]),
                    spec=spec,
                    reference=reference,
                    start_min=start_min,
                    direction=str(gap["direction"]),
                    magnitude=1.0,
                    plan_stage="independent_event_coverage",
                    profile_variant="binary_phase_hold",
                )
            else:
                library = build_phase_profile_library(
                    str(gap["direction"]),
                    magnitudes=magnitude_levels,
                    phase=str(gap["phase"]),
                    horizon_steps=6,
                )
                library = sorted(
                    library,
                    key=lambda item: _hash({"cell": cell, "event": event_id, "profile": item["variant"], "magnitude": item["magnitude"]}),
                )
                accepted = False
                selected_profile = None
                for profile in library:
                    spec = {
                        "family": "causal_coverage_single_continuous",
                        "kind": "single_continuous",
                        "mode": f"{gap['direction']}_{gap['phase']}_{profile['variant']}_{profile['magnitude']:.2f}",
                        "actuators": [str(gap["actuator_id"])],
                        "signed_profile": np.asarray(profile["profile"], dtype=np.float32).astype(float).tolist(),
                        "horizon_steps": 6,
                        "sequence_semantics": "relative_to_same_state_no_control_reference",
                    }
                    accepted = append_case(
                        event_id=event_id,
                        phase=str(gap["phase"]),
                        split=str(gap["split"]),
                        spec=spec,
                        reference=reference,
                        start_min=start_min,
                        direction=str(gap["direction"]),
                        magnitude=float(profile["magnitude"]),
                        plan_stage="independent_event_coverage",
                        profile_variant=str(profile["variant"]),
                    )
                    if accepted:
                        selected_profile = (float(profile["magnitude"]), str(profile["variant"]))
                        break
                if accepted:
                    planned_contexts.append({
                        "event_id": event_id,
                        "phase": str(gap["phase"]),
                        "split": str(gap["split"]),
                        "direction": str(gap["direction"]),
                        "actuator_id": str(gap["actuator_id"]),
                        "reference": reference,
                        "start_min": start_min,
                        "selected_profile": selected_profile,
                    })
            if not accepted:
                continue
            existing_events.setdefault(cell, set()).add(event_id)
            selected += 1
        if selected < missing:
            unplanned.append({
                **gap,
                "planned_events": selected,
                "reason": "reference_boundary_or_insufficient_non_noop_states",
            })

    # Add within-checkpoint amplitude/timing contrasts only after independent
    # event coverage has been attempted. These rows improve identifiability but
    # never count as additional independent events.
    for context in sorted(planned_contexts, key=lambda item: _hash({key: value for key, value in item.items() if key != "reference"})):
        if planned_candidates >= single_case_budget:
            break
        added = 0
        library = build_phase_profile_library(
            str(context["direction"]),
            magnitudes=magnitude_levels,
            phase=str(context["phase"]),
            horizon_steps=6,
        )
        library = sorted(
            library,
            key=lambda item: _hash({"context": context["event_id"], "actuator": context["actuator_id"], "profile": item["variant"], "magnitude": item["magnitude"]}),
        )
        for profile in library:
            if added >= max(0, int(args.profiles_per_context) - 1) or planned_candidates >= single_case_budget:
                break
            if (float(profile["magnitude"]), str(profile["variant"])) == context["selected_profile"]:
                continue
            spec = {
                "family": "causal_coverage_single_continuous",
                "kind": "single_continuous",
                "mode": f"{context['direction']}_{context['phase']}_{profile['variant']}_{profile['magnitude']:.2f}",
                "actuators": [str(context["actuator_id"])],
                "signed_profile": np.asarray(profile["profile"], dtype=np.float32).astype(float).tolist(),
                "horizon_steps": 6,
                "sequence_semantics": "relative_to_same_state_no_control_reference",
            }
            if append_case(
                event_id=str(context["event_id"]),
                phase=str(context["phase"]),
                split=str(context["split"]),
                spec=spec,
                reference=np.asarray(context["reference"], dtype=np.float32),
                start_min=float(context["start_min"]),
                direction=str(context["direction"]),
                magnitude=float(profile["magnitude"]),
                plan_stage="magnitude_timing_richness",
                profile_variant=str(profile["variant"]),
            ):
                added += 1

    # Reserve a bounded fraction for joint-action interaction evidence. The
    # configured groups are explicit hydraulic hypotheses, not arbitrary
    # combinations of all 36 actuators.
    paired_groups = [
        [str(actuator_id) for actuator_id in group if str(actuator_id) in action_ids]
        for group in (temporal.get("paired_groups", []) or [])
    ]
    opportunities: list[tuple[str, str, str, list[str], dict[str, object]]] = []
    for split in ("train", "validation"):
        for event_id in event_candidates[split]:
            for phase in phase_fraction:
                for group in paired_groups:
                    if len(group) < 2:
                        continue
                    for direction in ("decrease", "increase"):
                        for profile in build_phase_profile_library(
                            direction,
                            magnitudes=magnitude_levels,
                            phase=phase,
                            horizon_steps=6,
                        ):
                            opportunities.append((split, event_id, phase, group, {**profile, "direction": direction}))
    opportunities.sort(key=lambda item: _hash({
        "split": item[0], "event": item[1], "phase": item[2], "group": item[3],
        "direction": item[4]["direction"], "magnitude": item[4]["magnitude"], "variant": item[4]["variant"],
    }))
    for split, event_id, phase, group, profile in opportunities:
        if stage_counts["joint_action_evidence"] >= joint_case_target or planned_candidates >= max_cases:
            break
        reference, start_min = reference_for(event_id, phase)
        signed_profile = np.asarray(profile["profile"], dtype=np.float32).astype(float).tolist()
        spec = {
            "family": "causal_coverage_joint_group",
            "kind": "joint_continuous",
            "mode": f"{profile['direction']}_{phase}_{profile['variant']}_{profile['magnitude']:.2f}",
            "actuators": group,
            "signed_profiles": {actuator_id: signed_profile for actuator_id in group},
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
        }
        append_case(
            event_id=event_id,
            phase=phase,
            split=split,
            spec=spec,
            reference=reference,
            start_min=start_min,
            direction=str(profile["direction"]),
            magnitude=float(profile["magnitude"]),
            plan_stage="joint_action_evidence",
            profile_variant=str(profile["variant"]),
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out / "targeted_causal_effect_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(unplanned).to_csv(out / "unplanned_coverage_gaps.csv", index=False)
    candidate_rows = manifest[manifest.get("branch", pd.Series(dtype=str)).astype(str).eq("B")] if len(manifest) else manifest
    train_events = set(candidate_rows.loc[candidate_rows["split"].eq("train"), "event_id"]) if len(candidate_rows) else set()
    validation_events = set(candidate_rows.loc[candidate_rows["split"].eq("validation"), "event_id"]) if len(candidate_rows) else set()
    stage_distribution = (
        candidate_rows["plan_stage"].value_counts().sort_index().astype(int).to_dict() if len(candidate_rows) else {}
    )
    magnitude_distribution = (
        candidate_rows["planned_magnitude"].round(3).value_counts().sort_index().astype(int).to_dict()
        if len(candidate_rows) else {}
    )
    profile_distribution = (
        candidate_rows["profile_variant"].value_counts().sort_index().astype(int).to_dict()
        if len(candidate_rows) else {}
    )
    preflight = {
        "dataset": str(dataset_path),
        "existing_rows": int(len(event_ids)),
        "existing_events": int(len(set(event_ids))),
        "canonical_actions": len(action_ids),
        "coverage_cells": int(len(gaps)),
        "initial_missing_event_slots": int(gap_frame["missing_events"].sum()),
        "planned_candidate_cases": int(len(candidate_rows)),
        "recommended_candidate_case_budget": max_cases,
        "independent_event_targets": {
            "train": int(args.min_train_events),
            "validation": int(args.min_validation_events),
        },
        "magnitude_levels": list(magnitude_levels),
        "profiles_per_context": int(args.profiles_per_context),
        "joint_case_target": joint_case_target,
        "stage_distribution": stage_distribution,
        "magnitude_distribution": {str(key): value for key, value in magnitude_distribution.items()},
        "profile_distribution": profile_distribution,
        "planned_events_by_split": {
            "train": len(train_events),
            "validation": len(validation_events),
        },
        "planned_rain_patterns": sorted(candidate_rows["rain_pattern"].astype(str).unique().tolist()) if len(candidate_rows) else [],
        "planned_return_periods": sorted(candidate_rows["return_period"].astype(str).unique().tolist()) if len(candidate_rows) else [],
        "logical_manifest_rows": int(len(manifest)),
        "no_op_cases": 0,
        "train_validation_event_overlap": sorted(train_events & validation_events),
        "unplanned_gap_cells": len(unplanned),
        "max_candidate_cases": int(args.max_candidate_cases),
        "manifest": str(manifest_path),
        "passed": bool(len(candidate_rows) > 0 and not (train_events & validation_events)),
        "formal_or_calibration_events_added": False,
        "coverage_interpretation": (
            "Unplanned increase-direction cells at a saturated No-control setting are physical boundary gaps, "
            "not evidence that zero-effect duplicates should be generated."
        ),
    }
    (out / "targeted_manifest_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps(preflight, indent=2))


if __name__ == "__main__":
    main()
