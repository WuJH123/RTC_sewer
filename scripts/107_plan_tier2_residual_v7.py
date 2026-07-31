from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.control.temporal_joint_candidate_search import (
    TemporalJointCandidateConfig,
    validate_candidate_sequence,
)
from sewerrtc.experiments.targeted_joint_pairs import (
    action_window,
    event_pattern,
    event_return_period,
    materialize_candidate,
    sequence_diagnostics,
)
from sewerrtc.experiments.tier2_residual_v7 import (
    BINARY_RESIDUAL_PUMPS,
    build_residual_specifications,
    file_sha256,
    freeze_dataset_manifest,
    select_deployment_tier1_bases,
    select_fresh_event_roles,
)
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.simulation.kpi_metrics import compute_kpis


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_locked_json(path: Path, payload: dict[str, object]) -> None:
    text = json.dumps(payload, indent=2)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"locked artifact differs from requested plan: {path}")
    path.write_text(text, encoding="utf-8")


def _write_locked_csv(path: Path, frame: pd.DataFrame) -> None:
    normalized = frame.fillna("").astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype=str).fillna("")
        if list(existing.columns) != list(normalized.columns) or not existing.equals(normalized):
            raise ValueError(f"locked artifact differs from requested plan: {path}")
        return
    frame.to_csv(path, index=False)


def _phase_start(duration_min: float, phase: str) -> float:
    if phase == "peak":
        return 0.50 * float(duration_min)
    if phase == "recession":
        return float(duration_min) + 30.0
    raise ValueError(f"unsupported v7 deployment phase: {phase}")


def _manifest_rows(
    *,
    event_id: str,
    phase: str,
    split: str,
    role: str,
    start_min: float,
    duration_min: float,
    spec: dict[str, object],
    reference: np.ndarray,
    candidate: np.ndarray,
    diagnostics: dict[str, object],
    canonical_hash: str,
    semantics_hash: str,
    plan_stage: str,
) -> list[dict[str, object]]:
    pair_id = _hash({"event": event_id, "phase": phase, "start": start_min, "spec": spec, "schema": "tier2_residual_v7"})
    reference_spec = {"mode": "default_no_control", "horizon_steps": 6}
    common = {
        "pair_id": pair_id,
        "event_id": event_id,
        "phase": phase,
        "split": split,
        "event_role": role,
        "split_timestamp_fraction": float(start_min) / max(float(duration_min), 1.0),
        "override_start_min": float(start_min),
        "reference_policy": "no_control",
        "candidate_action_sequence": json.dumps(spec, sort_keys=True),
        "canonical_action_order_hash": canonical_hash,
        "actuator_semantics_hash": semantics_hash,
        "requires_same_state_branching": True,
        "status": "validated_plan_not_started",
        "materialized_reference_action_sequence": json.dumps(reference.astype(float).tolist()),
        "materialized_candidate_action_sequence": json.dumps(candidate.astype(float).tolist()),
        "candidate_kind": str(spec.get("kind", "unknown")),
        "candidate_family": str(spec.get("family", spec.get("kind", "unknown"))),
        "actuator_signature": "|".join(map(str, spec.get("actuators", []))),
        "changed_actuator_count": int(diagnostics["changed_actuator_count"]),
        "changed_time_step_count": int(diagnostics["changed_time_step_count"]),
        "max_simultaneous_changes": int(diagnostics["max_simultaneous_changes"]),
        "action_l1_difference": float(diagnostics["action_l1_difference"]),
        "action_linf_difference": float(diagnostics["action_linf_difference"]),
        "rain_pattern": event_pattern(event_id),
        "return_period": event_return_period(event_id),
        "plan_stage": plan_stage,
    }
    rows = []
    for branch, executed in (("A", reference_spec), ("B", spec)):
        payload = {"pair": pair_id, "branch": branch, "executed": executed}
        rows.append({
            **common,
            "case_id": _hash(payload),
            "execution_case_id": _hash(payload),
            "branch": branch,
            "executed_action_sequence": json.dumps(executed, sort_keys=True),
            "source_case_id": pair_id,
        })
    return rows


def _plan_tier1(args: argparse.Namespace, cfg: dict, root: Path, out: Path) -> None:
    base_path = root / args.base_dataset if not Path(args.base_dataset).is_absolute() else Path(args.base_dataset)
    freeze = freeze_dataset_manifest(base_path, intended_rows=int(args.frozen_rows))
    freeze["config"] = str((root / args.config).resolve())
    freeze["config_sha256"] = file_sha256(root / args.config)
    _write_locked_json(out / "frozen_1451_manifest.json", freeze)

    data = np.load(base_path, allow_pickle=True)
    excluded = set(data["event_ids"].astype(str))
    action_ids = data["action_ids"].astype(str).tolist()
    rainfall = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    rainfall = rainfall[
        rainfall["event_id"].astype(str).map(lambda event: (reference_bank / f"{event}__no_control_detail.csv").exists())
    ].copy()
    roles = select_fresh_event_roles(
        rainfall,
        excluded_events=excluded,
        fit_events=int(args.fit_events),
        calibration_events=int(args.calibration_events),
        validation_events=int(args.validation_events),
        seed=int(args.seed),
    )
    role_path = out / "locked_event_roles.csv"
    _write_locked_csv(role_path, roles)
    (out / "calibration_events.txt").write_text(
        "\n".join(roles.loc[roles["role"].eq("calibration"), "event_id"].astype(str)) + "\n", encoding="utf-8"
    )
    (out / "locked_validation_events.txt").write_text(
        "\n".join(roles.loc[roles["role"].eq("locked_validation"), "event_id"].astype(str)) + "\n", encoding="utf-8"
    )

    temporal = ((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {})
    groups = [list(map(str, group)) for group in temporal.get("legacy_groups", [])]
    if len(groups) != 3:
        raise ValueError(f"v7 expects the three frozen Tier 1 groups, got {len(groups)}")
    audit_path = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    canonical_hash = _hash(action_ids)
    semantics_hash = file_sha256(audit_path)
    rows: list[dict[str, object]] = []
    reference_cache: dict[str, pd.DataFrame] = {}
    for event in roles.itertuples(index=False):
        event_id = str(event.event_id)
        reference_frame = reference_cache.setdefault(
            event_id, pd.read_csv(reference_bank / f"{event_id}__no_control_detail.csv")
        )
        for phase in ("peak", "recession"):
            start_min = _phase_start(float(event.duration_min), phase)
            reference = action_window(reference_frame, action_ids=action_ids, start_min=start_min, horizon_steps=6)
            for group_index, group in enumerate(groups):
                for direction in (-1.0, 1.0):
                    profile = [direction * float(args.tier1_delta)] * 3 + [0.0] * 3
                    spec = {
                        "family": "tier1_v8_deployment_screen_v7",
                        "kind": "legacy_group",
                        "mode": f"tier1_group_{group_index}_{direction:+.0f}_{phase}",
                        "actuators": group,
                        "signed_profiles": {actuator_id: profile for actuator_id in group},
                        "horizon_steps": 6,
                        "sequence_semantics": "relative_to_same_state_no_control_reference",
                        "online_candidate_eligible": True,
                        "tier": 1,
                    }
                    candidate = materialize_candidate(reference, action_ids=action_ids, specification=spec)
                    diagnostics = sequence_diagnostics(
                        candidate, reference, action_ids=action_ids,
                        binary_pump_ids=set(BINARY_RESIDUAL_PUMPS), minimum_effective_delta=0.02,
                    )
                    if not diagnostics["valid"]:
                        continue
                    rows.extend(_manifest_rows(
                        event_id=event_id, phase=phase, split=str(event.split), role=str(event.role),
                        start_min=start_min, duration_min=float(event.duration_min), spec=spec, reference=reference, candidate=candidate,
                        diagnostics=diagnostics, canonical_hash=canonical_hash, semantics_hash=semantics_hash,
                        plan_stage="tier1_deployment_screen",
                    ))
    manifest = pd.DataFrame(rows)
    manifest_path = ensure_dir(out / "tier1_screen_plan") / "tier1_base_manifest.csv"
    _write_locked_csv(manifest_path, manifest)
    report = {
        "frozen_base_rows": int(freeze["rows"]),
        "frozen_base_sha256": freeze["sha256"],
        "fresh_events": int(roles["event_id"].nunique()),
        "event_roles": roles.groupby("role")["event_id"].nunique().astype(int).to_dict(),
        "old_new_event_overlap": sorted(excluded & set(roles["event_id"].astype(str))),
        "tier1_candidate_cases": int((manifest["branch"] == "B").sum()),
        "locked_validation_events": roles.loc[roles["role"].eq("locked_validation"), "event_id"].astype(str).tolist(),
        "manifest": str(manifest_path.resolve()),
        "passed": not bool(excluded & set(roles["event_id"].astype(str))),
    }
    _write_locked_json(out / "tier1_plan_preflight.json", report)
    print(json.dumps(report, indent=2))


def _plan_residual(args: argparse.Namespace, cfg: dict, root: Path, out: Path) -> None:
    roles = pd.read_csv(out / "locked_event_roles.csv")
    tier1_results_path = out / "tier1_screen_cases" / "paired_candidate_results.csv"
    if not tier1_results_path.exists():
        raise FileNotFoundError(f"Tier 1 screening results do not exist: {tier1_results_path}")
    tier1 = pd.read_csv(tier1_results_path)
    expected = pd.read_csv(out / "tier1_screen_plan" / "tier1_base_manifest.csv")
    expected_count = int((expected["branch"].astype(str) == "B").sum())
    if tier1["case_id"].astype(str).nunique() != expected_count:
        raise ValueError(f"Tier 1 screening incomplete: {tier1['case_id'].nunique()}/{expected_count}")
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    priority = [line.strip() for line in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines() if line.strip()]
    reference_kpi: dict[str, dict[str, float]] = {}
    records = []
    for row in tier1.itertuples(index=False):
        event_id = str(row.event_id)
        if event_id not in reference_kpi:
            reference_kpi[event_id] = compute_kpis(
                pd.read_csv(reference_bank / f"{event_id}__no_control_detail.csv"),
                priority,
                dt_sec=int(cfg["experiment"]["control_step_sec"]),
            )
        ref = reference_kpi[event_id]
        tier1_specification = json.loads(str(row.executed_action_sequence))
        records.append({
            **row._asdict(),
            "candidate_id": str(row.case_id),
            "tier1_mode": str(tier1_specification.get("mode", "unknown")),
            "reference_PFV": float(ref["PFV"]),
            "reference_TFV": float(ref["TFV"]),
            "reference_peak": float(ref["peak_TFV_rate"]),
            "delta_PFV": float(row.PFV) - float(ref["PFV"]),
            "delta_TFV": float(row.TFV) - float(ref["TFV"]),
            "delta_peak": float(row.peak_TFV_rate) - float(ref["peak_TFV_rate"]),
        })
    effects = pd.DataFrame(records).merge(
        roles[["event_id", "role", "split"]], on="event_id", how="left", suffixes=("", "_locked")
    )
    effects.to_csv(out / "tier1_screen_effects.csv", index=False)
    selected, phase_policy = select_deployment_tier1_bases(
        effects,
        pfv_abs_margin_m3=float(args.pfv_abs_margin_m3),
        pfv_rel_margin=float(args.pfv_rel_margin),
    )
    selected.to_csv(out / "selected_deployment_tier1_bases.csv", index=False)

    data = np.load(root / args.base_dataset if not Path(args.base_dataset).is_absolute() else Path(args.base_dataset), allow_pickle=True)
    action_ids = data["action_ids"].astype(str).tolist()
    audit_path = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    actuators = select_actuators_for_scope(pd.read_csv(audit_path), "control_enabled")
    if actuators["actuator_id"].astype(str).tolist() != action_ids:
        raise ValueError("canonical actuator order differs between dataset and runtime audit")
    search_cfg = (cfg.get("controller", {}).get("temporal_joint", {}).get("candidate_search", {}) or {})
    candidate_config = TemporalJointCandidateConfig(
        horizon_steps=6,
        max_simultaneous_changes=int(search_cfg.get("max_simultaneous_changes", 6)),
        max_change_points=int(search_cfg.get("max_change_points", 2)),
        continuous_max_delta=float(search_cfg.get("continuous_max_delta", 0.20)),
        binary_pump_ids=tuple(search_cfg.get("binary_pump_ids", BINARY_RESIDUAL_PUMPS)),
        binary_pump_min_dwell_steps=int(search_cfg.get("binary_pump_min_dwell_steps", 2)),
        storage_interlock=bool(search_cfg.get("storage_interlock", True)),
        max_storage_actuators=int(search_cfg.get("max_storage_actuators", 4)),
    )
    canonical_hash = _hash(action_ids)
    semantics_hash = file_sha256(audit_path)
    role_priority = {"locked_validation": 0, "calibration": 1, "fit": 2}
    selected = selected.sort_values(
        ["role", "event_id", "phase"],
        key=lambda series: series.map(role_priority) if series.name == "role" else series,
    )
    reference_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    residual_cases = 0
    rejection_rows = []
    for context in selected.itertuples(index=False):
        if residual_cases >= int(args.max_residual_cases):
            break
        event_id = str(context.event_id)
        reference_frame = reference_cache.setdefault(
            event_id, pd.read_csv(reference_bank / f"{event_id}__no_control_detail.csv")
        )
        start_min = float(context.override_start_min)
        reference = action_window(reference_frame, action_ids=action_ids, start_min=start_min, horizon_steps=6)
        tier1_spec = json.loads(str(context.executed_action_sequence))
        base_profiles = dict(tier1_spec.get("signed_profiles", {}) or {})
        magnitude = 0.05 if residual_cases % 2 == 0 else 0.10
        specs = build_residual_specifications(
            action_ids=action_ids,
            no_control_reference=reference,
            tier1_signed_profiles=base_profiles,
            phase=str(context.phase),
            magnitude=magnitude,
        )
        tier1_sequence = materialize_candidate(reference, action_ids=action_ids, specification=tier1_spec)
        for spec in specs:
            if residual_cases >= int(args.max_residual_cases):
                break
            candidate = materialize_candidate(reference, action_ids=action_ids, specification=spec)
            residual_changed = bool(np.any(np.abs(candidate - tier1_sequence) > 1.0e-7))
            diagnostics = sequence_diagnostics(
                candidate, reference, action_ids=action_ids,
                binary_pump_ids=set(BINARY_RESIDUAL_PUMPS), minimum_effective_delta=0.02,
            )
            engineering = validate_candidate_sequence(candidate, reference, actuators, candidate_config)
            if not residual_changed or not diagnostics["valid"] or not engineering["valid"]:
                rejection_rows.append({
                    "event_id": event_id, "phase": context.phase, "mode": spec["mode"],
                    "residual_changed": residual_changed, "sequence_reason": diagnostics["reason"],
                    "engineering_reason": engineering["reason"],
                })
                continue
            spec["tier1_base_case_id"] = str(context.case_id)
            spec["tier1_base_mode"] = str(tier1_spec.get("mode", "unknown"))
            rows.extend(_manifest_rows(
                event_id=event_id, phase=str(context.phase), split=str(context.split_locked),
                role=str(context.role), start_min=start_min, duration_min=float(context.duration_min), spec=spec, reference=reference,
                candidate=candidate, diagnostics=diagnostics, canonical_hash=canonical_hash,
                semantics_hash=semantics_hash, plan_stage="tier2_deployment_residual",
            ))
            residual_cases += 1
    manifest = pd.DataFrame(rows)
    manifest_path = ensure_dir(out / "residual_plan") / "tier2_residual_manifest.csv"
    _write_locked_csv(manifest_path, manifest)
    pd.DataFrame(rejection_rows).to_csv(out / "residual_plan" / "preflight_rejections.csv", index=False)
    candidate_rows = manifest[manifest["branch"].astype(str).eq("B")]
    validation_events = set(candidate_rows.loc[candidate_rows["event_role"].eq("locked_validation"), "event_id"].astype(str))
    locked_events = set(roles.loc[roles["role"].eq("locked_validation"), "event_id"].astype(str))
    report = {
        "tier1_screen_rows": int(len(tier1)),
        "selected_tier1_contexts": int(len(selected)),
        "fit_safe_tier1_contexts": int(selected["selection_basis"].eq("fit_same_context_true_safety").sum()),
        "fit_only_phase_policy": phase_policy,
        "planned_residual_cases": int(len(candidate_rows)),
        "target_residual_cases": int(args.max_residual_cases),
        "minimum_residual_cases": int(args.min_residual_cases),
        "candidate_kind_counts": candidate_rows["candidate_kind"].value_counts().astype(int).to_dict(),
        "event_role_rows": candidate_rows["event_role"].value_counts().astype(int).to_dict(),
        "locked_validation_events": sorted(locked_events),
        "locked_validation_events_with_cases": sorted(validation_events),
        "locked_validation_event_coverage": int(len(validation_events)),
        "old_validation_reused": False,
        "validation_labels_used_for_candidate_selection": False,
        "manifest": str(manifest_path.resolve()),
        "preflight_rejections": int(len(rejection_rows)),
        "passed": bool(
            len(candidate_rows) >= int(args.min_residual_cases)
            and len(validation_events) >= int(args.min_locked_validation_events)
            and validation_events.issubset(locked_events)
        ),
    }
    _write_locked_json(out / "residual_plan" / "targeted_manifest_preflight.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan deployment-aligned Tier 1 -> Tier 2 residual v7 data.")
    parser.add_argument("--stage", choices=("freeze_and_plan_tier1", "plan_residual"), required=True)
    parser.add_argument("--config", default="configs/wuhan_project6_36_hierarchical_residual_v7.yaml")
    parser.add_argument("--base-dataset", default="outputs/project6_36_causal_effect_coverage_v2/effect_dataset_boundary_v6_round2/same_state_raw_joint_36_causal_effect_boundary_v6_round2.npz")
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--out-dir", default="outputs/project6_36_tier2_residual_v7")
    parser.add_argument("--frozen-rows", type=int, default=1451)
    parser.add_argument("--fit-events", type=int, default=12)
    parser.add_argument("--calibration-events", type=int, default=4)
    parser.add_argument("--validation-events", type=int, default=8)
    parser.add_argument("--tier1-delta", type=float, default=0.05)
    parser.add_argument("--max-residual-cases", type=int, default=760)
    parser.add_argument("--min-residual-cases", type=int, default=700)
    parser.add_argument("--min-locked-validation-events", type=int, default=6)
    parser.add_argument("--pfv-abs-margin-m3", type=float, default=100.0)
    parser.add_argument("--pfv-rel-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    if args.stage == "freeze_and_plan_tier1":
        _plan_tier1(args, cfg, root, out)
    else:
        _plan_residual(args, cfg, root, out)


if __name__ == "__main__":
    main()
