from __future__ import annotations

import argparse
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
from sewerrtc.experiments.tier2_residual_v8 import (
    BINARY_RESIDUAL_PUMPS_V8,
    allocate_v8_case_budget,
    build_v8_boundary_specifications,
    build_v8_deployment_residual_specifications,
    file_sha256,
    phase_start_min,
    select_v8_event_roles,
    stable_hash,
    summarize_v8_manifest_preflight,
    temporal_delta_profile,
)
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_locked_csv(path: Path, frame: pd.DataFrame) -> None:
    normalized = frame.fillna("").astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype=str).fillna("")
        if list(existing.columns) != list(normalized.columns) or not existing.equals(normalized):
            raise ValueError(f"locked CSV differs from requested plan: {path}")
        return
    frame.to_csv(path, index=False)


def _write_locked_json(path: Path, payload: dict[str, object]) -> None:
    text = json.dumps(payload, indent=2)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"locked JSON differs from requested plan: {path}")
    path.write_text(text, encoding="utf-8")


def _reference_for(
    *,
    reference_cache: dict[str, pd.DataFrame],
    reference_bank: Path,
    event_id: str,
    action_ids: list[str],
    start_min: float,
) -> np.ndarray:
    if event_id not in reference_cache:
        detail_path = reference_bank / f"{event_id}__no_control_detail.csv"
        if not detail_path.exists():
            raise FileNotFoundError(f"missing No-control reference detail: {detail_path}")
        reference_cache[event_id] = pd.read_csv(detail_path)
    return action_window(
        reference_cache[event_id],
        action_ids=action_ids,
        start_min=float(start_min),
        horizon_steps=6,
    )


def _tier1_profiles(
    *,
    legacy_groups: list[list[str]],
    phase: str,
    context_index: int,
) -> dict[str, list[float]]:
    if not legacy_groups:
        return {}
    group = legacy_groups[int(context_index) % len(legacy_groups)]
    direction = -1.0 if (int(context_index) // max(1, len(legacy_groups))) % 2 == 0 else 1.0
    delta = 0.05 * direction
    return {
        str(actuator_id): temporal_delta_profile(delta, phase, variant="hold")
        for actuator_id in group
    }


def _case_rows(
    *,
    event_id: str,
    role: str,
    split: str,
    phase: str,
    start_min: float,
    duration_min: float,
    reference: np.ndarray,
    candidate: np.ndarray,
    specification: dict[str, object],
    diagnostics: dict[str, object],
    canonical_hash: str,
    semantics_hash: str,
    plan_bucket: str,
    selection_rank: int,
) -> list[dict[str, object]]:
    pair_id = stable_hash({
        "event_id": event_id,
        "phase": phase,
        "start_min": round(float(start_min), 6),
        "specification": specification,
        "schema": "tier2_residual_v8_expansion",
    })
    reference_spec = {"mode": "default_no_control", "horizon_steps": 6}
    common = {
        "pair_id": pair_id,
        "event_id": event_id,
        "event_role": role,
        "split": split,
        "phase": phase,
        "checkpoint_id": f"{event_id}|{phase}|{float(start_min):.3f}",
        "override_start_min": float(start_min),
        "split_timestamp_fraction": float(start_min) / max(float(duration_min), 1.0),
        "reference_policy": "no_control",
        "candidate_action_sequence": _json(specification),
        "canonical_action_order_hash": canonical_hash,
        "actuator_semantics_hash": semantics_hash,
        "requires_same_state_branching": True,
        "status": "validated_plan_not_started",
        "materialized_reference_action_sequence": json.dumps(reference.astype(float).tolist()),
        "materialized_candidate_action_sequence": json.dumps(candidate.astype(float).tolist()),
        "candidate_kind": str(specification.get("kind", "unknown")),
        "candidate_family": str(specification.get("family", specification.get("kind", "unknown"))),
        "actuator_signature": "|".join(map(str, specification.get("actuators", []))),
        "intended_evidence_role": str(specification.get("intended_evidence_role", "unknown")),
        "online_candidate_eligible": bool(specification.get("online_candidate_eligible", False)),
        "plan_bucket": plan_bucket,
        "selection_rank": int(selection_rank),
        "return_period": event_return_period(event_id),
        "rain_pattern": event_pattern(event_id),
        "changed_actuator_count": int(diagnostics["changed_actuator_count"]),
        "changed_time_step_count": int(diagnostics["changed_time_step_count"]),
        "max_simultaneous_changes": int(diagnostics["max_simultaneous_changes"]),
        "action_l1_difference": float(diagnostics["action_l1_difference"]),
        "action_linf_difference": float(diagnostics["action_linf_difference"]),
        "is_noop": bool(diagnostics["is_noop"]),
        "changed_actuator_ids": ",".join(map(str, diagnostics["changed_actuator_ids"])),
        "actual_delta_after_clipping": _json(diagnostics["actual_delta_after_clipping"]),
    }
    rows: list[dict[str, object]] = []
    for branch, executed in (("A", reference_spec), ("B", specification)):
        payload = {
            "pair_id": pair_id,
            "branch": branch,
            "executed": executed,
            "schema": "tier2_residual_v8_expansion",
        }
        rows.append({
            **common,
            "case_id": stable_hash(payload),
            "execution_case_id": stable_hash(payload),
            "branch": branch,
            "executed_action_sequence": _json(executed),
            "source_case_id": pair_id,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan large deployment-aware Tier2 residual-v8 paired data expansion.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_hierarchical_residual_v7.yaml")
    parser.add_argument("--base-dataset", default="outputs/project6_36_tier2_residual_v7/effect_dataset/same_state_raw_joint_36_tier2_residual_v7.npz")
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--out-dir", default="outputs/project6_36_tier2_residual_v8_expansion_balanced")
    parser.add_argument("--target-cases", type=int, default=1050)
    parser.add_argument("--fit-events", type=int, default=14)
    parser.add_argument("--calibration-events", type=int, default=6)
    parser.add_argument("--validation-events", type=int, default=8)
    parser.add_argument("--train-boundary-cases", type=int, default=360)
    parser.add_argument("--calibration-boundary-cases", type=int, default=180)
    parser.add_argument("--validation-boundary-cases", type=int, default=240)
    parser.add_argument("--min-locked-validation-cases", type=int, default=200)
    parser.add_argument("--min-locked-validation-events", type=int, default=6)
    parser.add_argument("--max-simultaneous-changes", type=int, default=8)
    parser.add_argument("--minimum-effective-delta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    base_path = root / args.base_dataset if not Path(args.base_dataset).is_absolute() else Path(args.base_dataset)
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    data = np.load(base_path, allow_pickle=True)
    action_ids = data["action_ids"].astype(str).tolist()
    if len(action_ids) != 36 or tuple(data["candidate_action_seq"].shape[1:]) != (6, 36):
        raise ValueError("v8 expansion requires a strict [N,6,36] base dataset")

    rainfall = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    rainfall = rainfall[
        rainfall["event_id"].astype(str).map(lambda event: (reference_bank / f"{event}__no_control_detail.csv").exists())
    ].copy()
    roles = select_v8_event_roles(
        rainfall,
        excluded_events=set(data["event_ids"].astype(str)),
        fit_events=int(args.fit_events),
        calibration_events=int(args.calibration_events),
        validation_events=int(args.validation_events),
        seed=int(args.seed),
    )
    _write_locked_csv(out / "locked_event_roles.csv", roles)
    (out / "calibration_events.txt").write_text(
        "\n".join(roles.loc[roles["role"].eq("calibration"), "event_id"].astype(str)) + "\n",
        encoding="utf-8",
    )
    (out / "locked_validation_events.txt").write_text(
        "\n".join(roles.loc[roles["role"].eq("locked_validation"), "event_id"].astype(str)) + "\n",
        encoding="utf-8",
    )

    budget = allocate_v8_case_budget(
        total_cases=int(args.target_cases),
        train_boundary_cases=int(args.train_boundary_cases),
        calibration_cases=int(args.calibration_boundary_cases),
        validation_cases=int(args.validation_boundary_cases),
    )
    audit_path = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    canonical_hash = stable_hash(action_ids)
    semantics_hash = file_sha256(audit_path)
    temporal = (cfg.get("controller", {}).get("temporal_joint", {}) or {})
    legacy_groups = [
        [str(actuator) for actuator in group if str(actuator) in action_ids]
        for group in temporal.get("legacy_groups", [])
    ]
    legacy_groups = [group for group in legacy_groups if group]
    contexts: dict[str, list[dict[str, object]]] = {"fit": [], "calibration": [], "locked_validation": []}
    rain_by_event = rainfall.set_index("event_id")
    for row in roles.itertuples(index=False):
        for phase in ("rising", "peak", "recession"):
            duration = float(getattr(row, "duration_min"))
            contexts[str(row.role)].append({
                "event_id": str(row.event_id),
                "role": str(row.role),
                "split": str(row.split),
                "phase": phase,
                "duration_min": duration,
                "start_min": phase_start_min(duration, phase),
            })
    for role in contexts:
        contexts[role].sort(key=lambda item: stable_hash(item, seed=int(args.seed)))

    reference_cache: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []
    planned_signatures: set[str] = set()
    rejections: list[dict[str, object]] = []

    def build_specs(context: dict[str, object], bucket: str, rank: int) -> list[dict[str, object]]:
        reference = _reference_for(
            reference_cache=reference_cache,
            reference_bank=reference_bank,
            event_id=str(context["event_id"]),
            action_ids=action_ids,
            start_min=float(context["start_min"]),
        )
        if bucket == "fit_deployment":
            tier1 = _tier1_profiles(
                legacy_groups=legacy_groups,
                phase=str(context["phase"]),
                context_index=rank,
            )
            specs = build_v8_deployment_residual_specifications(
                action_ids=action_ids,
                reference_action_seq=reference,
                tier1_profiles=tier1,
                phase=str(context["phase"]),
                magnitudes=(0.05, 0.10, 0.20),
            )
            specs.extend([
                spec for spec in build_v8_boundary_specifications(phase=str(context["phase"]), action_ids=action_ids)
                if spec.get("online_candidate_eligible") is True
            ])
            return specs
        return [
            spec for spec in build_v8_boundary_specifications(phase=str(context["phase"]), action_ids=action_ids)
            if spec.get("online_candidate_eligible") is False
        ]

    def append_planned_case(context: dict[str, object], specification: dict[str, object], bucket: str, rank: int) -> bool:
        reference = _reference_for(
            reference_cache=reference_cache,
            reference_bank=reference_bank,
            event_id=str(context["event_id"]),
            action_ids=action_ids,
            start_min=float(context["start_min"]),
        )
        candidate = materialize_candidate(reference, action_ids=action_ids, specification=specification)
        diagnostics = sequence_diagnostics(
            candidate,
            reference,
            action_ids=action_ids,
            binary_pump_ids=set(BINARY_RESIDUAL_PUMPS_V8),
            minimum_effective_delta=float(args.minimum_effective_delta),
        )
        if not diagnostics["valid"] or int(diagnostics["max_simultaneous_changes"]) > int(args.max_simultaneous_changes):
            rejections.append({
                "event_id": context["event_id"],
                "phase": context["phase"],
                "bucket": bucket,
                "mode": specification.get("mode", "unknown"),
                "reason": diagnostics.get("reason", "max_simultaneous_changes"),
                "max_simultaneous_changes": diagnostics.get("max_simultaneous_changes"),
            })
            return False
        signature = stable_hash({
            "event": context["event_id"],
            "checkpoint": f"{context['event_id']}|{context['phase']}|{float(context['start_min']):.3f}",
            "candidate": np.round(candidate, 6).astype(float).tolist(),
        })
        if signature in planned_signatures:
            return False
        planned_signatures.add(signature)
        manifest_rows.extend(_case_rows(
            event_id=str(context["event_id"]),
            role=str(context["role"]),
            split=str(context["split"]),
            phase=str(context["phase"]),
            start_min=float(context["start_min"]),
            duration_min=float(context["duration_min"]),
            reference=reference,
            candidate=candidate,
            specification=specification,
            diagnostics=diagnostics,
            canonical_hash=canonical_hash,
            semantics_hash=semantics_hash,
            plan_bucket=bucket,
            selection_rank=rank,
        ))
        return True

    bucket_roles = {
        "fit_deployment": "fit",
        "fit_boundary": "fit",
        "calibration_boundary": "calibration",
        "locked_validation_boundary": "locked_validation",
    }
    for bucket, target in budget.items():
        role = bucket_roles[bucket]
        role_contexts = contexts[role]
        if not role_contexts:
            raise ValueError(f"no contexts available for {bucket}")
        accepted = 0
        opportunities: list[tuple[str, dict[str, object], dict[str, object]]] = []
        for context_index, context in enumerate(role_contexts):
            for specification in build_specs(context, bucket, context_index):
                opportunities.append((
                    stable_hash({
                        "bucket": bucket,
                        "context": context,
                        "spec_mode": specification.get("mode", "unknown"),
                    }, seed=int(args.seed)),
                    context,
                    specification,
                ))
        opportunities.sort(key=lambda item: item[0])
        for rank, (_, context, spec) in enumerate(opportunities):
            if accepted >= int(target):
                break
            if append_planned_case(context, spec, bucket, rank):
                accepted += 1
        if accepted < int(target):
            raise RuntimeError(
                f"planned only {accepted}/{target} cases for {bucket}; "
                f"available_opportunities={len(opportunities)}; see preflight rejections"
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = ensure_dir(out / "paired_plan") / "tier2_residual_v8_expansion_manifest.csv"
    _write_locked_csv(manifest_path, manifest)
    pd.DataFrame(rejections).to_csv(out / "paired_plan" / "preflight_rejections.csv", index=False)
    candidates = manifest[manifest["branch"].astype(str).eq("B")].copy()
    preflight = summarize_v8_manifest_preflight(
        manifest,
        target_cases=int(args.target_cases),
        min_locked_validation_cases=int(args.min_locked_validation_cases),
        min_locked_validation_events=int(args.min_locked_validation_events),
    )
    preflight.update({
        "base_dataset": str(base_path.resolve()),
        "base_rows": int(len(data["event_ids"])),
        "base_events": int(len(set(data["event_ids"].astype(str)))),
        "base_dataset_sha256": file_sha256(base_path),
        "case_budget": budget,
        "event_roles": roles.groupby("role")["event_id"].nunique().astype(int).to_dict(),
        "candidate_modes": candidates["candidate_action_sequence"].map(lambda value: json.loads(value)["mode"]).value_counts().to_dict(),
        "planned_return_periods": sorted(candidates["return_period"].astype(str).unique().tolist()),
        "planned_rain_patterns": sorted(candidates["rain_pattern"].astype(str).unique().tolist()),
        "manifest": str(manifest_path.resolve()),
        "preflight_rejections": int(len(rejections)),
        "swmm_started": False,
        "label_note": (
            "Boundary rows are planned capacity for PFV/peak unsafe labels; actual unsafe counts must be confirmed "
            "after same-state SWMM execution and dataset audit."
        ),
    })
    _write_locked_json(out / "paired_plan" / "tier2_residual_v8_expansion_preflight.json", preflight)
    print(json.dumps(preflight, indent=2))
    if not preflight["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
