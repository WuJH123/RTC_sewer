from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.experiments.informative_effect_supplement import build_boundary_v5_specifications
from sewerrtc.experiments.safety_boundary_plan import build_boundary_case_slots, select_events_by_reference_load
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_causal_effect_coverage_v2.yaml")
    parser.add_argument(
        "--dataset",
        default="outputs/project6_36_causal_effect_coverage_v2/effect_dataset/same_state_raw_joint_36_causal_effect_coverage_v2.npz",
    )
    parser.add_argument(
        "--audit",
        default="outputs/project6_36_causal_effect_coverage_v2/effect_dataset_audit/paired_information_audit.csv",
    )
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--out-dir", default="outputs/project6_36_causal_effect_coverage_v2/safety_boundary_plan")
    parser.add_argument("--train-events", type=int, default=8)
    parser.add_argument("--validation-events", type=int, default=8)
    parser.add_argument("--target-validation-unsafe-rows", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    dataset_path = root / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    audit_path = root / args.audit if not Path(args.audit).is_absolute() else Path(args.audit)
    reference_bank = root / args.reference_bank if not Path(args.reference_bank).is_absolute() else Path(args.reference_bank)
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    data = np.load(dataset_path, allow_pickle=True)
    action_ids = data["action_ids"].astype(str).tolist()
    if len(action_ids) != 36 or tuple(data["candidate_action_seq"].shape[1:]) != (6, 36):
        raise ValueError("safety-boundary planning requires strict canonical [N,6,36] data")
    selected = select_events_by_reference_load(
        event_ids=data["event_ids"],
        splits=data["split"],
        reference_risk_rate_seq=data["reference_risk_rate_seq"],
        train_count=int(args.train_events),
        validation_count=int(args.validation_events),
        dt_sec=int(cfg["experiment"]["control_step_sec"]),
    )
    slots = build_boundary_case_slots(selected)
    expected_cases = int(args.train_events) * 4 + int(args.validation_events) * 5
    if len(slots) != expected_cases:
        raise ValueError(f"boundary allocation mismatch: {len(slots)}/{expected_cases}")

    rainfall = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    audit_table = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    canonical_hash = _hash(action_ids)
    semantics_hash = _file_hash(audit_table)
    binary_pumps = {"ADD301.2", "ADD301.3"}
    existing_signatures = {
        _hash({
            "event": str(event_id),
            "checkpoint": str(checkpoint_id),
            "candidate": np.round(candidate, 6).astype(float).tolist(),
        })
        for event_id, checkpoint_id, candidate in zip(
            data["event_ids"].astype(str),
            data["checkpoint_id"].astype(str),
            data["candidate_action_seq"],
        )
    }
    reference_cache: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []
    planned_signatures: set[str] = set()
    for selection_rank, slot in enumerate(slots):
        event_id = str(slot["event_id"])
        split = str(slot["split"])
        phase = str(slot["phase"])
        event = rainfall.loc[event_id]
        start_min = (
            float(event.duration_min) * 0.55
            if phase == "peak"
            else float(event.duration_min) + 30.0
        )
        if event_id not in reference_cache:
            detail_path = reference_bank / f"{event_id}__no_control_detail.csv"
            if not detail_path.exists():
                raise FileNotFoundError(f"missing No-control reference detail: {detail_path}")
            reference_cache[event_id] = pd.read_csv(detail_path)
        reference = action_window(
            reference_cache[event_id],
            action_ids=action_ids,
            start_min=start_min,
            horizon_steps=6,
        )
        specifications = build_boundary_v5_specifications(phase)
        specification = specifications[int(slot["specification_index"])]
        if specification.get("online_candidate_eligible") is not False:
            raise ValueError("safety-boundary cases must be explicitly ineligible for online execution")
        candidate = materialize_candidate(reference, action_ids=action_ids, specification=specification)
        diagnostics = sequence_diagnostics(
            candidate,
            reference,
            action_ids=action_ids,
            binary_pump_ids=binary_pumps,
            minimum_effective_delta=0.05,
        )
        if not diagnostics["valid"] or int(diagnostics["max_simultaneous_changes"]) > 8:
            raise ValueError(f"invalid boundary candidate {event_id}/{phase}: {diagnostics}")
        checkpoint_id = f"{event_id}|{phase}|{start_min:.1f}"
        signature = _hash({
            "event": event_id,
            "checkpoint": checkpoint_id,
            "candidate": np.round(candidate, 6).astype(float).tolist(),
        })
        if signature in existing_signatures:
            raise ValueError(f"planned boundary case duplicates existing data: {event_id}/{phase}/{specification['mode']}")
        if signature in planned_signatures:
            raise ValueError(f"duplicate planned boundary case: {event_id}/{phase}/{specification['mode']}")
        planned_signatures.add(signature)
        schema = "pfv_safety_boundary_v5_same_state"
        pair_id = _hash({"checkpoint": checkpoint_id, "specification": specification, "schema": schema})
        reference_specification = {"mode": "default_no_control", "horizon_steps": 6}
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
            "online_candidate_eligible": False,
            "selection_rank": selection_rank,
            "selection_required_family": "pfv_safety_boundary",
            "reference_selection_basis": "No-control horizon PFV load only; candidate labels were not used",
            "materialized_reference_action_sequence": json.dumps(reference.astype(float).tolist()),
            "materialized_candidate_action_sequence": json.dumps(candidate.astype(float).tolist()),
            **{key: value for key, value in diagnostics.items() if key not in {"actual_delta_after_clipping", "changed_actuator_ids"}},
            "changed_actuator_ids": ",".join(diagnostics["changed_actuator_ids"]),
            "actual_delta_after_clipping": json.dumps(diagnostics["actual_delta_after_clipping"], sort_keys=True),
            "return_period": event_return_period(event_id),
            "rain_pattern": event_pattern(event_id),
        }
        for branch, executed in (("A", reference_specification), ("B", specification)):
            execution_payload = {
                "event": event_id,
                "checkpoint": checkpoint_id,
                "branch": branch,
                "executed": executed,
                "schema": schema,
            }
            manifest_rows.append({
                **common,
                "case_id": _hash({"pair": pair_id, "branch": branch}),
                "execution_case_id": _hash(execution_payload),
                "branch": branch,
                "executed_action_sequence": json.dumps(executed, sort_keys=True),
                "status": "reference_reused" if branch == "A" else "preflight_validated_not_started",
            })

    manifest = pd.DataFrame(manifest_rows)
    candidates = manifest[manifest["branch"].astype(str).eq("B")].copy()
    train_events = set(candidates.loc[candidates["split"].eq("train"), "event_id"])
    validation_events = set(candidates.loc[candidates["split"].eq("validation"), "event_id"])
    current_validation_unsafe = 0
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        current_validation_unsafe = int(
            ((audit["split"].astype(str) == "validation") & (audit["PFV_noninferiority"].astype(str) == "unsafe")).sum()
        )
    required_new_unsafe = max(0, int(args.target_validation_unsafe_rows) - current_validation_unsafe)
    planned_validation_cases = int((candidates["split"] == "validation").sum())
    checks = {
        "exact_candidate_budget": len(candidates) == expected_cases,
        "validation_candidate_cases_40": planned_validation_cases == int(args.validation_events) * 5,
        "train_candidate_cases_32": int((candidates["split"] == "train").sum()) == int(args.train_events) * 4,
        "no_noops": not bool(candidates["is_noop"].astype(bool).any()),
        "event_group_split_disjoint": not bool(train_events & validation_events),
        "same_checkpoint_reference": candidates["checkpoint_id"].notna().all(),
        "canonical_shape_H36": candidates["materialized_candidate_action_sequence"].map(
            lambda value: np.asarray(json.loads(value)).shape == (6, 36)
        ).all(),
        "simultaneous_limit_8": int(candidates["max_simultaneous_changes"].max()) <= 8,
        "offline_only": not bool(candidates["online_candidate_eligible"].astype(bool).any()),
        "capacity_to_reach_validation_unsafe_target": planned_validation_cases >= required_new_unsafe,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    manifest_path = out / "pfv_safety_boundary_v5_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    event_scores = []
    reference_pfv = data["reference_risk_rate_seq"][:, :, 0].sum(axis=1) * float(cfg["experiment"]["control_step_sec"])
    for split, event_list in selected.items():
        for event_id in event_list:
            selected_rows = data["event_ids"].astype(str) == event_id
            event_scores.append({
                "split": split,
                "event_id": event_id,
                "max_reference_horizon_PFV_m3": float(reference_pfv[selected_rows].max()),
                "return_period": event_return_period(event_id),
                "rain_pattern": event_pattern(event_id),
            })
    pd.DataFrame(event_scores).to_csv(out / "selected_events_by_no_control_load.csv", index=False)
    report = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_cases": int(len(candidates)),
        "train_candidate_cases": int((candidates["split"] == "train").sum()),
        "validation_candidate_cases": planned_validation_cases,
        "reference_checkpoints_reused": int(candidates["checkpoint_id"].nunique()),
        "current_validation_PFV_unsafe_rows": current_validation_unsafe,
        "target_validation_PFV_unsafe_rows": int(args.target_validation_unsafe_rows),
        "required_new_validation_unsafe_rows": required_new_unsafe,
        "required_validation_unsafe_hit_rate": (
            float(required_new_unsafe / planned_validation_cases) if planned_validation_cases else None
        ),
        "train_events": sorted(train_events),
        "validation_events": sorted(validation_events),
        "candidate_modes": candidates["candidate_action_sequence"].map(
            lambda value: json.loads(value)["mode"]
        ).value_counts().to_dict(),
        "manifest": str(manifest_path),
        "dataset": str(dataset_path),
        "selection_policy": "Frozen split; rank events only by No-control horizon PFV load; no candidate label selection.",
        "swmm_started": False,
    }
    (out / "pfv_safety_boundary_v5_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("PFV safety-boundary v5 preflight failed")


if __name__ == "__main__":
    main()
