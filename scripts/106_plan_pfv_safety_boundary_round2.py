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
from sewerrtc.experiments.safety_boundary_plan import build_boundary_round2_slots
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
        default="outputs/project6_36_causal_effect_coverage_v2/effect_dataset_boundary_v5/same_state_raw_joint_36_causal_effect_coverage_boundary_v5.npz",
    )
    parser.add_argument(
        "--audit",
        default="outputs/project6_36_causal_effect_coverage_v2/effect_dataset_audit_boundary_v5/paired_information_audit.csv",
    )
    parser.add_argument(
        "--selection-table",
        default="outputs/project6_36_causal_effect_coverage_v2/safety_boundary_plan/selected_events_by_no_control_load.csv",
    )
    parser.add_argument("--reference-bank", default="outputs/data_bank_train_v8_storage_variablepump/trajectories")
    parser.add_argument("--out-dir", default="outputs/project6_36_causal_effect_coverage_v2/safety_boundary_round2_plan")
    parser.add_argument("--target-validation-unsafe-rows", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")

    def rooted(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    dataset_path = rooted(args.dataset)
    audit_path = rooted(args.audit)
    selection_path = rooted(args.selection_table)
    reference_bank = rooted(args.reference_bank)
    out = ensure_dir(rooted(args.out_dir))
    data = np.load(dataset_path, allow_pickle=True)
    action_ids = data["action_ids"].astype(str).tolist()
    if len(action_ids) != 36 or tuple(data["candidate_action_seq"].shape[1:]) != (6, 36):
        raise ValueError("round-2 planning requires strict canonical [N,6,36] data")
    selection = pd.read_csv(selection_path)
    selected = {
        split: selection.loc[selection["split"].astype(str).eq(split), "event_id"].astype(str).tolist()
        for split in ("train", "validation")
    }
    if any(len(selected[split]) != 8 for split in selected):
        raise ValueError(f"round-2 requires the frozen 8+8 event selection, got {selected}")
    slots = build_boundary_round2_slots(selected, recession_offsets_min=(15.0, 60.0))
    if len(slots) != 32:
        raise ValueError(f"round-2 allocation mismatch: {len(slots)}/32")

    rainfall = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    audit_table = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    canonical_hash = _hash(action_ids)
    semantics_hash = _file_hash(audit_table)
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
    specification = next(
        item
        for item in build_boundary_v5_specifications("recession")
        if item["mode"] == "outlet_regulators_close_hold"
    )
    reference_cache: dict[str, pd.DataFrame] = {}
    planned_signatures: set[str] = set()
    rows: list[dict[str, object]] = []
    for rank, slot in enumerate(slots):
        event_id = str(slot["event_id"])
        split = str(slot["split"])
        offset_min = float(slot["recession_offset_min"])
        event = rainfall.loc[event_id]
        start_min = float(event.duration_min) + offset_min
        if event_id not in reference_cache:
            detail_path = reference_bank / f"{event_id}__no_control_detail.csv"
            if not detail_path.exists():
                raise FileNotFoundError(f"missing No-control reference detail: {detail_path}")
            reference_cache[event_id] = pd.read_csv(detail_path)
        reference = action_window(
            reference_cache[event_id], action_ids=action_ids, start_min=start_min, horizon_steps=6
        )
        candidate = materialize_candidate(reference, action_ids=action_ids, specification=specification)
        diagnostics = sequence_diagnostics(
            candidate,
            reference,
            action_ids=action_ids,
            binary_pump_ids={"ADD301.2", "ADD301.3"},
            minimum_effective_delta=0.05,
        )
        if not diagnostics["valid"] or int(diagnostics["max_simultaneous_changes"]) > 8:
            raise ValueError(f"invalid round-2 candidate {event_id}/{offset_min}: {diagnostics}")
        checkpoint_id = f"{event_id}|recession|{start_min:.1f}"
        signature = _hash({
            "event": event_id,
            "checkpoint": checkpoint_id,
            "candidate": np.round(candidate, 6).astype(float).tolist(),
        })
        if signature in existing_signatures:
            raise ValueError(f"round-2 case duplicates existing data: {event_id}/{checkpoint_id}")
        if signature in planned_signatures:
            raise ValueError(f"duplicate round-2 case: {event_id}/{checkpoint_id}")
        planned_signatures.add(signature)
        executed_specification = {
            **specification,
            "family": "pfv_safety_boundary_v6_timing",
            "mode": f"outlet_regulators_close_hold_recession_plus_{int(offset_min)}m",
            "round2_selection_basis": "v5 aggregate mode-level response; formal events remain untouched",
        }
        schema = "pfv_safety_boundary_v6_round2_same_state"
        pair_id = _hash({"checkpoint": checkpoint_id, "specification": executed_specification, "schema": schema})
        reference_specification = {"mode": "default_no_control", "horizon_steps": 6}
        common = {
            "pair_id": pair_id,
            "event_id": event_id,
            "phase": "recession",
            "split": split,
            "checkpoint_id": checkpoint_id,
            "override_start_min": start_min,
            "split_timestamp_fraction": start_min / float(event.duration_min),
            "reference_policy": "no_control",
            "candidate_action_sequence": json.dumps(executed_specification, sort_keys=True),
            "canonical_action_order_hash": canonical_hash,
            "actuator_semantics_hash": semantics_hash,
            "code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "requires_same_state_branching": True,
            "candidate_kind": executed_specification["kind"],
            "candidate_family": executed_specification["family"],
            "intended_evidence_role": executed_specification["intended_evidence_role"],
            "online_candidate_eligible": False,
            "selection_rank": rank,
            "selection_required_family": "pfv_safety_boundary_timing",
            "recession_offset_min": offset_min,
            "materialized_reference_action_sequence": json.dumps(reference.astype(float).tolist()),
            "materialized_candidate_action_sequence": json.dumps(candidate.astype(float).tolist()),
            **{key: value for key, value in diagnostics.items() if key not in {"actual_delta_after_clipping", "changed_actuator_ids"}},
            "changed_actuator_ids": ",".join(diagnostics["changed_actuator_ids"]),
            "actual_delta_after_clipping": json.dumps(diagnostics["actual_delta_after_clipping"], sort_keys=True),
            "return_period": event_return_period(event_id),
            "rain_pattern": event_pattern(event_id),
        }
        for branch, executed in (("A", reference_specification), ("B", executed_specification)):
            payload = {
                "event": event_id,
                "checkpoint": checkpoint_id,
                "branch": branch,
                "executed": executed,
                "schema": schema,
            }
            rows.append({
                **common,
                "case_id": _hash({"pair": pair_id, "branch": branch}),
                "execution_case_id": _hash(payload),
                "branch": branch,
                "executed_action_sequence": json.dumps(executed, sort_keys=True),
                "status": "reference_reused" if branch == "A" else "preflight_validated_not_started",
            })

    manifest = pd.DataFrame(rows)
    candidates = manifest[manifest["branch"].astype(str).eq("B")].copy()
    audit = pd.read_csv(audit_path)
    current_validation_unsafe = int(
        ((audit["split"].astype(str) == "validation") & (audit["PFV_noninferiority"].astype(str) == "unsafe")).sum()
    )
    required_new_unsafe = max(0, int(args.target_validation_unsafe_rows) - current_validation_unsafe)
    validation_cases = int((candidates["split"] == "validation").sum())
    checks = {
        "exact_candidate_budget_32": len(candidates) == 32,
        "train_cases_16": int((candidates["split"] == "train").sum()) == 16,
        "validation_cases_16": validation_cases == 16,
        "no_noops": not bool(candidates["is_noop"].astype(bool).any()),
        "same_state_reference": candidates["checkpoint_id"].notna().all(),
        "event_group_split_disjoint": not bool(
            set(candidates.loc[candidates["split"].eq("train"), "event_id"])
            & set(candidates.loc[candidates["split"].eq("validation"), "event_id"])
        ),
        "canonical_shape_H36": candidates["materialized_candidate_action_sequence"].map(
            lambda value: np.asarray(json.loads(value)).shape == (6, 36)
        ).all(),
        "offline_only": not bool(candidates["online_candidate_eligible"].astype(bool).any()),
        "capacity_to_reach_target": validation_cases >= required_new_unsafe,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    manifest_path = out / "pfv_safety_boundary_v6_round2_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    report = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_cases": int(len(candidates)),
        "train_candidate_cases": int((candidates["split"] == "train").sum()),
        "validation_candidate_cases": validation_cases,
        "reference_checkpoints_reused": int(candidates["checkpoint_id"].nunique()),
        "current_validation_PFV_unsafe_rows": current_validation_unsafe,
        "target_validation_PFV_unsafe_rows": int(args.target_validation_unsafe_rows),
        "required_new_validation_unsafe_rows": required_new_unsafe,
        "required_validation_unsafe_hit_rate": float(required_new_unsafe / validation_cases),
        "recession_offsets_min": [15.0, 60.0],
        "candidate_mode": "outlet_regulators_close_hold",
        "manifest": str(manifest_path),
        "selection_policy": (
            "Adaptive development supplement: retain the only v5 mode with cross-event unsafe consistency; "
            "test two predeclared new recession checkpoints; never use these events as final formal evidence."
        ),
        "swmm_started": False,
    }
    (out / "pfv_safety_boundary_v6_round2_preflight.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("PFV safety-boundary round-2 preflight failed")


if __name__ == "__main__":
    main()
