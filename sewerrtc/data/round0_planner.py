from __future__ import annotations

from pathlib import Path
from typing import Any

from sewerrtc.contracts.prompt3a import OUT_ROOT, read_csv, utc_now, write_csv, write_json
from sewerrtc.data.candidate_prefilter import prefilter_candidate


def plan_round0(event_catalog_path: str | Path, checkpoint_catalog_path: str | Path, out_dir: str | Path = OUT_ROOT / "round0") -> tuple[int, dict[str, Any], list[Path]]:
    out_dir = Path(out_dir)
    events = [row for row in read_csv(Path(event_catalog_path)) if row.get("round0_eligible") == "true"]
    checkpoints = read_csv(Path(checkpoint_catalog_path))
    planned: list[dict[str, Any]] = []
    for cp in checkpoints[:300]:
        event_id = cp.get("event_id", "")
        if not any(row.get("event_id") == event_id for row in events):
            continue
        for concurrency in ["0", "1-2", "3-4", "5-8"]:
            candidate = {
                "case_id": f"round0_{len(planned):06d}",
                "event_id": event_id,
                "checkpoint_id": cp.get("checkpoint_id", ""),
                "storm_family_id": cp.get("storm_family_id", ""),
                "split": cp.get("split", "action_effect_fit"),
                "state_clone_hash": cp.get("state_clone_hash", ""),
                "controller_memory_hash": cp.get("controller_memory_hash", ""),
                "forecast_scenario_id": "operational_nominal",
                "anchor_type": "selected_safe_fallback",
                "selected_fallback": cp.get("selected_fallback", ""),
                "facility_ids": "",
                "hydraulic_group": "",
                "requested_action_ref": "planned_12x36",
                "projected_action_ref": "planned_12x36",
                "expected_actual_action_ref": "planned_12x36",
                "override_mask_ref": "planned_12x36",
                "duration_steps": 12,
                "free_residual_steps": 3,
                "concurrency": concurrency,
                "ttl": 1,
                "continuation_policy_id": "fixed_anchor_continuation_after_30min",
                "coverage_cell_id": f"{event_id}:{cp.get('phase','')}:{concurrency}",
                "sampling_reason": "coverage_gap",
                "support_status": "planned",
                "ood_status": "not_evaluated",
                "override_count": 0 if concurrency == "0" else int(concurrency.split("-")[-1]),
                "add350_residual_override": False,
                "binary_legality": "pending",
                "noop": concurrency == "0",
                "duplicate": False,
            }
            ok, reason = prefilter_candidate(candidate)
            candidate["feasibility"] = "planned" if ok else "excluded"
            candidate["exclusion_reason"] = reason
            planned.append(candidate)
            if len([row for row in planned if row["feasibility"] == "planned"]) >= 2000:
                break
        if len([row for row in planned if row["feasibility"] == "planned"]) >= 2000:
            break
    planned_effective = [row for row in planned if row["feasibility"] == "planned"]
    files = [
        write_csv(out_dir / "paired_manifest_round0.csv", planned),
        write_csv(out_dir / "checkpoint_coverage_round0.csv", [{"checkpoint_id": row.get("checkpoint_id", ""), "case_id": row["case_id"], "coverage_cell_id": row["coverage_cell_id"]} for row in planned]),
        write_csv(out_dir / "preflight_noop_audit_round0.csv", [row for row in planned if row.get("noop") is True]),
        write_csv(out_dir / "planned_facility_support_round0.csv", []),
        write_csv(out_dir / "planned_phase_support_round0.csv", []),
        write_csv(out_dir / "planned_concurrency_support_round0.csv", [{"concurrency": c, "planned_count": sum(1 for row in planned if row["concurrency"] == c)} for c in ["0", "1-2", "3-4", "5-8"]]),
        write_csv(out_dir / "planned_interaction_support_round0.csv", []),
        write_csv(out_dir / "structural_infeasible_cells.csv", [row for row in planned if row["feasibility"] == "excluded"]),
    ]
    report = {"status": "completed" if planned_effective else "blocked", "created_at": utc_now(), "planned_candidate_count": len(planned), "effective_candidate_count": len(planned_effective), "target_effective_range": "1500-2000", "full_round0_unlock_allowed": False}
    files.append(write_json(out_dir / "round0_plan_report.json", report))
    return (0 if planned_effective else 3), report, files

