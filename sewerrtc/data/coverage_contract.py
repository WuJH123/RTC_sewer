"""Information-coverage helpers for Project6 V3 candidate planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CoverageTarget:
    name: str
    minimum: int
    current: int = 0

    @property
    def status(self) -> str:
        if self.minimum <= 0:
            return "structural_infeasible"
        if self.current >= self.minimum:
            return "sufficient"
        if self.current == 0:
            return "missing"
        return "insufficient_support"


def classify_coverage(current: Mapping[str, int], minimums: Mapping[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, minimum in minimums.items():
        target = CoverageTarget(name=key, minimum=int(minimum), current=int(current.get(key, 0)))
        rows.append(
            {
                "coverage_key": target.name,
                "current": target.current,
                "minimum": target.minimum,
                "status": target.status,
                "gap": max(target.minimum - target.current, 0),
            }
        )
    return rows


def is_noop(actual_residual_linf: float, threshold: float = 1e-6) -> bool:
    return abs(float(actual_residual_linf)) <= float(threshold)


def candidate_kept_by_reason(reasons: Iterable[str]) -> bool:
    return bool(set(reasons) & SAMPLING_REASON_VALUES)


SAMPLING_REASON_VALUES = {
    "fill_coverage_gap",
    "increase_independent_event_support",
    "cover_facility_direction_phase_magnitude_duration",
    "calibrate_pfv_tfv_peak_boundary",
    "repair_false_safe",
    "validate_h30_h120_reversal",
    "validate_fallback",
    "validate_low_support_mpc_action",
    "validate_optimizer_exploitation",
    "active_learning_selected",
}


COVERAGE_CELL_FIELDS = (
    "event_id",
    "storm_family",
    "split",
    "checkpoint_id",
    "phase",
    "state_risk_cluster",
    "anchor_type",
    "facility_or_hydraulic_group",
    "direction",
    "magnitude",
    "duration_steps",
    "concurrency",
    "interaction_type",
    "unique_event_support",
    "feasibility_status",
    "outcome_class",
    "decision_relevance",
)


CANDIDATE_MANIFEST_FIELDS = (
    "case_id",
    "event_id",
    "storm_family",
    "split",
    "checkpoint_id",
    "phase",
    "forecast_scenario_id",
    "state_clone_hash",
    "coverage_cell_id",
    "anchor_type",
    "hydraulic_group_id",
    "facility_ids",
    "direction",
    "magnitude",
    "duration_steps",
    "concurrency",
    "interaction_type",
    "override_count",
    "binary_switch_count",
    "requested_action_ref",
    "projected_action_ref",
    "expected_actual_action_ref",
    "dwell_precheck",
    "interlock_precheck",
    "rate_limit_precheck",
    "support_status",
    "ood_status",
    "sampling_reason",
    "branch_definitions",
    "continuation_policy_id",
    "tail_policy_id",
    "pre_run_status",
)


def validate_coverage_cell(row: Mapping[str, object]) -> list[str]:
    missing = [field for field in COVERAGE_CELL_FIELDS if field not in row]
    errors = [f"missing:{field}" for field in missing]
    status = str(row.get("feasibility_status", ""))
    if status and status not in {"missing", "sufficient", "structural_infeasible", "insufficient_support", "feasible"}:
        errors.append(f"invalid_feasibility_status:{status}")
    return errors


def validate_candidate_manifest_row(row: Mapping[str, object]) -> list[str]:
    missing = [field for field in CANDIDATE_MANIFEST_FIELDS if field not in row]
    errors = [f"missing:{field}" for field in missing]
    if str(row.get("pre_run_status", "")) in {"noop", "illegal", "near_duplicate"}:
        errors.append(f"filtered_candidate_should_not_be_scheduled:{row.get('pre_run_status')}")
    sampling_reason = str(row.get("sampling_reason", ""))
    if sampling_reason and sampling_reason not in SAMPLING_REASON_VALUES:
        errors.append(f"invalid_sampling_reason:{sampling_reason}")
    return errors
