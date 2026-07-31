"""Dataset builders for Pilot400 and Train1600 sample/branch manifests.

A sample row is one ``event + checkpoint + unique actual candidate
schedule``; a branch row is one ``sample_id + branch_role``.  Reference
branches (no_control / dynamic_internal_rules / hold_previous) provide
evidence for labels but are never counted as accepted samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .manifests import accounting_summary
from .reference_cache import REFERENCE_BRANCHES


CANDIDATE_BRANCH = "candidate"

BRANCH_ROLES = (
    CANDIDATE_BRANCH,
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)

IDENTITY_COLUMNS = (
    "sample_id",
    "event_id",
    "rainfall_sha256",
    "checkpoint_id",
    "checkpoint_state_sha256",
    "split",
    "event_tier",
    "candidate_family",
    "actual_schedule_sha256",
)

CONTINUOUS_LABELS = (
    "delta_pfv_h120_vs_no_control",
    "delta_tfv_h120_vs_dynamic_internal",
    "delta_peak_h120_vs_dynamic_internal",
)

TEMPORAL_RESIDUALS = (
    "priority_depth_residual",
    "sentinel_depth_residual",
    "active_link_flow_residual",
    "storage_volume_residual",
    "tfv_rate_residual",
    "system_stored_volume_residual",
    "conduit_fullness_summary_residual",
)

CLASSIFICATION_LABELS = (
    "pfv_safe",
    "tfv_improved",
    "peak_noninferior",
    "joint_noninferior",
    "materially_beneficial",
    "neutral",
    "flat_state",
    "action_authority_class",
    "hard_negative_type",
)

RANKING_LABELS = (
    "feasible_rank",
    "pairwise_preference",
    "best_feasible_candidate",
    "regret_to_exact_best",
)

ELIGIBILITY_MASKS = (
    "h120_eligible",
    "full_event_eligible",
    "label_validity_pfv",
    "label_validity_tfv",
    "label_validity_peak",
    "label_validity_full",
    "recovery_class",
    "censored",
)


def feature_schema() -> dict:
    """Input feature contract for every accepted sample (spec section 10)."""
    return {
        "identity": list(IDENTITY_COLUMNS),
        "state_history": {
            "frames": 13,
            "fields": ["state_history_paths", "state_history_sha256"],
        },
        "rainfall_forecast": {
            "steps": 12,
            "fields": ["rainfall_forecast_json"],
        },
        "actions": {
            "steps": 12,
            "facilities": 36,
            "fields": [
                "actual_action_matrix_json",
                "readback_action_matrix_json",
                "candidate_minus_dynamic_internal_json",
                "candidate_minus_hold_json",
                "k_sequence_json",
                "switch_ramp_dwell_summary_json",
            ],
        },
        "static_graph": {
            "fields": ["static_graph_sha256", "static_schema_sha256"],
        },
        "branch_evidence": {
            "branches": list(BRANCH_ROLES),
            "fields": [
                "candidate_trajectory_path",
                "no_control_trajectory_path",
                "dynamic_internal_trajectory_path",
                "hold_previous_trajectory_path",
                "network_sha256",
                "rainfall_sha256",
                "prefix_sha256",
                "state_hash_match",
                "readback_ok",
            ],
        },
    }


def label_schema() -> dict:
    """Label contract; full-event labels without eligibility must be NaN."""
    return {
        "continuous": list(CONTINUOUS_LABELS),
        "temporal_12step": list(TEMPORAL_RESIDUALS),
        "classification": list(CLASSIFICATION_LABELS),
        "ranking": list(RANKING_LABELS),
        "eligibility_masks": list(ELIGIBILITY_MASKS),
        "nan_policy": "full-event labels are NaN unless full_event_eligible",
    }


def build_branch_manifest(branch_records: pd.DataFrame) -> pd.DataFrame:
    """One row per ``sample_id + branch_role`` with validated roles."""
    required = {"sample_id", "event_id", "checkpoint_id", "branch_role"}
    missing = required - set(branch_records)
    if missing:
        raise ValueError(f"branch records missing: {sorted(missing)}")
    unknown = set(branch_records["branch_role"].astype(str)) - set(BRANCH_ROLES)
    if unknown:
        raise ValueError(f"unknown branch roles: {sorted(unknown)}")
    if branch_records.duplicated(["sample_id", "branch_role"]).any():
        raise ValueError("duplicate sample_id + branch_role rows")
    manifest = branch_records.copy()
    manifest["is_reference_branch"] = ~manifest["branch_role"].eq(
        CANDIDATE_BRANCH
    )
    return manifest.reset_index(drop=True)


def build_sample_manifest(
    branch_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse branch evidence into accepted / duplicate / missing samples.

    Returns ``(samples, actual_duplicates, missing)``.  A sample is accepted
    only when its candidate branch and all three reference branches exist;
    reference branches contribute evidence but never count as samples.
    """
    required = {
        "sample_id",
        "event_id",
        "checkpoint_id",
        "branch_role",
        "actual_schedule_sha256",
    }
    missing_columns = required - set(branch_manifest)
    if missing_columns:
        raise ValueError(
            f"branch manifest missing: {sorted(missing_columns)}"
        )
    candidates = branch_manifest[
        branch_manifest["branch_role"] == CANDIDATE_BRANCH
    ].copy()
    if "is_noop" in candidates:
        noop_mask = candidates["is_noop"].fillna(False).astype(bool)
        noops = candidates[noop_mask].copy()
        if len(noops):
            noops["rejection_reason"] = "no_op_not_accepted"
        candidates = candidates[~noop_mask].copy()
    else:
        noops = candidates.head(0).copy()
    duplicate_mask = candidates.duplicated(
        ["event_id", "checkpoint_id", "actual_schedule_sha256"], keep="first"
    )
    duplicates = candidates[duplicate_mask].copy()
    if len(duplicates):
        duplicates["rejection_reason"] = "duplicate_actual_schedule"
    if len(noops):
        duplicates = pd.concat([duplicates, noops], ignore_index=True)
    unique_candidates = candidates[~duplicate_mask].copy()
    reference_roles = {
        "no_control",
        "dynamic_internal_rules",
        "hold_previous",
    }
    present = (
        branch_manifest[branch_manifest["branch_role"].isin(reference_roles)]
        .groupby("sample_id")["branch_role"]
        .apply(set)
    )
    complete = unique_candidates["sample_id"].map(
        lambda sample_id: present.get(sample_id, set()) == reference_roles
    )
    samples = unique_candidates[complete].copy()
    incomplete = unique_candidates[~complete].copy()
    if len(incomplete):
        incomplete["missing_reason"] = "reference_branch_incomplete"
    samples["reference_branches_counted_as_samples"] = 0
    return (
        samples.reset_index(drop=True),
        duplicates.reset_index(drop=True),
        incomplete.reset_index(drop=True),
    )


def dataset_accounting(
    planned: int,
    samples: pd.DataFrame,
    rejected: pd.DataFrame,
    pending: pd.DataFrame,
    missing: pd.DataFrame,
) -> dict:
    return accounting_summary(
        planned,
        accepted=len(samples),
        rejected=len(rejected),
        pending=len(pending),
        missing=len(missing),
    )


def enforce_nan_for_ineligible(frame: pd.DataFrame) -> pd.DataFrame:
    """Full-event label columns must be NaN when not full_event_eligible."""
    if "full_event_eligible" not in frame:
        raise ValueError("full_event_eligible mask is required")
    result = frame.copy()
    full_columns = [
        column
        for column in result
        if column.endswith("_full") or "_full_event" in column
    ]
    ineligible = ~result["full_event_eligible"].astype(bool)
    if full_columns:
        result.loc[ineligible, full_columns] = np.nan
    return result


def audit_train1600_dataset(
    samples: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    accounting: dict,
) -> dict:
    """Train1600 quality gate (spec section 11) on the sample manifest."""
    required = {
        "sample_id",
        "event_id",
        "rainfall_sha256",
        "checkpoint_id",
        "actual_schedule_sha256",
        "split",
    }
    missing = required - set(samples)
    if missing:
        return {"status": "blocked", "missing_columns": sorted(missing)}
    state_groups = samples.groupby(["event_id", "checkpoint_id"])
    per_state = state_groups["actual_schedule_sha256"].nunique()
    split_events = {
        split: set(
            samples[samples["split"] == split]["event_id"].astype(str)
        )
        for split in ("train", "calibration", "locked_validation")
    }
    overlap = (
        (split_events["train"] & split_events["calibration"])
        | (split_events["train"] & split_events["locked_validation"])
        | (split_events["calibration"] & split_events["locked_validation"])
    )
    shortfall_states = per_state[per_state < 5]
    checks = {
        "accepted_1600": len(samples) == 1600,
        "events_64": samples["event_id"].nunique() == 64,
        "state_groups_320": state_groups.ngroups == 320,
        "five_actual_unique_per_state_or_recorded": bool(
            per_state.le(5).all()
        ),
        "no_split_event_overlap": not overlap,
        "split_counts_48_8_8": {
            split: len(split_events[split])
            for split in split_events
        }
        == {"train": 48, "calibration": 8, "locked_validation": 8},
        "reserve_not_in_training": not samples["split"].eq("reserve").any(),
        "no_actual_duplicates": not samples.duplicated(
            ["event_id", "checkpoint_id", "actual_schedule_sha256"]
        ).any(),
        "rainfall_sha_split_isolated": not samples.groupby(
            "rainfall_sha256"
        )["split"].nunique().gt(1).any(),
        "accounting_closed": bool(accounting.get("accounting_closed", False)),
    }
    # JSON gate fields must be native Python bool, never numpy.bool_.
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "checks": checks,
        "states_below_five_candidates": {
            f"{event}|{checkpoint}": int(count)
            for (event, checkpoint), count in shortfall_states.items()
        },
        "accounting": accounting,
    }
