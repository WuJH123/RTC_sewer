"""Incremental (Partial) dataset audits for progressive release.

Partial mode reads only cases that already carry a valid completion marker.
Cases that have not run yet are ``pending`` -- never ``missing`` and never
failures.  A Partial gate may stop further scale-up, but a Partial pass is
never a full-scope pass: ``scope_complete`` is always False in every partial
accounting record, and the formal complete manifests are never touched.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


PARTIAL_FILES = (
    "partial_sample_manifest.csv",
    "partial_branch_manifest.csv",
    "partial_rejected.csv",
    "partial_actual_duplicates.csv",
    "partial_quality_audit.json",
    "partial_completion.json",
)

# Hard execution-authenticity evidence. Every completed case must prove each
# of these; a missing column is missing evidence and fails closed.
HARD_AUTHENTICITY_COLUMNS = (
    "completion_valid",
    "four_branches_complete",
    "same_state_ok",
    "physical_sha_ok",
    "rainfall_sha_ok",
    "prefix_sha_ok",
    "action_stage_chain_complete",
    "no_hotstart",
    "k_le_8",
    "actuator_semantics_ok",
    "engineering_limits_ok",
    "h120_window_complete",
    "kpi_recompute_ok",
    "reference_cache_sha_ok",
)

DELTA_COLUMNS = (
    "delta_pfv_h120_vs_no_control",
    "delta_tfv_h120_vs_dynamic_internal",
    "delta_peak_h120_vs_dynamic_internal",
)

COVERAGE_COLUMNS = (
    "event_id",
    "checkpoint_id",
    "checkpoint_role",
    "rainfall_family",
    "rainfall_phase",
    "risk_level",
    "candidate_family",
    "K",
)

RELEASE_LEVEL_THRESHOLDS = {0: 1, 1: 16, 2: 40}


def _bool_col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name].fillna(False).astype(bool)
    return pd.Series(False, index=frame.index)


def _num_col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(float("nan"), index=frame.index)


def classify_partial_cases(
    plan: pd.DataFrame, completions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Split the plan into completed / failed / pending against markers.

    Pending cases have simply not run yet; they are never counted as missing
    and never as failures.  ``missing_confirmed`` only holds completed cases
    whose recorded detail artifact no longer exists on disk.
    """
    if "case_id" not in plan:
        raise ValueError("plan is missing case_id")
    plan_ids = plan["case_id"].astype(str)
    if plan_ids.duplicated().any():
        raise ValueError("plan has duplicate case_id rows")
    if completions is None or completions.empty or "case_id" not in completions:
        completed = plan.head(0).copy()
        failed = plan.head(0).copy()
        pending = plan.copy()
        missing = plan.head(0).copy()
        return {
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "missing_confirmed": missing,
        }
    comp = completions.copy()
    comp["case_id"] = comp["case_id"].astype(str)
    comp = comp[comp["case_id"].isin(set(plan_ids))]
    passed = comp[comp.get("status", "") == "pass"].copy()
    failed = comp[comp.get("status", "") == "failed"].copy()
    if "detail_path" in passed and len(passed):
        exists = passed["detail_path"].map(
            lambda value: bool(value) and Path(str(value)).exists()
        )
        missing = passed[~exists].copy()
        passed = passed[exists].copy()
    else:
        missing = passed.head(0).copy()
    terminal = set(passed["case_id"]) | set(failed["case_id"]) | set(
        missing["case_id"]
    )
    pending = plan[~plan_ids.isin(terminal)].copy()
    return {
        "completed": passed,
        "failed": failed,
        "pending": pending,
        "missing_confirmed": missing,
    }


def build_partial_bundle(
    plan: pd.DataFrame, completions: pd.DataFrame
) -> dict:
    """Build the partial sample/branch manifests from completed cases only.

    Acceptance rules mirror the formal builder: reference branches never
    count as samples, no-op candidates are never accepted, and only
    actual-unique schedules are accepted; hard-authenticity failures are
    rejected fail-closed (missing evidence counts as a violation).
    """
    parts = classify_partial_cases(plan, completions)
    completed = parts["completed"]
    branch_manifest = completed.copy()
    if "branch_role" not in branch_manifest:
        branch_manifest["branch_role"] = "candidate"
    branch_manifest["is_reference_branch"] = ~branch_manifest[
        "branch_role"
    ].astype(str).eq("candidate")
    candidates = branch_manifest[
        ~branch_manifest["is_reference_branch"]
    ].copy()

    hard_matrix = pd.DataFrame(
        {
            column: _bool_col(candidates, column)
            for column in HARD_AUTHENTICITY_COLUMNS
        },
        index=candidates.index,
    )
    authentic_mask = (
        hard_matrix.all(axis=1)
        if len(candidates)
        else pd.Series(dtype=bool)
    )
    hard_failed = candidates[~authentic_mask].copy()
    if len(hard_failed):
        hard_failed["rejection_reason"] = "hard_authenticity_failed"
    authentic = candidates[authentic_mask].copy()

    noop_mask = _bool_col(authentic, "is_noop")
    noops = authentic[noop_mask].copy()
    if len(noops):
        noops["rejection_reason"] = "no_op_not_accepted"
    informative = authentic[~noop_mask].copy()

    dup_keys = ["event_id", "checkpoint_id", "actual_schedule_sha256"]
    if len(informative) and all(key in informative for key in dup_keys):
        dup_mask = informative.duplicated(dup_keys, keep="first")
    else:
        dup_mask = pd.Series(False, index=informative.index)
    duplicates = informative[dup_mask].copy()
    if len(duplicates):
        duplicates["rejection_reason"] = "duplicate_actual_schedule"
    accepted = informative[~dup_mask].copy()

    runtime_failed = parts["failed"].copy()
    if len(runtime_failed):
        runtime_failed["rejection_reason"] = "runtime_failed"
    rejected = pd.concat(
        [hard_failed, noops, runtime_failed], ignore_index=True
    )
    return {
        "sample_manifest": accepted.reset_index(drop=True),
        "branch_manifest": branch_manifest.reset_index(drop=True),
        "rejected": rejected.reset_index(drop=True),
        "actual_duplicates": duplicates.reset_index(drop=True),
        "pending": parts["pending"].reset_index(drop=True),
        "missing_confirmed": parts["missing_confirmed"].reset_index(
            drop=True
        ),
        "completed_total": int(len(completed)),
        "hard_violation_total": int(len(hard_failed)),
    }


def partial_accounting(
    bundle: dict,
    *,
    planned_scope_total: int,
    run_uuid: str,
    input_sha: str,
    config_sha: str,
    code_sha: str,
    current_batch_completed: int | None = None,
) -> dict:
    """Fail-closed partial accounting; ``scope_complete`` is always False."""
    accepted = int(len(bundle["sample_manifest"]))
    rejected = int(len(bundle["rejected"]))
    duplicates = int(len(bundle["actual_duplicates"]))
    pending = int(len(bundle["pending"]))
    missing = int(len(bundle["missing_confirmed"]))
    completed = int(bundle["completed_total"])
    batch = (
        int(current_batch_completed)
        if current_batch_completed is not None
        else completed
    )
    return {
        "planned_scope_total": int(planned_scope_total),
        "completed_scope_total": completed,
        "current_batch_completed": batch,
        "accepted": accepted,
        "rejected": rejected + duplicates,
        "pending": pending,
        "missing_confirmed": missing,
        "remaining": pending,
        "batch_complete": True,
        "scope_complete": False,
        "partial_only": True,
        "run_uuid": str(run_uuid),
        "input_sha": str(input_sha),
        "config_sha": str(config_sha),
        "code_git_sha": str(code_sha),
    }


def _share(series: pd.Series) -> float:
    if not len(series):
        return 0.0
    return float(series.value_counts(normalize=True).max())


def audit_partial_quality(
    bundle: dict, *, thresholds: dict | None = None
) -> dict:
    """Unified partial scientific audit over completed cases only.

    Sections: execution authenticity (hard, any violation fails), action
    informativeness, coverage, and label boundaries.  Small batches never
    fail for not yet reaching final counts, but a completed set whose
    continuous labels are all identical or whose actions all projected to a
    single actual schedule stops scale-up immediately.
    """
    thresholds = thresholds or {}
    accepted = bundle["sample_manifest"]
    completed_total = int(bundle["completed_total"])
    missing_total = int(len(bundle["missing_confirmed"]))
    hard_violations = int(bundle["hard_violation_total"])

    authenticity = {
        "hard_violation_total": hard_violations,
        "missing_confirmed_total": missing_total,
        "all_completed_cases_authentic": hard_violations == 0
        and missing_total == 0,
    }

    requested = accepted.get(
        "requested_schedule_sha256", pd.Series(dtype=object)
    )
    actual = accepted.get(
        "actual_schedule_sha256", pd.Series(dtype=object)
    )
    collapse = 0
    if len(accepted) and len(requested) and len(actual):
        grouped = accepted.groupby(
            ["event_id", "checkpoint_id", "actual_schedule_sha256"],
            dropna=False,
        )["requested_schedule_sha256"].nunique()
        collapse = int((grouped - 1).clip(lower=0).sum())
    distance = _num_col(accepted, "actual_action_distance")
    local = _num_col(accepted, "local_response_magnitude")
    deltas = {name: _num_col(accepted, name) for name in DELTA_COLUMNS}
    informativeness = {
        "actual_duplicate_total": int(len(bundle["actual_duplicates"])),
        "noop_total": int(
            (bundle["rejected"].get("rejection_reason", pd.Series(dtype=object))
             == "no_op_not_accepted").sum()
        ),
        "projection_collapse_total": collapse,
        "action_distance_nonzero": int((distance.fillna(0) > 0).sum()),
        "local_response_nonzero": int((local.fillna(0) > 0).sum()),
        "delta_variance": {
            name: float(series.var(ddof=0))
            if series.notna().sum() > 1
            else 0.0
            for name, series in deltas.items()
        },
        "dead_zone_ratio": float(_bool_col(accepted, "neutral").mean())
        if len(accepted)
        else 0.0,
    }

    coverage = {
        f"{column}_count": int(accepted[column].nunique())
        if column in accepted
        else 0
        for column in COVERAGE_COLUMNS
    }
    coverage["state_group_count"] = (
        int(
            accepted.drop_duplicates(["event_id", "checkpoint_id"]).shape[0]
        )
        if {"event_id", "checkpoint_id"} <= set(accepted)
        else 0
    )
    coverage["max_event_share"] = _share(
        accepted.get("event_id", pd.Series(dtype=object))
    )
    coverage["max_family_share"] = _share(
        accepted.get("candidate_family", pd.Series(dtype=object))
    )

    label_flags = (
        "pfv_safe",
        "tfv_improved",
        "peak_noninferior",
        "neutral",
        "joint_noninferior",
        "materially_beneficial",
    )
    boundaries = {}
    for flag in label_flags:
        series = _bool_col(accepted, flag)
        boundaries[f"{flag}_true"] = int(series.sum())
        boundaries[f"{flag}_false"] = int((~series).sum()) if len(
            accepted
        ) else 0
    boundaries["hard_negative_total"] = (
        int(accepted["hard_negative_type"].notna().sum())
        if "hard_negative_type" in accepted
        else 0
    )
    all_constant = bool(
        len(accepted) >= 2
        and all(
            series.notna().any() and series.nunique(dropna=True) <= 1
            for series in deltas.values()
        )
    )
    all_collapsed = bool(
        len(accepted) >= 2
        and "actual_schedule_sha256" in accepted
        and accepted["actual_schedule_sha256"].nunique() == 1
    )
    boundaries["all_continuous_labels_constant"] = all_constant
    boundaries["all_actions_projected_to_one_actual"] = all_collapsed

    hard_pass = authenticity["all_completed_cases_authentic"]
    stop_now = all_constant or all_collapsed
    status = "pass" if hard_pass and not stop_now else "scientific_fail"
    return {
        "status": status,
        "partial_only": True,
        "full_gate_pass": False,
        "completed_scope_total": completed_total,
        "accepted_total": int(len(accepted)),
        "authenticity": {
            key: bool(value) if isinstance(value, bool) else value
            for key, value in authenticity.items()
        },
        "informativeness": informativeness,
        "coverage": coverage,
        "label_boundaries": boundaries,
        "stop_scale_up": bool(not hard_pass or stop_now),
    }


def applicable_release_level(completed_total: int) -> int:
    level = -1
    for candidate, minimum in sorted(RELEASE_LEVEL_THRESHOLDS.items()):
        if int(completed_total) >= minimum:
            level = candidate
    return level


def progressive_release_gate(
    bundle: dict,
    *,
    level: int,
    evidence: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Level 0 (1 case) / 1 (16) / 2 (pilot 40) scale-up gates.

    Passing a level authorises only the next batch size; it is never a full
    gate pass.  Any hard authenticity violation at any level blocks.
    """
    evidence = evidence or {}
    config = config or {}
    accepted = bundle["sample_manifest"]
    completed = int(bundle["completed_total"])
    quality = audit_partial_quality(bundle)
    checks: dict[str, bool] = {
        "completed_minimum_reached": completed
        >= RELEASE_LEVEL_THRESHOLDS.get(int(level), 1),
        "hard_authenticity_clean": quality["authenticity"][
            "all_completed_cases_authentic"
        ],
        "no_stop_condition": not quality["stop_scale_up"],
    }
    if int(level) == 0:
        checks.update(
            {
                "single_case_completed": completed == 1,
                "single_case_accepted": len(accepted) == 1
                and len(bundle["rejected"]) == 0,
                "output_isolated": bool(
                    _bool_col(accepted, "output_isolated").all()
                )
                if len(accepted)
                else False,
            }
        )
    if int(level) >= 1:
        distance = _num_col(accepted, "actual_action_distance").fillna(0)
        local = _num_col(accepted, "local_response_magnitude").fillna(0)
        deltas = [
            _num_col(accepted, name).fillna(0) for name in DELTA_COLUMNS
        ]
        checks.update(
            {
                "actual_duplicates_zero": len(bundle["actual_duplicates"])
                == 0,
                "no_broken_process_pool": not bool(
                    evidence.get("broken_process_pool", False)
                ),
                "no_reference_cache_conflict": int(
                    evidence.get("reference_lock_conflicts", 0)
                )
                == 0,
                "candidate_families_ge_2": (
                    accepted["candidate_family"].nunique() >= 2
                    if "candidate_family" in accepted
                    else False
                ),
                "action_distance_not_all_zero": bool((distance > 0).any()),
                "local_response_not_all_zero": bool((local > 0).any()),
                "deltas_not_all_constant": not all(
                    series.nunique() <= 1 for series in deltas
                )
                if len(accepted) >= 2
                else True,
            }
        )
    if int(level) >= 2:
        noise = config.get("noise_floor", {})
        caps = config.get("caps", {})
        responsive = accepted[
            accepted.get("checkpoint_role", "") == "responsive"
        ]
        local = _num_col(responsive, "local_response_magnitude").fillna(0)
        response_rate = (
            float((local > 0).mean()) if len(responsive) else 0.0
        )
        moving = 0
        for name in DELTA_COLUMNS:
            series = _num_col(accepted, name).fillna(0)
            floor = float(noise.get(name, 0.0))
            if (series.abs() > floor).any() and series.nunique() > 1:
                moving += 1
        joint = _bool_col(accepted, "joint_noninferior")
        safe = _bool_col(accepted, "pfv_safe")
        noop_cap = float(caps.get("noop_ratio_max", 0.2))
        collapse_cap = float(caps.get("projection_collapse_ratio_max", 0.2))
        total = max(completed, 1)
        quality_info = quality["informativeness"]
        checks.update(
            {
                "responsive_local_response_rate_ge_70": response_rate
                >= 0.70,
                "two_deltas_exceed_noise": moving >= 2,
                "not_all_joint_beneficial": not bool(joint.all())
                if len(accepted)
                else False,
                "not_all_unsafe": bool(safe.any()) if len(accepted) else False,
                "max_family_share_le_50": quality["coverage"][
                    "max_family_share"
                ]
                <= 0.50,
                "max_event_share_le_25": quality["coverage"][
                    "max_event_share"
                ]
                <= 0.25,
                "noop_ratio_below_cap": (
                    quality_info["noop_total"] / total
                )
                <= noop_cap,
                "collapse_ratio_below_cap": (
                    quality_info["projection_collapse_total"] / total
                )
                <= collapse_cap,
            }
        )
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "level": int(level),
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "completed_scope_total": completed,
        "accepted_total": int(len(accepted)),
        "partial_only": True,
        "full_gate_pass": False,
    }
