"""Pilot400 dedicated reducers (spec section VII).

``reduce_pilot_sample`` collapses exactly four same-state branches into one
labelled sample row; ``build_pilot_partial_bundle`` reduces whatever samples
are already complete (incomplete samples are ``pending`` -- never missing);
``build_pilot_dataset`` is the full-scope builder used only after
RunPilot400 reports ``scope_complete``.

Cross-branch labels are computed here and only here: a single-branch run
worker never sees another branch's output.  All KPI windows use
``checkpoint < elapsed <= checkpoint + 120``; Pilot runs stop at H120, so
full-event labels are structurally ineligible and stay NaN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.prompt3.gate5r_pipeline import (
    branch_state_hashes,
    hashes_match_across_branches,
    schedule_action_cost,
)
from .labels import add_ranking_labels, classify_labels, window_kpis
from .partial_audit import HARD_AUTHENTICITY_COLUMNS
from .peak_boundary import (
    _actual_setting_columns,
    _local_response_magnitude,
    _post_window,
    _schedule_sha,
)
from .pilot_candidates import PILOT_BRANCH_ROLES


REQUIRED_BRANCHES = set(PILOT_BRANCH_ROLES)

TEMPORAL_STEPS = 12
STEP_MINUTES = 10.0

TEMPORAL_RESIDUAL_SIGNALS = (
    "priority_depth_residual",
    "sentinel_depth_residual",
    "active_link_flow_residual",
    "storage_volume_residual",
    "tfv_rate_residual",
    "system_stored_volume_residual",
    "conduit_fullness_summary_residual",
)


def _twelve_step_means(
    detail: pd.DataFrame, checkpoint: float, series: pd.Series
) -> list[float]:
    """Aggregate a 5-min signal into 12 x 10-min decision-step means."""
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    values = pd.to_numeric(series, errors="coerce")
    steps: list[float] = []
    for step in range(TEMPORAL_STEPS):
        low = checkpoint + step * STEP_MINUTES
        high = checkpoint + (step + 1) * STEP_MINUTES
        mask = (elapsed > low) & (elapsed <= high)
        chunk = values[mask]
        steps.append(float(chunk.mean()) if len(chunk) else float("nan"))
    return steps


def _signal_series(
    detail: pd.DataFrame,
    signal: str,
    priority_nodes: list[str],
    facility_ids: list[str],
) -> pd.Series:
    def mean_of(prefix: str, names: list[str]) -> pd.Series:
        columns = [
            f"{prefix}:{name}"
            for name in names
            if f"{prefix}:{name}" in detail
        ]
        if not columns:
            return pd.Series(np.nan, index=detail.index)
        return detail[columns].apply(pd.to_numeric, errors="coerce").mean(
            axis=1
        )

    if signal == "priority_depth_residual":
        return mean_of("h", priority_nodes)
    if signal == "sentinel_depth_residual":
        head = mean_of("head", priority_nodes)
        return head if head.notna().any() else mean_of("h", priority_nodes)
    if signal == "active_link_flow_residual":
        return mean_of("flow", facility_ids)
    if signal == "storage_volume_residual":
        columns = [c for c in detail if c.startswith("storage_volume:")]
        if not columns:
            return pd.Series(np.nan, index=detail.index)
        return detail[columns].apply(pd.to_numeric, errors="coerce").sum(
            axis=1
        )
    if signal == "tfv_rate_residual":
        return detail.get(
            "tfv_rate_m3s", pd.Series(np.nan, index=detail.index)
        )
    if signal == "system_stored_volume_residual":
        return detail.get(
            "system_stored_volume_m3",
            pd.Series(np.nan, index=detail.index),
        )
    if signal == "conduit_fullness_summary_residual":
        return detail.get(
            "excess_fullness_mean", pd.Series(np.nan, index=detail.index)
        )
    raise ValueError(f"unknown temporal residual signal: {signal}")


def temporal_residuals(
    candidate_detail: pd.DataFrame,
    reference_detail: pd.DataFrame,
    checkpoint: float,
    *,
    priority_nodes: list[str],
    facility_ids: list[str],
) -> dict[str, str]:
    """Seven 12-step candidate-minus-dynamic-internal residual series."""
    result: dict[str, str] = {}
    for signal in TEMPORAL_RESIDUAL_SIGNALS:
        cand = _twelve_step_means(
            candidate_detail,
            checkpoint,
            _signal_series(
                candidate_detail, signal, priority_nodes, facility_ids
            ),
        )
        ref = _twelve_step_means(
            reference_detail,
            checkpoint,
            _signal_series(
                reference_detail, signal, priority_nodes, facility_ids
            ),
        )
        residual = [
            float(c - r) if np.isfinite(c) and np.isfinite(r) else None
            for c, r in zip(cand, ref)
        ]
        result[signal] = json.dumps(residual)
    return result


def _action_authority_class(k_target: int, is_noop: bool) -> str:
    if is_noop:
        return "none"
    if k_target <= 2:
        return "low"
    if k_target <= 5:
        return "medium"
    return "high"


def reduce_pilot_sample(
    candidate_row: pd.Series | dict,
    completion_rows: dict[str, pd.Series],
    details: dict[str, pd.DataFrame],
    *,
    priority_nodes: list[str],
    facility_ids: list[str],
    scientific_margin: dict[str, float],
    dead_zone: dict[str, float],
) -> dict:
    """Reduce one complete four-branch Pilot sample into a labelled row.

    ``completion_rows`` and ``details`` must both cover exactly the four
    branch roles; the caller is responsible for pending bookkeeping.
    """
    base = (
        candidate_row.to_dict()
        if isinstance(candidate_row, pd.Series)
        else dict(candidate_row)
    )
    if set(details) != REQUIRED_BRANCHES:
        raise ValueError("reduce_pilot_sample needs exactly four branches")
    checkpoint = float(base["checkpoint_min"])
    statuses = {
        branch: str(row.get("status", ""))
        for branch, row in completion_rows.items()
    }
    hashes = {
        branch: branch_state_hashes(
            detail, checkpoint_min=checkpoint, facility_ids=facility_ids
        )
        for branch, detail in details.items()
    }
    same_state = hashes_match_across_branches(
        hashes,
        keys=("prefix_history_sha256", "checkpoint_pre_action_sha256"),
    )
    prefix_ok = (
        len({h.get("prefix_history_sha256") for h in hashes.values()}) == 1
    )
    kpis = {
        branch: window_kpis(detail, priority_nodes, checkpoint, dt_sec=300)
        for branch, detail in details.items()
    }
    cand = kpis["candidate"]
    delta_pfv = cand["PFV"] - kpis["no_control"]["PFV"]
    delta_tfv = cand["TFV"] - kpis["dynamic_internal_rules"]["TFV"]
    delta_peak = (
        cand["peak_TFV_rate"]
        - kpis["dynamic_internal_rules"]["peak_TFV_rate"]
    )
    kpi_ok = all(
        np.isfinite(
            [value["PFV"], value["TFV"], value["peak_TFV_rate"]]
        ).all()
        for value in kpis.values()
    )
    projected = np.asarray(
        json.loads(str(base.get("projected_schedule_json", "[]"))),
        dtype=float,
    )
    anchor = np.asarray(
        json.loads(str(base.get("anchor_schedule_json", "[]"))),
        dtype=float,
    )
    action_cost = (
        schedule_action_cost(projected, anchor)
        if projected.size and projected.shape == anchor.shape
        else 0.0
    )
    labels = classify_labels(
        delta_pfv,
        delta_tfv,
        delta_peak,
        scientific_margin=scientific_margin,
        dead_zone=dead_zone,
        action_cost=action_cost,
    )
    candidate_detail = details["candidate"]
    post = _post_window(candidate_detail, checkpoint)
    actual_columns, explicit = _actual_setting_columns(post, facility_ids)
    actual_matrix = (
        post[actual_columns].to_numpy(float)
        if actual_columns
        else np.empty((0, 0))
    )
    actual_sha = _schedule_sha(actual_matrix)
    anchor_vector = (
        anchor[0] if anchor.ndim == 2 and anchor.size else np.zeros(0)
    )
    actual_distance = (
        float(np.abs(actual_matrix - anchor_vector).sum())
        if actual_matrix.size
        and anchor_vector.size == actual_matrix.shape[1]
        else 0.0
    )
    readback_columns = [
        f"readback_setting:{item}"
        for item in facility_ids
        if f"readback_setting:{item}" in post
    ]
    readback_ok = bool(
        explicit
        and len(readback_columns) == len(facility_ids)
        and np.allclose(
            post[
                [f"actual_setting:{item}" for item in facility_ids]
            ].to_numpy(float),
            post[readback_columns].to_numpy(float),
            atol=1e-8,
            equal_nan=False,
        )
    )
    local_magnitude = _local_response_magnitude(
        details["candidate"],
        details["hold_previous"],
        checkpoint,
        facility_ids,
    )
    chain_ok = all(
        f"{prefix}:{item}" in candidate_detail
        for item in facility_ids
        for prefix in (
            "requested_setting",
            "target_setting",
            "actual_setting",
            "readback_setting",
        )
    )

    def _no_hotstart(row: pd.Series) -> bool:
        result = row.get("result")
        result = result if isinstance(result, dict) else {}
        return (
            not bool(result.get("hotstart_used", False))
            and int(result.get("use_hotstart_call_count", 0) or 0) == 0
            and int(result.get("save_hotstart_call_count", 0) or 0) == 0
        )

    no_hotstart = all(
        _no_hotstart(row) for row in completion_rows.values()
    )

    def _inp_path(row: pd.Series) -> str | None:
        try:
            return json.loads(str(row.get("runner_kwargs", "{}"))).get(
                "inp_path"
            )
        except (ValueError, TypeError):
            return None

    inp_paths = {_inp_path(row) for row in completion_rows.values()}
    physical_ok = len(inp_paths) == 1 and None not in inp_paths
    rainfall_ok = (
        len(
            {
                str(row.get("rainfall_sha256"))
                for row in completion_rows.values()
            }
        )
        == 1
        and bool(base.get("rainfall_sha256"))
    )
    elapsed = pd.to_numeric(candidate_detail["elapsed_min"], errors="coerce")
    h120_ok = bool(
        float(elapsed.max()) >= checkpoint + 120.0 - 1e-6 and len(post) >= 1
    )
    k_target = int(base.get("k_target", 0) or 0)
    actuator_ok = bool(base.get("binary_semantics_ok")) and bool(
        base.get("rate_limit_ok")
    )
    engineering_ok = (
        bool(base.get("binary_semantics_ok"))
        and bool(base.get("rate_limit_ok"))
        and bool(base.get("dwell_ok"))
        and bool(base.get("interlock_ok"))
    )
    hard = {
        "completion_valid": all(
            statuses[branch] == "pass" for branch in REQUIRED_BRANCHES
        )
        and all(
            bool(row.get("input_sha"))
            for row in completion_rows.values()
        ),
        "four_branches_complete": True,
        "same_state_ok": bool(same_state),
        "physical_sha_ok": bool(physical_ok),
        "rainfall_sha_ok": bool(rainfall_ok),
        "prefix_sha_ok": bool(prefix_ok),
        "action_stage_chain_complete": bool(chain_ok),
        "no_hotstart": bool(no_hotstart),
        "k_le_8": 1 <= k_target <= 8,
        "actuator_semantics_ok": actuator_ok,
        "engineering_limits_ok": engineering_ok,
        "h120_window_complete": h120_ok,
        "kpi_recompute_ok": bool(kpi_ok),
        "reference_cache_sha_ok": bool(same_state),
    }
    residuals = temporal_residuals(
        details["candidate"],
        details["dynamic_internal_rules"],
        checkpoint,
        priority_nodes=priority_nodes,
        facility_ids=facility_ids,
    )
    is_noop = actual_distance <= 1e-9
    # Pilot output isolation: the candidate detail lives inside the
    # sample's own run directory while each reference detail lives in the
    # shared cache directory named after its branch role; all four detail
    # paths must be distinct.
    detail_paths = {
        branch: str(row.get("detail_path", ""))
        for branch, row in completion_rows.items()
    }
    sample_id = str(base.get("sample_id", ""))
    output_isolated = (
        bool(sample_id)
        and sample_id in detail_paths.get("candidate", "")
        and all(
            branch in detail_paths.get(branch, "")
            for branch in REQUIRED_BRANCHES
            if branch != "candidate"
        )
        and len(set(detail_paths.values())) == len(REQUIRED_BRANCHES)
    )
    return {
        **base,
        "branch_role": "candidate",
        "is_reference_branch": False,
        "actual_schedule_sha256": actual_sha,
        "actual_action_distance": actual_distance,
        "local_response_magnitude": local_magnitude,
        "is_noop": bool(is_noop),
        "state_hash_match": bool(same_state),
        "output_isolated": bool(output_isolated),
        "readback_ok": readback_ok,
        "action_cost": float(action_cost),
        "delta_pfv_h120_vs_no_control": float(delta_pfv),
        "delta_tfv_h120_vs_dynamic_internal": float(delta_tfv),
        "delta_peak_h120_vs_dynamic_internal": float(delta_peak),
        **residuals,
        **hard,
        **labels,
        "action_authority_class": _action_authority_class(
            k_target, bool(is_noop)
        ),
        "h120_eligible": bool(h120_ok),
        "full_event_eligible": False,
        "label_validity_pfv": bool(np.isfinite(delta_pfv)),
        "label_validity_tfv": bool(np.isfinite(delta_tfv)),
        "label_validity_peak": bool(np.isfinite(delta_peak)),
        "label_validity_full": False,
        "recovery_class": "",
        "censored": not bool(h120_ok),
    }


def build_pilot_partial_bundle(
    candidate_plan: pd.DataFrame,
    branch_plan: pd.DataFrame,
    completions: pd.DataFrame,
    *,
    priority_nodes: list[str],
    facility_ids: list[str],
    scientific_margin: dict[str, float],
    dead_zone: dict[str, float],
) -> dict:
    """Reduce every Pilot sample whose four branches are complete.

    A sample missing any branch completion is ``pending`` -- never missing
    and never a failure; ``scope_complete`` is decided by the pipeline, not
    here.  Completed samples failing any hard-authenticity check are
    rejected fail-closed; no-ops and duplicate actual schedules are
    funnelled out exactly like the Peak partial builder.
    """
    for name, frame, needed in (
        ("candidate_plan", candidate_plan, {"sample_id", "checkpoint_min"}),
        ("branch_plan", branch_plan, {"sample_id", "case_id", "branch_role"}),
    ):
        absent = needed - set(frame)
        if absent:
            raise ValueError(f"{name} missing: {sorted(absent)}")
    comp = (
        completions.copy()
        if completions is not None and not completions.empty
        else pd.DataFrame(columns=["case_id"])
    )
    if "case_id" in comp:
        comp["case_id"] = comp["case_id"].astype(str)
        comp_by_case = {
            str(row["case_id"]): row for _, row in comp.iterrows()
        }
    else:
        comp_by_case = {}
    candidates_by_id = {
        str(row["sample_id"]): row for _, row in candidate_plan.iterrows()
    }

    accepted_rows: list[dict] = []
    hard_failed_rows: list[dict] = []
    noop_rows: list[dict] = []
    branch_rows: list[dict] = []
    pending_ids: list[str] = []
    completed_samples = 0

    for sample_id, group in branch_plan.groupby("sample_id"):
        sample_id = str(sample_id)
        candidate_row = candidates_by_id.get(sample_id)
        if candidate_row is None:
            pending_ids.append(sample_id)
            continue
        branch_ids = {
            str(row["branch_role"]): str(row["case_id"])
            for _, row in group.iterrows()
        }
        present = {
            branch: comp_by_case[case_id]
            for branch, case_id in branch_ids.items()
            if case_id in comp_by_case
        }
        if set(present) != REQUIRED_BRANCHES:
            pending_ids.append(sample_id)
            continue
        completed_samples += 1
        for branch, row in present.items():
            record = row.to_dict()
            record["sample_id"] = sample_id
            record["branch_role"] = branch
            record["is_reference_branch"] = branch != "candidate"
            record["counted_as_sample"] = branch == "candidate"
            branch_rows.append(record)
        statuses = {
            branch: str(row.get("status", ""))
            for branch, row in present.items()
        }
        details: dict[str, pd.DataFrame] = {}
        details_ok = all(
            statuses[branch] == "pass" for branch in REQUIRED_BRANCHES
        )
        if details_ok:
            for branch, row in present.items():
                path = Path(str(row.get("detail_path", "")))
                if not path.exists():
                    details_ok = False
                    break
                details[branch] = pd.read_csv(path)
        if not details_ok or set(details) != REQUIRED_BRANCHES:
            hard_failed_rows.append(
                {
                    **candidate_row.to_dict(),
                    "sample_id": sample_id,
                    "branch_role": "candidate",
                    "rejection_reason": "four_branch_incomplete_or_failed",
                    **{
                        column: False
                        for column in HARD_AUTHENTICITY_COLUMNS
                    },
                }
            )
            continue
        row = reduce_pilot_sample(
            candidate_row,
            present,
            details,
            priority_nodes=priority_nodes,
            facility_ids=facility_ids,
            scientific_margin=scientific_margin,
            dead_zone=dead_zone,
        )
        if not all(row[column] for column in HARD_AUTHENTICITY_COLUMNS):
            hard_failed_rows.append(
                {**row, "rejection_reason": "hard_authenticity_failed"}
            )
        elif row["is_noop"]:
            noop_rows.append(
                {**row, "rejection_reason": "no_op_not_accepted"}
            )
        else:
            accepted_rows.append(row)

    accepted_df = pd.DataFrame(accepted_rows)
    dup_keys = ["event_id", "checkpoint_id", "actual_schedule_sha256"]
    if len(accepted_df) and all(key in accepted_df for key in dup_keys):
        dup_mask = accepted_df.duplicated(dup_keys, keep="first")
    else:
        dup_mask = pd.Series(False, index=accepted_df.index)
    duplicates_df = accepted_df[dup_mask].copy()
    if len(duplicates_df):
        duplicates_df["rejection_reason"] = "duplicate_actual_schedule"
    final_accepted = accepted_df[~dup_mask].copy()
    rejected = pd.concat(
        [pd.DataFrame(hard_failed_rows), pd.DataFrame(noop_rows)],
        ignore_index=True,
    )
    pending = (
        candidate_plan[
            candidate_plan["sample_id"].astype(str).isin(pending_ids)
        ]
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )
    return {
        "sample_manifest": final_accepted.reset_index(drop=True),
        "branch_manifest": pd.DataFrame(branch_rows).reset_index(drop=True),
        "rejected": rejected.reset_index(drop=True),
        "actual_duplicates": duplicates_df.reset_index(drop=True),
        "pending": pending,
        "missing_confirmed": candidate_plan.head(0).copy(),
        "completed_total": int(completed_samples),
        "hard_violation_total": int(len(hard_failed_rows)),
    }


def add_flat_state_labels(samples: pd.DataFrame) -> pd.DataFrame:
    """A state is flat when every accepted candidate there is neutral."""
    result = samples.copy()
    if not len(result):
        result["flat_state"] = pd.Series(dtype=bool)
        return result
    flat_by_state = result.groupby(["event_id", "checkpoint_id"])[
        "neutral"
    ].transform(lambda column: column.astype(bool).all())
    result["flat_state"] = flat_by_state.astype(bool)
    return result


def build_pilot_dataset(
    candidate_plan: pd.DataFrame,
    branch_plan: pd.DataFrame,
    completions: pd.DataFrame,
    *,
    priority_nodes: list[str],
    facility_ids: list[str],
    scientific_margin: dict[str, float],
    dead_zone: dict[str, float],
) -> dict:
    """Full-scope Pilot dataset; only valid once RunPilot400 is complete.

    Accounting: planned == accepted + rejected + pending + missing.  In the
    full build every still-pending sample is confirmed missing.
    """
    bundle = build_pilot_partial_bundle(
        candidate_plan,
        branch_plan,
        completions,
        priority_nodes=priority_nodes,
        facility_ids=facility_ids,
        scientific_margin=scientific_margin,
        dead_zone=dead_zone,
    )
    samples = bundle["sample_manifest"]
    if len(samples):
        samples = add_flat_state_labels(samples)
        samples = add_ranking_labels(samples)
    planned = int(len(candidate_plan))
    rejected = int(len(bundle["rejected"])) + int(
        len(bundle["actual_duplicates"])
    )
    # Full scope: anything still pending is a confirmed missing sample.
    missing_confirmed = bundle["pending"].copy()
    accounting = {
        "planned": planned,
        "accepted": int(len(samples)),
        "rejected": rejected,
        "pending": 0,
        "missing": int(len(missing_confirmed)),
        "accounting_closed": planned
        == int(len(samples)) + rejected + int(len(missing_confirmed)),
    }
    return {
        **bundle,
        "sample_manifest": samples.reset_index(drop=True),
        "pending": candidate_plan.head(0).copy(),
        "missing_confirmed": missing_confirmed.reset_index(drop=True),
        "accounting": accounting,
    }
