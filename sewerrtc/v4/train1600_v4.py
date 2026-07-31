"""Train1600 V4 model-training readiness pure functions (spec sections 1-3).

Everything here is side-effect free: functions take the frozen Train1600 V3
sample manifest (a ``pandas.DataFrame``) plus frozen thresholds and return
plain dicts / record lists.  The ``pipeline_train_v4`` handlers own all file
I/O so these functions stay unit-testable and cannot mutate the frozen
1600-sample evidence.

Scope covered:

* section 1  -- ``build_train1600_v3_freeze_payload`` (immutable freeze record)
* section 2  -- ``audit_train1600_learnability_v4`` (training readiness audit)
* section 3  -- ``evaluate_model_training_authorization_v4`` (authorization)

The audit never rewrites labels, splits, margins, dead-zones or Locked data;
it only measures them.  When the online candidate generator can emit K values
that never appear in the training domain, the verdict degrades to
``conditional_pass`` and records a K supplement obligation -- it never claims
coverage it does not have.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .dataset import CONTINUOUS_LABELS, TEMPORAL_RESIDUALS, feature_schema

# ---------------------------------------------------------------------------
# Frozen semantic constants
# ---------------------------------------------------------------------------

# The online V4 candidate generator explicitly emits these K values
# (``sewerrtc/control/v4_candidate_generator.py`` toward_no_control /
# staggered_sparse / priority_protection loops).  K=1 and K=2 are online-only
# and never appear in the frozen Train1600 V3 domain.
ONLINE_CANDIDATE_K_VALUES: tuple[int, ...] = (1, 2, 4, 6, 8)

SPLITS: tuple[str, ...] = ("train", "calibration", "locked_validation")

# Core deployable-safety labels that must be two-sided in Train.
CORE_CLASSIFICATION_LABELS: tuple[str, ...] = (
    "pfv_safe",
    "tfv_improved",
    "peak_noninferior",
)

AUXILIARY_CLASSIFICATION_LABELS: tuple[str, ...] = (
    "joint_noninferior",
    "materially_beneficial",
    "neutral",
)

# The horizon is 120 min at a 10 min control step -> 12 residual steps.
HORIZON_STEPS = 12

# Process residual columns that must exist as populated 12-step future series.
PROCESS_RESIDUAL_COLUMNS: tuple[str, ...] = TEMPORAL_RESIDUALS

# Columns that are outcomes / labels / future information and must never be
# offered to the model as inputs (spec section 2.5).
PROHIBITED_INPUT_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *CONTINUOUS_LABELS,
            *TEMPORAL_RESIDUALS,
            "pfv_safe",
            "tfv_improved",
            "peak_noninferior",
            "joint_noninferior",
            "materially_beneficial",
            "neutral",
            "benefit_cost_ratio",
            "hard_negative_type",
            "action_authority_class",
            "recovery_class",
            "feasible_rank",
            "regret_to_exact_best",
            "pairwise_preference",
            "locally_responsive",
            "confirmed_flat",
            "tfv_noninferior",
            "joint_found_in_sampled_set",
            "exact_search_performed",
            "candidate_search_budget",
            "state_feasibility_label_source",
            "state_feasibility_label_validity",
        )
    )
)

# Minimum feasible Locked states required before a model-safety PASS can be
# claimed (spec sections 2.1 / 8).  Below this the Locked split is reported as
# underpowered; it is never faked to PASS.
MIN_FEASIBLE_LOCKED_STATES = 5

# Minimum near-boundary sample support before a continuous head is flagged as a
# calibration risk (spec section 2.4).  A flag never deletes data or blocks
# training; it only prioritises later boundary accrual.
NEAR_BOUNDARY_SUPPORT_FLOOR = 20

CONTINUOUS_HEAD_KEYS: dict[str, str] = {
    "pfv": "delta_pfv_h120_vs_no_control",
    "tfv": "delta_tfv_h120_vs_dynamic_internal",
    "peak": "delta_peak_h120_vs_dynamic_internal",
}

# Frozen threshold config keys per head.
_MARGIN_KEYS = {"pfv": "pfv_m3", "tfv": "tfv_m3", "peak": "peak_m3s"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _state_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["event_id"].astype(str)
        + "::"
        + frame["checkpoint_id"].astype(str)
    )


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    if raw.dtype == bool:
        return raw
    return raw.map(
        lambda value: str(value).strip().lower() in {"true", "1", "1.0"}
    )


def _two_sided(frame: pd.DataFrame, column: str) -> tuple[int, int, bool]:
    flags = _bool_series(frame, column)
    positive = int(flags.sum())
    negative = int((~flags).sum())
    return positive, negative, positive > 0 and negative > 0


def _parse_residual_lengths(series: pd.Series) -> set[int]:
    lengths: set[int] = set()
    for value in series.dropna():
        try:
            arr = json.loads(value) if isinstance(value, str) else value
        except (ValueError, TypeError):
            lengths.add(-1)
            continue
        if isinstance(arr, list):
            lengths.add(len(arr))
        else:
            lengths.add(-1)
    return lengths


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


# ---------------------------------------------------------------------------
# Section 1: freeze payload
# ---------------------------------------------------------------------------

def build_train1600_v3_freeze_payload(
    *,
    dataset_audit: dict,
    split_counts: dict[str, int],
    event_count: int,
    state_count: int,
    file_manifest: dict[str, str],
    skipped_runs_dirs: int,
    reference_cache_sha256: str,
    code_sha256: str,
) -> dict:
    """Immutable freeze record for the accepted Train1600 V3 evidence.

    Raw per-case SWMM trajectories under ``runs/`` are intentionally not copied
    (they are regenerable and multi-GB); their provenance SHAs already live in
    the copied run/branch manifests, so freezing those manifests freezes the
    trajectory provenance.
    """
    if str(dataset_audit.get("status")) != "pass":
        raise ValueError(
            "Train1600 V3 freeze requires a passing Dataset Gate audit, got "
            f"{dataset_audit.get('status')!r}"
        )
    return {
        "freeze_id": "train1600_v3_freeze",
        "dataset_gate_pass": True,
        "accepted": 1600,
        "train": int(split_counts.get("train", 0)),
        "calibration": int(split_counts.get("calibration", 0)),
        "locked": int(split_counts.get("locked_validation", 0)),
        "events": int(event_count),
        "states": int(state_count),
        "immutable": True,
        "model_training_authorized_pending_readiness": True,
        "raw_trajectory_provenance": "captured_in_run_and_branch_manifests",
        "raw_runs_dirs_not_copied": int(skipped_runs_dirs),
        "reference_cache_sha256": reference_cache_sha256,
        "code_sha256": code_sha256,
        "dataset_audit_checks": dict(dataset_audit.get("checks", {})),
        "file_sha256": dict(sorted(file_manifest.items())),
        "file_count": len(file_manifest),
    }


# ---------------------------------------------------------------------------
# Section 2.1: split / event distributions
# ---------------------------------------------------------------------------

def _label_side_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {"true": 0, "false": 0}
    flags = _bool_series(frame, column)
    return {"true": int(flags.sum()), "false": int((~flags).sum())}


def split_label_distribution(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = _state_key(df)
    for split in SPLITS:
        mask = df["split"].astype(str) == split
        sub = df[mask]
        record: dict[str, Any] = {
            "split": split,
            "events": int(sub["event_id"].nunique()),
            "states": int(keys[mask].nunique()),
            "samples": int(len(sub)),
        }
        for label in (*CORE_CLASSIFICATION_LABELS, *AUXILIARY_CLASSIFICATION_LABELS):
            sides = _label_side_counts(sub, label)
            record[f"{label}_true"] = sides["true"]
            record[f"{label}_false"] = sides["false"]
        record["candidate_families"] = int(sub["candidate_family"].nunique())
        record["k_values"] = json.dumps(
            sorted(int(k) for k in sub["k_actual"].dropna().unique())
        )
        record["strata"] = json.dumps(
            sorted(sub["predicted_stratum"].astype(str).unique().tolist())
        )
        record["hard_negative_types"] = int(
            sub["hard_negative_type"].replace("", np.nan).dropna().nunique()
        )
        records.append(record)
    return records


def event_label_distribution(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (split, event_id), sub in df.groupby(
        [df["split"].astype(str), df["event_id"].astype(str)]
    ):
        record: dict[str, Any] = {
            "split": split,
            "event_id": event_id,
            "samples": int(len(sub)),
            "states": int(_state_key(sub).nunique()),
        }
        for label in CORE_CLASSIFICATION_LABELS:
            sides = _label_side_counts(sub, label)
            record[f"{label}_true"] = sides["true"]
            record[f"{label}_false"] = sides["false"]
        records.append(record)
    return records


def state_candidate_variance(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = _state_key(df)
    for state, sub in df.groupby(keys):
        record: dict[str, Any] = {
            "state_key": str(state),
            "split": str(sub["split"].iloc[0]),
            "candidates": int(len(sub)),
        }
        for head, column in CONTINUOUS_HEAD_KEYS.items():
            values = _finite(sub[column]) if column in sub else np.array([])
            record[f"{head}_std"] = (
                float(np.std(values)) if values.size else 0.0
            )
            record[f"{head}_range"] = (
                float(values.max() - values.min()) if values.size else 0.0
            )
        record["pfv_safe_true"] = int(
            _bool_series(sub, "pfv_safe").sum()
        ) if "pfv_safe" in sub else 0
        records.append(record)
    return records


def action_coverage(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        sub = df[df["split"].astype(str) == split]
        for family, fam_sub in sub.groupby(sub["candidate_family"].astype(str)):
            records.append(
                {
                    "split": split,
                    "candidate_family": family,
                    "samples": int(len(fam_sub)),
                    "k_min": int(fam_sub["k_actual"].min())
                    if fam_sub["k_actual"].notna().any()
                    else -1,
                    "k_max": int(fam_sub["k_actual"].max())
                    if fam_sub["k_actual"].notna().any()
                    else -1,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Section 2.2: K semantics
# ---------------------------------------------------------------------------

def k_semantics_audit(
    df: pd.DataFrame, *, online_k_values: Iterable[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    online = sorted({int(k) for k in online_k_values})
    target_values = sorted(int(k) for k in df["k_target"].dropna().unique())
    actual_values = sorted(int(k) for k in df["k_actual"].dropna().unique())
    requested_has_k12 = any(k in (1, 2) for k in target_values)
    online_has_k12 = any(k in (1, 2) for k in online)
    training_has_k12 = any(k in (1, 2) for k in actual_values)

    records: list[dict[str, Any]] = []
    for k in sorted(set(target_values) | set(actual_values) | set(online)):
        target_n = int((df["k_target"] == k).sum())
        actual_n = int((df["k_actual"] == k).sum())
        records.append(
            {
                "k": int(k),
                "requested_samples": target_n,
                "actual_samples": actual_n,
                "in_online_generator": bool(k in online),
                "in_training_domain": bool(actual_n > 0),
            }
        )

    per_step_max = int(df["k_actual"].max()) if df["k_actual"].notna().any() else -1
    mismatch = online_has_k12 and not training_has_k12
    summary = {
        "online_candidate_k_values": online,
        "requested_k_values": target_values,
        "actual_k_values": actual_values,
        "k_actual_max": per_step_max,
        "requested_contains_k1_or_k2": bool(requested_has_k12),
        "online_contains_k1_or_k2": bool(online_has_k12),
        "training_contains_k1_or_k2": bool(training_has_k12),
        "k_computed_against": "actual_executed_fallback_anchor",
        "online_domain_exceeds_training_domain": bool(mismatch),
        "k1_k2_supplement_required": bool(mismatch),
        "disable_online_k1_k2_until_backfilled": bool(mismatch),
        "supplement_plan": {
            "scope": "train_events_only",
            "must_not_touch_calibration_or_locked": True,
            "target_k_values": [1, 2],
        }
        if mismatch
        else None,
    }
    return records, summary


# ---------------------------------------------------------------------------
# Section 2.3: PFV zero inflation
# ---------------------------------------------------------------------------

def pfv_zero_inflation(df: pd.DataFrame, *, pfv_dead_zone: float) -> dict[str, Any]:
    column = CONTINUOUS_HEAD_KEYS["pfv"]
    values = df[column].astype(float)
    finite = values[np.isfinite(values)]
    inactive = int((finite.abs() <= pfv_dead_zone).sum())
    active = int((finite.abs() > pfv_dead_zone).sum())
    total = int(finite.size)
    keys = _state_key(df)
    per_state_zero_ratio = []
    for state, idx in df.groupby(keys).groups.items():
        sub = values.loc[idx]
        sub = sub[np.isfinite(sub)]
        if sub.size:
            per_state_zero_ratio.append(
                float((sub.abs() <= pfv_dead_zone).mean())
            )
    near_boundary = int(
        (
            (finite.abs() > pfv_dead_zone)
            & (finite.abs() <= 3.0 * pfv_dead_zone)
        ).sum()
    )
    return {
        "pfv_dead_zone": float(pfv_dead_zone),
        "inactive_samples": inactive,
        "active_samples": active,
        "total_finite_samples": total,
        "inactive_fraction": float(inactive / total) if total else 0.0,
        "mean_per_state_zero_ratio": float(np.mean(per_state_zero_ratio))
        if per_state_zero_ratio
        else 0.0,
        "near_boundary_samples": near_boundary,
        "recommended_modeling": {
            "use_pfv_hurdle_head": True,
            "active_inactive_gate": True,
            "conditional_delta_regression_on_active": True,
            "pfv_safe_auxiliary_classifier": True,
            "plain_mse_over_zeros_prohibited": True,
        },
    }


# ---------------------------------------------------------------------------
# Section 2.4: label boundary bands
# ---------------------------------------------------------------------------

def boundary_coverage(
    df: pd.DataFrame, *, margins: dict[str, float], dead_zones: dict[str, float]
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    calibration_risk_heads: list[str] = []
    for head, column in CONTINUOUS_HEAD_KEYS.items():
        margin = float(margins.get(head, 0.0))
        dead_zone = float(dead_zones.get(head, 0.0))
        scale = margin if margin > 0.0 else dead_zone
        values = _finite(df[column]) if column in df else np.array([])
        bands = {
            "lt_neg_2s": int((values < -2 * scale).sum()),
            "neg_2s_to_neg_s": int(
                ((values >= -2 * scale) & (values < -scale)).sum()
            ),
            "neg_s_to_neg_dz": int(
                ((values >= -scale) & (values < -dead_zone)).sum()
            ),
            "dead_zone": int(
                ((values >= -dead_zone) & (values <= dead_zone)).sum()
            ),
            "dz_to_s": int(
                ((values > dead_zone) & (values <= scale)).sum()
            ),
            "s_to_2s": int(
                ((values > scale) & (values <= 2 * scale)).sum()
            ),
            "gt_2s": int((values > 2 * scale).sum()),
        }
        near_boundary = int(
            (
                (np.abs(values) > dead_zone)
                & (np.abs(values) <= 3.0 * (dead_zone if dead_zone > 0 else scale))
            ).sum()
        )
        risk = near_boundary < NEAR_BOUNDARY_SUPPORT_FLOOR
        if risk:
            calibration_risk_heads.append(head)
        record = {
            "head": head,
            "margin": margin,
            "dead_zone": dead_zone,
            "effective_scale": scale,
            "near_boundary_support": near_boundary,
            "calibration_risk": bool(risk),
            **bands,
        }
        records.append(record)
    return records, calibration_risk_heads


# ---------------------------------------------------------------------------
# Section 2.5: feature leakage (static contract)
# ---------------------------------------------------------------------------

def feature_leakage_audit() -> dict[str, Any]:
    schema = feature_schema()
    input_fields: set[str] = set()
    for group in schema.values():
        if isinstance(group, list):
            input_fields.update(group)
        elif isinstance(group, dict):
            for key, value in group.items():
                if key == "fields" and isinstance(value, list):
                    input_fields.update(value)
                elif key in {"branches", "identity"} and isinstance(value, list):
                    input_fields.update(value)
    leaks = sorted(input_fields.intersection(PROHIBITED_INPUT_COLUMNS))
    # branch_evidence must expose only paths / SHAs, never realised values.
    branch_fields = schema.get("branch_evidence", {}).get("fields", [])
    value_like = sorted(
        field
        for field in branch_fields
        if field.endswith(("_kpi", "_depth", "_flow", "_delta"))
    )
    leaks = sorted(set(leaks) | set(value_like))
    return {
        "allowed_input_fields": sorted(input_fields),
        "prohibited_columns_checked": list(PROHIBITED_INPUT_COLUMNS),
        "leakage_fields": leaks,
        "leakage_count": len(leaks),
        "allowed_future_rainfall_forecast": True,
        "no_future_swmm_state_in_inputs": len(leaks) == 0,
    }


# ---------------------------------------------------------------------------
# Section 2.6: process label completeness
# ---------------------------------------------------------------------------

def residual_schema_audit(df: pd.DataFrame) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    complete = True
    for column in PROCESS_RESIDUAL_COLUMNS:
        present = column in df.columns
        nonnull = int(df[column].notna().sum()) if present else 0
        lengths = _parse_residual_lengths(df[column]) if present else set()
        ok = present and nonnull == len(df) and lengths == {HORIZON_STEPS}
        complete = complete and ok
        columns[column] = {
            "present": present,
            "nonnull_rows": nonnull,
            "step_lengths": sorted(lengths),
            "all_horizon_steps": lengths == {HORIZON_STEPS},
        }
    # Peak timing is derivable in place from the existing per-step residual
    # series (argmax step); no new SWMM run is required.
    peak_timing = {
        "materialized_column": False,
        "derivable_from_existing_residual_series": True,
        "source": "priority_depth_residual / tfv_rate_residual per-step curves",
        "requires_new_swmm": False,
    }
    sentinel = columns.get("sentinel_depth_residual", {})
    return {
        "columns": columns,
        "all_process_residuals_complete": bool(complete),
        "sentinel_present_or_masked": bool(sentinel.get("present", False)),
        "peak_timing": peak_timing,
        "enriched_manifest_required": not complete,
        "original_manifest_read_only": True,
    }


# ---------------------------------------------------------------------------
# Section 2.1 (Locked) + covariate shift
# ---------------------------------------------------------------------------

def locked_power_report(df: pd.DataFrame) -> dict[str, Any]:
    locked = df[df["split"].astype(str) == "locked_validation"]
    keys = _state_key(locked)
    feasible_states = 0
    for _state, idx in locked.groupby(keys).groups.items():
        sub = locked.loc[idx]
        if "pfv_safe" in sub and _bool_series(sub, "pfv_safe").any():
            feasible_states += 1
    label_two_sided = {}
    for label in CORE_CLASSIFICATION_LABELS:
        _, _, two_sided = _two_sided(locked, label)
        label_two_sided[label] = bool(two_sided)
    total_states = int(keys.nunique())
    state_sizes = locked.groupby(keys).size()
    states_with_full_candidate_set = int((state_sizes >= 5).sum())
    return {
        "locked_states": total_states,
        "feasible_states": feasible_states,
        "min_feasible_required": MIN_FEASIBLE_LOCKED_STATES,
        "underpowered_locked": feasible_states < MIN_FEASIBLE_LOCKED_STATES,
        "core_label_two_sided": label_two_sided,
        "states_with_5_candidates": states_with_full_candidate_set,
        "topk_computable": states_with_full_candidate_set >= 1,
    }


def covariate_shift_report(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"heads": {}}
    train = df[df["split"].astype(str) == "train"]
    for split in ("calibration", "locked_validation"):
        other = df[df["split"].astype(str) == split]
        head_stats: dict[str, Any] = {}
        for head, column in CONTINUOUS_HEAD_KEYS.items():
            tvals = _finite(train[column]) if column in train else np.array([])
            ovals = _finite(other[column]) if column in other else np.array([])
            head_stats[head] = {
                "train_mean": float(np.mean(tvals)) if tvals.size else 0.0,
                "split_mean": float(np.mean(ovals)) if ovals.size else 0.0,
                "train_std": float(np.std(tvals)) if tvals.size else 0.0,
                "split_std": float(np.std(ovals)) if ovals.size else 0.0,
            }
        report["heads"][split] = head_stats
    # Stratum share drift across splits (informational).
    strata: dict[str, Any] = {}
    for split in SPLITS:
        sub = df[df["split"].astype(str) == split]
        counts = sub["predicted_stratum"].astype(str).value_counts()
        total = int(counts.sum()) or 1
        strata[split] = {k: float(v / total) for k, v in counts.items()}
    report["stratum_share"] = strata
    return report


# ---------------------------------------------------------------------------
# Section 2: orchestrator
# ---------------------------------------------------------------------------

def _split_leakage(df: pd.DataFrame) -> dict[str, Any]:
    sha_by_split = {
        split: set(
            df[df["split"].astype(str) == split]["rainfall_sha256"]
            .astype(str)
            .unique()
        )
        for split in SPLITS
    }
    overlaps: dict[str, list[str]] = {}
    pairs = (
        ("train", "calibration"),
        ("train", "locked_validation"),
        ("calibration", "locked_validation"),
    )
    for left, right in pairs:
        shared = sha_by_split[left] & sha_by_split[right]
        if shared:
            overlaps[f"{left}__{right}"] = sorted(shared)
    return {"rainfall_sha_overlaps": overlaps, "leakage_free": not overlaps}


def _train_core_two_sided(df: pd.DataFrame) -> dict[str, Any]:
    train = df[df["split"].astype(str) == "train"]
    result: dict[str, Any] = {}
    all_two_sided = True
    for label in CORE_CLASSIFICATION_LABELS:
        pos, neg, two_sided = _two_sided(train, label)
        result[label] = {"true": pos, "false": neg, "two_sided": bool(two_sided)}
        all_two_sided = all_two_sided and two_sided
    result["all_core_two_sided"] = bool(all_two_sided)
    return result


def _continuous_degenerate(df: pd.DataFrame) -> dict[str, bool]:
    degenerate: dict[str, bool] = {}
    for head, column in CONTINUOUS_HEAD_KEYS.items():
        values = _finite(df[column]) if column in df else np.array([])
        degenerate[head] = bool(values.size == 0 or float(np.std(values)) == 0.0)
    return degenerate


def audit_train1600_learnability_v4(
    df: pd.DataFrame,
    *,
    margins: dict[str, float],
    dead_zones: dict[str, float],
    online_k_values: Iterable[int] = ONLINE_CANDIDATE_K_VALUES,
) -> dict[str, Any]:
    """Full training readiness audit (spec section 2)."""
    k_records, k_summary = k_semantics_audit(
        df, online_k_values=online_k_values
    )
    boundary_records, calibration_risk_heads = boundary_coverage(
        df, margins=margins, dead_zones=dead_zones
    )
    leakage = feature_leakage_audit()
    residuals = residual_schema_audit(df)
    locked = locked_power_report(df)
    split_leak = _split_leakage(df)
    train_core = _train_core_two_sided(df)
    degenerate = _continuous_degenerate(df)
    zero_inflation = pfv_zero_inflation(
        df, pfv_dead_zone=float(dead_zones.get("pfv", 0.0))
    )

    hard_failures: list[str] = []
    if leakage["leakage_count"] > 0:
        hard_failures.append("feature_leakage_present")
    if not split_leak["leakage_free"]:
        hard_failures.append("split_rainfall_sha_leakage")
    if not train_core["all_core_two_sided"]:
        hard_failures.append("train_core_label_single_sided")
    if not residuals["all_process_residuals_complete"]:
        hard_failures.append("process_residual_schema_incomplete")
    if any(degenerate.values()):
        hard_failures.append("continuous_label_degenerate")

    conditional_reasons: list[str] = []
    if k_summary["k1_k2_supplement_required"]:
        conditional_reasons.append("online_k1_k2_not_in_training_domain")
    if calibration_risk_heads:
        conditional_reasons.append("continuous_head_boundary_calibration_risk")
    if locked["underpowered_locked"]:
        conditional_reasons.append("underpowered_locked")

    if hard_failures:
        readiness = "fail"
    elif conditional_reasons:
        readiness = "conditional_pass"
    else:
        readiness = "pass"

    verdict = {
        "stage": "AuditTrain1600LearnabilityV4",
        "training_readiness": readiness,
        "hard_failures": hard_failures,
        "conditional_reasons": conditional_reasons,
        "calibration_risk_heads": calibration_risk_heads,
        "k1_k2_supplement_required": k_summary["k1_k2_supplement_required"],
        "disable_online_k1_k2": k_summary["disable_online_k1_k2_until_backfilled"],
        "underpowered_locked": locked["underpowered_locked"],
        "feature_leakage_count": leakage["leakage_count"],
        "split_leakage_free": split_leak["leakage_free"],
        "train_core_labels_two_sided": train_core["all_core_two_sided"],
        "process_residuals_complete": residuals["all_process_residuals_complete"],
        "training_permitted": readiness in {"pass", "conditional_pass"},
    }

    return {
        "stage": "AuditTrain1600LearnabilityV4",
        "tables": {
            "split_label_distribution": split_label_distribution(df),
            "event_label_distribution": event_label_distribution(df),
            "state_candidate_variance": state_candidate_variance(df),
            "action_coverage": action_coverage(df),
            "k_semantics_audit": k_records,
            "boundary_coverage": boundary_records,
        },
        "residual_schema_audit": residuals,
        "feature_leakage_audit": leakage,
        "covariate_shift_report": covariate_shift_report(df),
        "pfv_zero_inflation": zero_inflation,
        "k_semantics_summary": k_summary,
        "locked_power_report": locked,
        "split_leakage": split_leak,
        "train_core_two_sided": train_core,
        "continuous_degenerate": degenerate,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Section 3: model training authorization
# ---------------------------------------------------------------------------

def evaluate_model_training_authorization_v4(
    *,
    dataset_audit: dict,
    learnability: dict,
    freeze: dict,
    unresolved_files: Iterable[str] = (),
) -> dict[str, Any]:
    """Evaluate the frozen minimum conditions for model training (section 3).

    A ``conditional_pass`` readiness (e.g. online K1/K2 supplement pending)
    still authorises training; the K supplement obligation and online K1/K2
    disable are recorded as conditions, not blockers.  Model Safety Gate stays
    deferred regardless.
    """
    verdict = learnability.get("verdict", {})
    leakage = learnability.get("feature_leakage_audit", {})
    residuals = learnability.get("residual_schema_audit", {})
    train_core = learnability.get("train_core_two_sided", {})
    locked = learnability.get("locked_power_report", {})
    unresolved = sorted(str(item) for item in unresolved_files)

    conditions = {
        "dataset_gate_pass": str(dataset_audit.get("status")) == "pass",
        "feature_leakage_zero": int(leakage.get("leakage_count", 1)) == 0,
        "split_no_leakage": bool(verdict.get("split_leakage_free", False)),
        "train_core_labels_two_sided": bool(
            train_core.get("all_core_two_sided", False)
        ),
        "calibration_calibratable": bool(
            not verdict.get("hard_failures")
        ),
        "locked_continuous_regression_possible": int(
            locked.get("locked_states", 0)
        )
        > 0,
        "input_schema_complete": bool(
            residuals.get("all_process_residuals_complete", False)
        )
        and int(leakage.get("leakage_count", 1)) == 0,
        "unresolved_files_zero": len(unresolved) == 0,
        "freeze_immutable": bool(freeze.get("immutable", False)),
    }
    readiness = str(verdict.get("training_readiness", "fail"))
    readiness_ok = readiness in {"pass", "conditional_pass"}
    authorized = readiness_ok and all(conditions.values())

    return {
        "gate": "PROJECT6_V4_MODEL_TRAINING_AUTHORIZATION_V4",
        "status": "pass" if authorized else "blocked",
        "model_training_authorized": bool(authorized),
        "training_readiness": readiness,
        "conditions": conditions,
        "unresolved_files": unresolved,
        "pending_obligations": {
            "k1_k2_supplement_required": bool(
                verdict.get("k1_k2_supplement_required", False)
            ),
            "disable_online_k1_k2": bool(verdict.get("disable_online_k1_k2", False)),
            "calibration_risk_heads": list(
                verdict.get("calibration_risk_heads", [])
            ),
            "underpowered_locked": bool(verdict.get("underpowered_locked", False)),
        },
        "model_safety_gate_status": "deferred",
        "does_not_authorize": [
            "policy_lock",
            "challenge",
            "formal_blind",
        ],
    }
