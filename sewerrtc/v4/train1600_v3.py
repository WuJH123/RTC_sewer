"""Torch-free Train1600 V3 logic: gate split, planning, rotation, audits.

Implements the frozen contracts:

- docs/contracts/PROJECT6_V4_DATA_GENERATION_AUTHORIZATION_V3.json
- docs/contracts/PROJECT6_V4_MODEL_SAFETY_GATE_V3.json
- docs/contracts/PROJECT6_V4_TRAIN1600_DATASET_V3.json
- docs/contracts/PROJECT6_V4_LOCKED_VALIDATION_ACCRUAL_V3.json

This module must never import torch; it feeds SWMM planning stages only.
The P3 verdict (underpowered_validation) is read-only evidence here and is
never rewritten by any function in this module.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .pilot_candidates import checkpoint_state_sha

V3_STRATA = (
    "predicted_high_feasibility",
    "predicted_boundary",
    "predicted_fallback_likely",
)

# (v3_role, materializer_role) per stratum; the materializer only understands
# its own frozen role vocabulary, so the v3 role rides in candidate_role_v3.
V3_ROLES_BY_STRATUM: dict[str, tuple[tuple[str, str], ...]] = {
    "predicted_high_feasibility": (
        ("joint_seeking_family_a", "joint_beneficial_a"),
        ("joint_seeking_family_b", "joint_beneficial_b"),
        ("safety_boundary_pfv_tfv_peak", "pfv_boundary"),
        ("hard_negative", "TFV_hard_negative"),
        ("uncertainty_coverage_gap", "coverage_gap"),
    ),
    "predicted_boundary": (
        ("toward_di_boundary", "di_neighbourhood"),
        ("toward_no_control_boundary", "toward_no_control"),
        ("temporal_interpolation", "uncertainty"),
        ("hard_negative", "PFV_hard_negative"),
        ("uncertainty", "uncertainty"),
    ),
    "predicted_fallback_likely": (
        ("near_reference_neutral", "neutral"),
        ("pfv_safe_tfv_degraded", "TFV_hard_negative"),
        ("peak_degraded", "PFV_safe_Peak_hard_negative"),
        ("high_response_hard_negative", "PFV_hard_negative"),
        ("uncertainty_ood", "uncertainty"),
    ),
    "low_opportunity": (
        ("hold_di_near_reference", "hold_neighbourhood"),
        ("k1_strong_legal_probe", "k1_strong_legal"),
        ("k2_strong_legal_probe", "k2_strong_legal"),
        ("temporal_pulse", "temporal_pulse"),
        ("uncertainty_response_magnitude_probe", "expected_low_response"),
    ),
}

TRAIN_V3_BUDGET = {
    "target_accepted_per_state": 5,
    "initial_candidate_budget_per_state": 6,
    "maximum_candidate_budget_per_state": 10,
    "accepted_total": 1600,
}

ROUND_TARGETS_V3 = (400, 400, 400)

_HIGH_CLASSES = {"joint_feasible_robust", "joint_feasible_found"}
_FALLBACK_CLASSES = {"no_joint_found_under_budget", "no_pfv_safe_found"}
_BOUNDARY_CLASSES = {"joint_boundary_found"}

MODEL_SAFETY_DEFERRED_METRICS = (
    "balanced_accuracy",
    "mcc",
    "auprc",
    "pfv_false_safe",
    "peak_false_safe",
    "top_k_feasible_recall",
    "decision_regret",
    "uncertainty_calibration",
    "held_out_feasible_states_at_least_5",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Section 6: state stratification from online features only
# ---------------------------------------------------------------------------

def fit_state_feasibility_scorer(
    p3_map: pd.DataFrame, p3_catalog: pd.DataFrame
) -> dict:
    """Fit the frozen online-feature stratification scorer.

    Only P3 development/train (``pilot_train``) states may inform the
    thresholds; the scorer itself uses the online ``opportunity_score``
    feature and never any exact-search label of a *new* state.
    """
    required_map = {"event_id", "checkpoint_id", "split",
                    "state_feasibility_class"}
    missing = required_map - set(p3_map)
    if missing:
        raise ValueError(f"p3 map missing columns: {sorted(missing)}")
    if "opportunity_score" not in p3_catalog:
        raise ValueError("p3 catalog missing opportunity_score")
    joined = p3_map.merge(
        p3_catalog[["event_id", "checkpoint_id", "opportunity_score"]]
        .drop_duplicates(["event_id", "checkpoint_id"]),
        on=["event_id", "checkpoint_id"],
        how="inner",
    )
    train = joined[joined["split"].astype(str) == "pilot_train"].copy()
    if train.empty:
        raise ValueError("no pilot_train states available to fit scorer")
    scores = pd.to_numeric(train["opportunity_score"], errors="coerce")
    classes = train["state_feasibility_class"].astype(str)
    high = scores[classes.isin(_HIGH_CLASSES)].dropna()
    fallback = scores[classes.isin(_FALLBACK_CLASSES)].dropna()
    method = "class_median"
    if len(high) and len(fallback):
        t_high = float(high.median())
        t_low = float(fallback.median())
    else:
        t_high = float("nan")
        t_low = float("nan")
    if not np.isfinite(t_high) or not np.isfinite(t_low) or t_high <= t_low:
        method = "tertile_fallback"
        clean = scores.dropna()
        t_low = float(clean.quantile(1.0 / 3.0))
        t_high = float(clean.quantile(2.0 / 3.0))
    return {
        "scorer_version": "v3",
        "feature": "opportunity_score",
        "online_features_only": True,
        "training_split": "pilot_train",
        "training_states": int(len(train)),
        "method": method,
        "t_high": t_high,
        "t_low": t_low,
        "strata": list(V3_STRATA),
    }


def apply_state_feasibility_scorer(
    scorer: dict, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Assign predicted strata using online features only (never exact
    labels of the states being stratified)."""
    frame = catalog.copy()
    t_high = float(scorer["t_high"])
    t_low = float(scorer["t_low"])
    scores = pd.to_numeric(
        frame.get("opportunity_score", np.nan), errors="coerce"
    ).fillna(t_low)
    strata: list[str] = []
    for role, score in zip(
        frame["checkpoint_role"].astype(str), scores.astype(float)
    ):
        if role == "low_opportunity":
            strata.append("low_opportunity")
        elif score >= t_high:
            strata.append("predicted_high_feasibility")
        elif score <= t_low:
            strata.append("predicted_fallback_likely")
        else:
            strata.append("predicted_boundary")
    frame["predicted_stratum"] = strata
    return frame


# ---------------------------------------------------------------------------
# Section 7: role plan (320 states x 10 candidates = 3200 rows)
# ---------------------------------------------------------------------------

def build_v3_role_plan(stratified_catalog: pd.DataFrame) -> pd.DataFrame:
    """Materializable role plan: 5 primary + 5 state-reserve rows per state.

    ``candidate_role`` carries the frozen materializer vocabulary;
    ``candidate_role_v3`` carries the contract role. Reserve rows repeat the
    stratum menu under new case ids (new seeds) and are only consumed as
    ``state_reserve_candidate`` replenishment inside the 10-candidate budget.
    """
    required = {
        "event_id", "rainfall_sha256", "checkpoint_id", "checkpoint_role",
        "checkpoint_min", "split", "predicted_stratum",
    }
    missing = required - set(stratified_catalog)
    if missing:
        raise ValueError(f"stratified catalog missing: {sorted(missing)}")
    rows: list[dict] = []
    for _, checkpoint in stratified_catalog.drop_duplicates(
        ["event_id", "checkpoint_id"]
    ).iterrows():
        stratum = str(checkpoint["predicted_stratum"])
        if stratum not in V3_ROLES_BY_STRATUM:
            raise ValueError(f"unknown stratum: {stratum}")
        menu = V3_ROLES_BY_STRATUM[stratum]
        state_id = checkpoint_state_sha(checkpoint)
        for idx in range(10):
            v3_role, mat_role = menu[idx % 5]
            case_id = (
                f"t16v3__{checkpoint['event_id']}__"
                f"{checkpoint['checkpoint_id']}__{v3_role}__{idx}"
            )
            rows.append(
                {
                    "event_id": checkpoint["event_id"],
                    "rainfall_sha256": checkpoint["rainfall_sha256"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_role": checkpoint["checkpoint_role"],
                    "state_id": state_id,
                    "candidate_role": mat_role,
                    "candidate_role_v3": v3_role,
                    "predicted_stratum": stratum,
                    "split": checkpoint["split"],
                    "case_id": case_id,
                    "priority": idx,
                    "plan_tier": "primary" if idx < 5 else "state_reserve",
                    "replenish_source": (
                        "" if idx < 5 else "state_reserve_candidate"
                    ),
                    "source_anchor_role": (
                        "peak_boundary_anchor"
                        if mat_role
                        in ("peak_boundary", "PFV_safe_Peak_hard_negative")
                        else "checkpoint_anchor"
                    ),
                }
            )
    plan = pd.DataFrame(rows)
    if plan["case_id"].duplicated().any():
        raise ValueError("v3 role plan has duplicate case ids")
    per_state = plan.groupby(["event_id", "checkpoint_id"]).size()
    if not per_state.eq(10).all():
        raise ValueError("every state needs exactly 10 planned candidates")
    return plan


# ---------------------------------------------------------------------------
# Section 9: three-round rotation (240 states, 3 x 400, 5 per state)
# ---------------------------------------------------------------------------

def build_train_round_rotation_v3(
    train_catalog: pd.DataFrame, *, expected_states: int | None = 240
) -> pd.DataFrame:
    """Rest-round rotation: each state rests (1 candidate) in exactly one
    round and takes 2 extras in each of the other two rounds -> 5 total."""
    states = train_catalog[
        train_catalog["split"].astype(str) == "train"
    ].drop_duplicates(["event_id", "checkpoint_id"]).copy()
    states = states.sort_values(["event_id", "checkpoint_id"]).reset_index(
        drop=True
    )
    if expected_states is not None and len(states) != expected_states:
        raise ValueError(
            f"rotation expects {expected_states} train states, got "
            f"{len(states)}"
        )
    events = sorted(states["event_id"].astype(str).unique())
    ordinal = {event: index for index, event in enumerate(events)}
    rows: list[dict] = []
    for event in events:
        block = states[states["event_id"].astype(str) == event]
        for pos, (_, state) in enumerate(block.iterrows()):
            rest = (pos + ordinal[event]) % 3
            targets = {
                round_index: (1 if round_index == rest else 2)
                for round_index in (0, 1, 2)
            }
            rows.append(
                {
                    "event_id": state["event_id"],
                    "checkpoint_id": state["checkpoint_id"],
                    "state_id": checkpoint_state_sha(state),
                    "split": "train",
                    "pos_in_event": pos,
                    "event_ordinal": ordinal[event],
                    "rest_round": rest,
                    "round0_target": targets[0],
                    "round1_target": targets[1],
                    "round2_target": targets[2],
                    "total_target": 5,
                }
            )
    return pd.DataFrame(rows)


def verify_round_rotation_v3(
    rotation: pd.DataFrame, *, expected_states: int = 240
) -> dict:
    """Frozen arithmetic checks for the 3 x 400 rotation."""
    totals = {
        round_index: int(rotation[f"round{round_index}_target"].sum())
        for round_index in (0, 1, 2)
    }
    per_state = (
        rotation["round0_target"]
        + rotation["round1_target"]
        + rotation["round2_target"]
    )
    extra_rounds = sum(
        (rotation[f"round{round_index}_target"] == 2).astype(int)
        for round_index in (0, 1, 2)
    )
    expected_round_total = expected_states * 5 // 3
    checks = {
        "states_expected": len(rotation) == expected_states,
        "each_round_target_equal": len(set(totals.values())) == 1,
        "each_round_target_expected": all(
            value == expected_round_total for value in totals.values()
        ),
        "each_state_total_5": bool(per_state.eq(5).all()),
        "each_state_exactly_2_extra_rounds": bool(extra_rounds.eq(2).all()),
        "each_state_exactly_1_rest_round": bool(
            (3 - extra_rounds).eq(1).all()
        ),
        "grand_total_expected": int(per_state.sum())
        == expected_states * 5,
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "round_totals": totals,
        "grand_total": int(per_state.sum()),
    }


def assign_primary_candidates_to_rounds_v3(
    role_plan: pd.DataFrame, rotation: pd.DataFrame
) -> pd.DataFrame:
    """Deterministically split each train state's 5 primary candidates over
    the three rounds according to its rotation targets."""
    primary = role_plan[
        (role_plan["plan_tier"] == "primary")
        & (role_plan["split"].astype(str) == "train")
    ].copy()
    targets = {
        (str(row["event_id"]), str(row["checkpoint_id"])): [
            int(row["round0_target"]),
            int(row["round1_target"]),
            int(row["round2_target"]),
        ]
        for _, row in rotation.iterrows()
    }
    rounds: list[int] = []
    consumed: dict[tuple[str, str], int] = {}
    primary = primary.sort_values(
        ["event_id", "checkpoint_id", "priority"]
    ).reset_index(drop=True)
    for _, row in primary.iterrows():
        key = (str(row["event_id"]), str(row["checkpoint_id"]))
        if key not in targets:
            raise ValueError(f"state missing from rotation: {key}")
        position = consumed.get(key, 0)
        cuts = targets[key]
        if position < cuts[0]:
            rounds.append(0)
        elif position < cuts[0] + cuts[1]:
            rounds.append(1)
        else:
            rounds.append(2)
        consumed[key] = position + 1
    primary["round"] = rounds
    return primary


# ---------------------------------------------------------------------------
# Section 2: Data Generation Authorization V3 / Model Safety Gate V3
# ---------------------------------------------------------------------------

def evaluate_data_generation_authorization_v3(
    map_audit: dict, gate_verdict: dict, dataset_audit: dict
) -> dict:
    """The 12 frozen pass conditions of the Data Generation Authorization.

    Reads the frozen P3 evidence only; the P3 verdict is quoted verbatim and
    never rewritten. no_joint_found_under_budget is reported as-is and never
    reinterpreted as physical infeasibility.
    """
    map_checks = map_audit.get("checks", {})
    ds_checks = dataset_audit.get("checks", {})
    recall = map_audit.get("recall_report", {})
    class_counts = map_audit.get("class_counts", {})
    fallback_states = recall.get("fallback_only_states", [])
    fallback_events = {
        str(item.get("event_id")) for item in fallback_states
    }
    robust = int(class_counts.get("joint_feasible_robust", 0)) + int(
        class_counts.get("joint_feasible_found", 0)
    )
    ridge = float(
        gate_verdict.get("metrics_used", {}).get(
            "ridge_rmse_improvement_vs_zero", float("nan")
        )
    )
    verdict_checks = gate_verdict.get("checks", {})
    conditions = {
        "hard_authenticity_100pct": bool(
            map_checks.get("hard_authenticity_all_true")
        )
        and bool(ds_checks.get("hard_authenticity_100pct")),
        "exact_feasibility_map_complete": bool(
            map_checks.get("all_catalog_states_classified")
        )
        and bool(map_checks.get("accounting_closed"))
        and bool(map_checks.get("missing_zero")),
        "unresolved_states_zero": bool(
            map_checks.get("execution_unresolved_zero")
        ),
        "robust_feasible_states_at_least_8": robust >= 8,
        "feasible_event_support_at_least_3": int(
            recall.get("event_support", 0)
        )
        >= 3,
        "fallback_only_states_at_least_8": len(fallback_states) >= 8,
        "fallback_event_support_at_least_3": len(fallback_events) >= 3,
        "candidate_generator_recall_at_least_0p80": float(
            recall.get("candidate_generator_state_recall", 0.0)
        )
        >= 0.80,
        "missed_feasible_states_within_frozen_cap": len(
            recall.get("missed_feasible_states", [1])
        )
        <= 0,
        "continuous_delta_improvement_at_least_10pct": bool(
            np.isfinite(ridge) and ridge >= 0.10
        ),
        "no_event_or_rainfall_sha_leakage": bool(
            ds_checks.get("rainfall_sha_split_isolated")
        )
        and bool(ds_checks.get("no_eval_split_search_rows_trainable")),
        "candidate_reference_execution_chain_complete": bool(
            map_checks.get("replay_success_rate_100")
        )
        and bool(map_checks.get("actual_duplicates_zero"))
        and bool(map_checks.get("accounting_closed")),
    }
    scientific_pass = all(conditions.values())
    return {
        "gate": "PROJECT6_V4_DATA_GENERATION_AUTHORIZATION_V3",
        "status": "pass" if scientific_pass else "blocked",
        "scientific_pass": scientific_pass,
        "train1600_planning_authorized": scientific_pass,
        "conditions": conditions,
        "values": {
            "robust_feasible_states": robust,
            "boundary_states": int(
                class_counts.get("joint_boundary_found", 0)
            ),
            "fallback_only_states": len(fallback_states),
            "fallback_event_support": len(fallback_events),
            "feasible_event_support": int(recall.get("event_support", 0)),
            "candidate_generator_recall": float(
                recall.get("candidate_generator_state_recall", 0.0)
            ),
            "missed_feasible_states": len(
                recall.get("missed_feasible_states", [])
            ),
            "ridge_rmse_improvement_vs_zero": ridge,
            "positive_control_replay_checks": {
                key: bool(value)
                for key, value in verdict_checks.items()
                if "replay" in key or "positive_control" in key
            },
        },
        "p3_verdict_preserved": str(gate_verdict.get("status", "")),
        "p3_verdict_never_overwritten": True,
        "no_joint_found_under_budget_kept_verbatim": True,
    }


def model_safety_gate_v3_status() -> dict:
    """Model Safety Gate V3 is deferred until a powered Locked Validation."""
    return {
        "gate": "PROJECT6_V4_MODEL_SAFETY_GATE_V3",
        "status": "deferred",
        "reason": "requires_train1600_and_powered_locked_validation",
        "deferred_metrics": list(MODEL_SAFETY_DEFERRED_METRICS),
        "controls": [
            "policy_lock",
            "surrogate_closed_loop",
            "challenge",
            "formal_blind",
        ],
        "does_not_control": ["train1600_data_generation"],
        "insufficient_positives_never_interpreted_as_pass": True,
    }


# ---------------------------------------------------------------------------
# Section 4/5: plan audit
# ---------------------------------------------------------------------------

def audit_train1600_plan_v3(
    train_catalog: pd.DataFrame,
    reserve_catalog: pd.DataFrame,
    selection: dict[str, list[str]],
    role_plan: pd.DataFrame,
    rotation: pd.DataFrame,
    *,
    plan_freeze: dict | None = None,
) -> dict:
    """Mechanical audit of the frozen Train1600 V3 plan (spec sections 4-9)."""
    from .training_plan import audit_train1600_plan

    base = audit_train1600_plan(train_catalog, reserve_catalog, selection)
    rotation_report = verify_round_rotation_v3(
        rotation, expected_states=len(rotation)
    )
    primary = role_plan[role_plan["plan_tier"] == "primary"]
    per_state = role_plan.groupby(["event_id", "checkpoint_id"]).size()
    per_state_primary = primary.groupby(
        ["event_id", "checkpoint_id"]
    ).size()
    low = train_catalog[
        train_catalog["checkpoint_role"].astype(str) == "low_opportunity"
    ]
    strata_train = set(
        role_plan[role_plan["split"].astype(str) == "train"][
            "predicted_stratum"
        ]
        .astype(str)
        .unique()
    ) - {"low_opportunity"}
    freeze = plan_freeze or {}
    checks = {
        "role_plan_rows_10_per_state": bool(per_state.eq(10).all()),
        "role_plan_total_rows": len(role_plan)
        == 10 * len(per_state),
        "primary_rows_5_per_state": bool(per_state_primary.eq(5).all()),
        "case_ids_unique": not role_plan["case_id"].duplicated().any(),
        "rotation_arithmetic": rotation_report["status"] == "pass",
        "rotation_states_match_train_split": len(rotation)
        == int(
            train_catalog[train_catalog["split"].astype(str) == "train"]
            .drop_duplicates(["event_id", "checkpoint_id"])
            .shape[0]
        ),
        "low_opportunity_one_per_event": bool(
            low.groupby("event_id").size().eq(1).all()
        )
        and low["event_id"].nunique()
        == train_catalog["event_id"].nunique(),
        "stratification_online_features_only": "state_feasibility_class"
        not in set(role_plan.columns),
        "calibration_plan_sha_frozen": bool(
            freeze.get("calibration_plan_sha256")
        ),
        "locked_validation_plan_sha_frozen": bool(
            freeze.get("locked_validation_plan_sha256")
        ),
        "reserve_rows_are_state_reserve_candidates": bool(
            (
                role_plan[role_plan["plan_tier"] == "state_reserve"][
                    "replenish_source"
                ]
                == "state_reserve_candidate"
            ).all()
        ),
    }
    informational = {
        "train_strata_present": sorted(strata_train),
        "train_strata_count": len(strata_train),
        "fallback_likely_states_never_forced_joint": True,
    }
    status = (
        "pass"
        if base["status"] == "pass" and all(checks.values())
        else "blocked"
    )
    return {
        "status": status,
        "base_plan_audit": base,
        "checks": checks,
        "rotation_report": rotation_report,
        "informational": informational,
    }


# ---------------------------------------------------------------------------
# Section 8: per-state budget / progress
# ---------------------------------------------------------------------------

def build_per_state_progress_v3(
    role_plan: pd.DataFrame, accepted: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-state accepted-vs-budget ledger (accepted counts actual-unique)."""
    states = role_plan.drop_duplicates(["event_id", "checkpoint_id"])[
        ["event_id", "checkpoint_id", "state_id", "split",
         "predicted_stratum"]
    ].copy()
    counts: dict[tuple[str, str], int] = {}
    uniques: dict[tuple[str, str], set[str]] = {}
    if accepted is not None and len(accepted):
        for _, row in accepted.iterrows():
            key = (str(row["event_id"]), str(row["checkpoint_id"]))
            sha = str(
                row.get(
                    "actual_schedule_sha256",
                    row.get("projected_schedule_sha256", ""),
                )
            )
            seen = uniques.setdefault(key, set())
            if sha and sha in seen:
                continue
            seen.add(sha)
            counts[key] = counts.get(key, 0) + 1
    states["accepted"] = [
        counts.get((str(row["event_id"]), str(row["checkpoint_id"])), 0)
        for _, row in states.iterrows()
    ]
    states["target_accepted"] = TRAIN_V3_BUDGET["target_accepted_per_state"]
    states["initial_budget"] = TRAIN_V3_BUDGET[
        "initial_candidate_budget_per_state"
    ]
    states["maximum_budget"] = TRAIN_V3_BUDGET[
        "maximum_candidate_budget_per_state"
    ]
    states["remaining_to_target"] = (
        states["target_accepted"] - states["accepted"]
    ).clip(lower=0)
    states["state_shortfall"] = False
    return states.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Active Learning (rule-based, train split only, torch-free)
# ---------------------------------------------------------------------------

def assert_train_split_only(frame: pd.DataFrame, *, column: str = "split"):
    """Fail closed if any non-train rows reach the active learner."""
    if column not in frame:
        raise ValueError("active learner input has no split column")
    values = set(frame[column].astype(str).unique())
    if values - {"train"}:
        raise ValueError(
            "active learner must only read the Train split, got "
            f"{sorted(values)}"
        )


_AL_ROLE_PRIORITY = (
    "hard_negative",
    "uncertainty",
    "uncertainty_ood",
    "uncertainty_coverage_gap",
    "temporal_interpolation",
    "safety_boundary_pfv_tfv_peak",
    "toward_di_boundary",
    "toward_no_control_boundary",
)


def rank_remaining_candidates_v3(
    remaining_plan: pd.DataFrame,
    accepted_train: pd.DataFrame,
    progress: pd.DataFrame,
) -> pd.DataFrame:
    """Rank not-yet-run candidates: boundary/uncertainty/hard-negative and
    coverage gaps first; states that already reached 5 accepted are dropped.

    ``accepted_train`` must contain only Train-split rows (asserted); the
    Calibration and Locked Validation splits are never readable here.
    """
    if len(accepted_train):
        assert_train_split_only(accepted_train)
    full = {
        (str(row["event_id"]), str(row["checkpoint_id"]))
        for _, row in progress.iterrows()
        if int(row["accepted"]) >= int(row["target_accepted"])
    }
    ranked = remaining_plan[
        remaining_plan["split"].astype(str) == "train"
    ].copy()
    ranked = ranked[
        ~ranked.apply(
            lambda row: (str(row["event_id"]), str(row["checkpoint_id"]))
            in full,
            axis=1,
        )
    ]
    family_counts = (
        accepted_train.get("candidate_family", pd.Series(dtype=str))
        .astype(str)
        .value_counts()
        .to_dict()
        if len(accepted_train)
        else {}
    )
    role_rank = {
        role: index for index, role in enumerate(_AL_ROLE_PRIORITY)
    }
    ranked["al_role_rank"] = [
        role_rank.get(str(role), len(_AL_ROLE_PRIORITY))
        for role in ranked["candidate_role_v3"].astype(str)
    ]
    ranked["al_family_seen"] = [
        family_counts.get(str(role), 0)
        for role in ranked.get(
            "candidate_family", ranked["candidate_role"]
        ).astype(str)
    ]
    ranked = ranked.sort_values(
        ["al_role_rank", "al_family_seen", "event_id", "checkpoint_id",
         "priority"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    ranked["al_rank"] = range(len(ranked))
    return ranked


def _interleave_round_plan(selected: pd.DataFrame) -> pd.DataFrame:
    """Round-robin the selected rows across candidate families so that any
    completed prefix spans multiple families (progressive partial audits stay
    meaningful). This only reorders rows; the selected set is byte-identical,
    so the Active-Learning selection itself is unchanged."""
    if len(selected) <= 1:
        return selected.reset_index(drop=True)
    ordered = selected.reset_index(drop=True)
    ordered["_seq"] = range(len(ordered))
    if "candidate_family" in ordered.columns:
        family_values = ordered["candidate_family"].astype(str).values
    else:
        family_values = ["_"] * len(ordered)
    # Rank within each family preserves the incoming (AL-rank) priority order;
    # sorting by that rank first yields a family round-robin.
    ordered["_within_family"] = ordered.groupby(family_values).cumcount()
    result = ordered.sort_values(["_within_family", "_seq"], kind="stable")
    return result.drop(
        columns=["_seq", "_within_family"]
    ).reset_index(drop=True)


def select_round_candidates_v3(
    role_plan: pd.DataFrame,
    rotation: pd.DataFrame,
    round_index: int,
    progress: pd.DataFrame,
    used_case_ids: set[str],
    *,
    ranking: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select this round's candidates for every train state from the frozen
    pool (primary first, then state-reserve within the 10-candidate budget).
    States already at 5 accepted are skipped; nothing is ever copied."""
    target_column = f"round{round_index}_target"
    remaining_targets = {
        (str(row["event_id"]), str(row["checkpoint_id"])): int(
            row[target_column]
        )
        for _, row in rotation.iterrows()
    }
    full = {
        (str(row["event_id"]), str(row["checkpoint_id"]))
        for _, row in progress.iterrows()
        if int(row["accepted"]) >= int(row["target_accepted"])
    }
    pool = role_plan[
        (role_plan["split"].astype(str) == "train")
        & (~role_plan["case_id"].astype(str).isin(used_case_ids))
    ].copy()
    if ranking is not None and "al_rank" in ranking:
        rank_map = {
            str(row["case_id"]): int(row["al_rank"])
            for _, row in ranking.iterrows()
        }
        pool["order"] = [
            rank_map.get(str(case), 10_000 + int(priority))
            for case, priority in zip(pool["case_id"], pool["priority"])
        ]
    else:
        pool["order"] = pool["priority"].astype(int)
    pool = pool.sort_values(
        ["order", "case_id"]
    ).sort_values(
        "plan_tier",
        key=lambda column: column.map({"primary": 0, "state_reserve": 1}),
        kind="stable",
    )
    selected_rows: list[pd.Series] = []
    taken: dict[tuple[str, str], int] = {}
    for _, row in pool.iterrows():
        key = (str(row["event_id"]), str(row["checkpoint_id"]))
        if key in full:
            continue
        needed = remaining_targets.get(key, 0)
        if taken.get(key, 0) >= needed:
            continue
        selected_rows.append(row)
        taken[key] = taken.get(key, 0) + 1
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(selected):
        selected = selected.drop(columns=["order"], errors="ignore")
        selected["round"] = round_index
        selected = _interleave_round_plan(selected)
    return selected


# ---------------------------------------------------------------------------
# Section 12/13: round and final dataset audits
# ---------------------------------------------------------------------------

SAMPLED_ONLY_LABEL_COLUMNS = (
    "state_feasibility_label_source",
    "state_feasibility_label_validity",
    "candidate_search_budget",
    "exact_search_performed",
    "joint_found_in_sampled_set",
)
SAMPLED_ONLY_TOKEN = "sampled_only"
# Verdicts a P3 Exact search may assign; a plain sampled-only Train1600 state
# must never wear any of them (spec section 1 rules 1-2 and 5).
FORBIDDEN_SAMPLED_ONLY_VERDICTS = frozenset(
    {
        "fallback_only_under_budget",
        "intervention_feasible_found",
        "boundary_or_uncertain",
        "physically_infeasible",
        "physical_infeasible",
        "impossible",
        "uncontrollable",
    }
)


def _forbidden_feasibility_label_present(samples: pd.DataFrame) -> bool:
    """True when a sampled-only row wears a P3-Exact-only feasibility verdict."""
    for column in (
        "state_feasibility_label_source",
        "state_feasibility_label_validity",
    ):
        if column in samples and samples[column].astype(str).isin(
            FORBIDDEN_SAMPLED_ONLY_VERDICTS
        ).any():
            return True
    return False


def _sampled_only_contract_valid(samples: pd.DataFrame) -> bool:
    """Final-gate check: 5 fields present with correct sampled-only meaning."""
    if not len(samples):
        return False
    if any(column not in samples for column in SAMPLED_ONLY_LABEL_COLUMNS):
        return False
    source = samples["state_feasibility_label_source"].astype(str)
    validity = samples["state_feasibility_label_validity"].astype(str)
    exact = samples["exact_search_performed"].astype(bool)
    budget = pd.to_numeric(
        samples["candidate_search_budget"], errors="coerce"
    )
    return bool(
        source.eq(SAMPLED_ONLY_TOKEN).all()
        and validity.eq(SAMPLED_ONLY_TOKEN).all()
        and (~exact).all()
        and budget.notna().all()
        and (budget > 0).all()
        and not _forbidden_feasibility_label_present(samples)
    )


def audit_round_dataset_v3(
    samples: pd.DataFrame,
    accounting: dict,
    *,
    stage: str,
    accepted_target: int,
    hard_columns: tuple[str, ...],
    reference_cache_sha256: str = "",
    expected_reference_cache_sha256: str = "",
) -> dict:
    """Hard + informational partial/round audit (spec section 12)."""
    hard = {
        "accounting_closed": bool(accounting.get("accounting_closed")),
        "accepted_target_met": int(accounting.get("accepted", 0))
        >= accepted_target,
        "actual_duplicates_zero": int(
            accounting.get("actual_duplicates", 0)
        )
        == 0,
        "hard_authenticity_100pct": bool(
            len(samples)
            and all(
                column in samples
                and samples[column].astype(bool).all()
                for column in hard_columns
            )
        ),
        "reference_cache_sha_consistent": (
            not expected_reference_cache_sha256
            or reference_cache_sha256 == expected_reference_cache_sha256
        ),
        "no_forbidden_sampled_only_label": not (
            _forbidden_feasibility_label_present(samples)
        ),
    }
    delta_columns = [
        column
        for column in samples.columns
        if str(column).startswith("delta_")
        and pd.api.types.is_numeric_dtype(samples[column])
    ]
    informational = {
        "continuous_deltas_not_all_constant": bool(
            any(
                samples[column].dropna().nunique() > 1
                for column in delta_columns
            )
        )
        if delta_columns
        else False,
        "events_covered": int(samples["event_id"].nunique())
        if "event_id" in samples
        else 0,
        "states_covered": int(
            samples.drop_duplicates(["event_id", "checkpoint_id"]).shape[0]
        )
        if {"event_id", "checkpoint_id"} <= set(samples.columns)
        else 0,
        "families_covered": int(
            samples.get("candidate_family", pd.Series(dtype=str)).nunique()
        ),
        "per_state_joint_candidate_not_required": True,
        "sampled_only_labels_present": all(
            column in samples for column in SAMPLED_ONLY_LABEL_COLUMNS
        ),
        "sampled_only_label_validity_ok": bool(
            "state_feasibility_label_validity" not in samples
            or samples["state_feasibility_label_validity"]
            .astype(str)
            .eq(SAMPLED_ONLY_TOKEN)
            .all()
        ),
    }
    status = "pass" if all(hard.values()) else "blocked"
    return {
        "stage": stage,
        "status": status,
        "accepted": int(accounting.get("accepted", 0)),
        "accepted_target": accepted_target,
        "hard_checks": hard,
        "informational": informational,
        "accounting": accounting,
    }


def audit_train1600_dataset_v3(
    samples: pd.DataFrame,
    train_catalog: pd.DataFrame,
    selection: dict[str, list[str]],
    *,
    hard_columns: tuple[str, ...] = (),
) -> dict:
    """Final 1600-sample quality gate (spec section 13).

    Calibration/Locked are never required to contain 5 feasible states at
    generation time; that requirement belongs to the deferred Model Safety
    Gate and the pre-registered accrual rules.
    """
    split_of_event = {}
    for split in ("train", "calibration", "locked_validation", "reserve"):
        for event in selection.get(split, []):
            split_of_event[str(event)] = split
    sample_events = samples["event_id"].astype(str)
    split_series = sample_events.map(split_of_event)
    per_state = samples.groupby(["event_id", "checkpoint_id"]).size()
    unique_col = (
        "actual_schedule_sha256"
        if "actual_schedule_sha256" in samples
        else "projected_schedule_sha256"
    )
    actual_unique = (
        samples.groupby(["event_id", "checkpoint_id"])[unique_col]
        .nunique()
        .eq(per_state)
        .all()
        if unique_col in samples
        else False
    )
    sha_splits = (
        samples.assign(_split=split_series)
        .groupby("rainfall_sha256")["_split"]
        .nunique()
        if "rainfall_sha256" in samples
        else pd.Series(dtype=int)
    )
    split_counts = split_series.value_counts().to_dict()
    stratum_column = (
        "state_feasibility_class"
        if "state_feasibility_class" in samples
        else "predicted_stratum"
    )
    train_rows = samples[split_series == "train"]
    strata_support = {}
    if stratum_column in samples:
        strata_events = train_rows.groupby(
            train_rows[stratum_column].astype(str)
        )["event_id"].nunique()
        strata_support = {
            str(key): int(value) for key, value in strata_events.items()
        }
    role_v3 = samples.get("candidate_role_v3", pd.Series(dtype=str)).astype(
        str
    )
    delta_columns = [
        column
        for column in samples.columns
        if str(column).startswith("delta_")
        and pd.api.types.is_numeric_dtype(samples[column])
    ]
    checks = {
        "accepted_total_1600": len(samples) == 1600,
        "events_64": samples["event_id"].nunique() == 64,
        "state_groups_320": len(per_state) == 320,
        "split_48_8_8": (
            len(selection.get("train", [])) == 48
            and len(selection.get("calibration", [])) == 8
            and len(selection.get("locked_validation", [])) == 8
        ),
        "train_samples_1200": int(split_counts.get("train", 0)) == 1200,
        "calibration_samples_200": int(
            split_counts.get("calibration", 0)
        )
        == 200,
        "locked_validation_samples_200": int(
            split_counts.get("locked_validation", 0)
        )
        == 200,
        "reserve_not_in_main_table": int(split_counts.get("reserve", 0))
        == 0,
        "per_state_exactly_5": bool(per_state.eq(5).all()),
        "per_state_actual_unique": bool(actual_unique),
        "no_rainfall_sha_leakage": bool(
            len(sha_splits) == 0 or sha_splits.eq(1).all()
        ),
        "hard_authenticity_100pct": bool(
            len(samples)
            and all(
                column in samples
                and samples[column].astype(bool).all()
                for column in hard_columns
            )
        )
        if hard_columns
        else True,
        "continuous_labels_non_degenerate": bool(
            any(
                samples[column].dropna().nunique() > 1
                for column in delta_columns
            )
        )
        if delta_columns
        else False,
        "hard_negatives_present": bool(
            role_v3.str.contains("hard_negative").any()
        ),
        "uncertainty_coverage_gap_present": bool(
            role_v3.str.contains("uncertainty").any()
        ),
        "sampled_only_label_contract_valid": _sampled_only_contract_valid(
            samples
        ),
    }
    informational = {
        "strata_event_support_train": strata_support,
        "boundary_scarcity_reported": strata_support.get(
            "predicted_boundary", strata_support.get(
                "joint_boundary_found", 0
            )
        )
        < 3,
        "calibration_locked_not_required_to_contain_5_feasible_states": True,
    }
    status = "pass" if all(checks.values()) else "blocked"
    return {
        "status": status,
        "checks": checks,
        "split_sample_counts": {
            key: int(value) for key, value in split_counts.items()
        },
        "informational": informational,
    }


# ---------------------------------------------------------------------------
# Section 10: locked validation accrual
# ---------------------------------------------------------------------------

def validate_accrual_plan_v3(
    contract: dict,
    *,
    initial_event_ids: list[str],
    accrual_batches: list[list[str]],
    deleted_events: list[str] | None = None,
    model_modified: bool = False,
    thresholds_modified: bool = False,
) -> dict:
    """Pre-registered accrual validation: power only, never retraining."""
    rules = contract.get("rules", {})
    maxima = rules.get("6_frozen_maxima", {})
    allowed_sizes = set(
        rules.get("4_when_exact_feasible_states_below_5", {}).get(
            "accrue_new_events_batch_size", [4, 8]
        )
    )
    total_events = len(initial_event_ids) + sum(
        len(batch) for batch in accrual_batches
    )
    all_ids = list(initial_event_ids) + [
        event for batch in accrual_batches for event in batch
    ]
    checks = {
        "initial_events_8": len(initial_event_ids)
        == int(rules.get("1_initial_events", 8)),
        "no_deleted_original_events": not (deleted_events or []),
        "model_not_modified": not model_modified,
        "thresholds_not_modified": not thresholds_modified,
        "batch_sizes_allowed": all(
            len(batch) in allowed_sizes for batch in accrual_batches
        ),
        "max_batches_respected": len(accrual_batches)
        <= int(maxima.get("maximum_accrual_batches", 2)),
        "max_total_events_respected": total_events
        <= int(maxima.get("maximum_total_locked_events", 24)),
        "no_duplicate_events": len(all_ids) == len(set(all_ids)),
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "total_locked_events": total_events,
        "purpose": "statistical_power_only",
    }


# ---------------------------------------------------------------------------
# Section 1: P3 freeze payload
# ---------------------------------------------------------------------------

def build_p3_freeze_payload(
    *,
    verdict: dict,
    map_audit: dict,
    dataset_audit: dict,
    file_manifest: dict[str, str],
    reference_cache_sha256: str,
    code_sha256: str,
) -> dict:
    """Immutable freeze record for the Gate P3 evidence archive."""
    recall = map_audit.get("recall_report", {})
    class_counts = map_audit.get("class_counts", {})
    if str(verdict.get("status")) != "underpowered_validation":
        raise ValueError(
            "P3 freeze requires the preserved underpowered_validation "
            f"verdict, got {verdict.get('status')!r}"
        )
    return {
        "freeze_id": "pilot_feasibility_p3_freeze",
        "verdict": "underpowered_validation",
        "data_generation_evidence_pass": True,
        "model_safety_evaluation_underpowered": True,
        "robust_feasible_states": int(
            class_counts.get("joint_feasible_robust", 0)
        ),
        "fallback_only_states": len(
            recall.get("fallback_only_states", [])
        ),
        "boundary_states": int(
            class_counts.get("joint_boundary_found", 0)
        ),
        "candidate_generator_recall": float(
            recall.get("candidate_generator_state_recall", 0.0)
        ),
        "unresolved": int(
            map_audit.get("p3_gate", {}).get("unresolved_states", -1)
        ),
        "immutable": True,
        "dataset_total_samples": int(
            dataset_audit.get("headline", {}).get("total_samples", 0)
        ),
        "code_sha256": code_sha256,
        "reference_cache_sha256": reference_cache_sha256,
        "file_sha256": dict(sorted(file_manifest.items())),
        "file_count": len(file_manifest),
    }
