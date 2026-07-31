"""Gate P3 feasibility map: state classification, recall and hard audit.

Classification vocabulary is frozen by the P3 contract; a state that
exhausts the frozen budget without a joint candidate is
``no_joint_found_under_budget`` -- never "physically infeasible",
"impossible" or "uncontrollable" (limited-search semantics only).
"""
from __future__ import annotations

import json

import pandas as pd

from .pilot_extension import _bool
from .pilot_feasibility_search import (
    FEASIBILITY_PHASE,
    ROUND_A,
    ROUND_B,
    ROUND_B_MAX_PER_STATE,
    TOTAL_BUDGET_PER_MISSING_STATE,
)

STATE_CLASSES = (
    "joint_feasible_found",
    "joint_feasible_robust",
    "joint_boundary_found",
    "no_joint_found_under_budget",
    "no_pfv_safe_found",
    "execution_unresolved",
)

FORBIDDEN_CLASS_TERMS = ("physically", "infeasible", "impossible", "uncontrollable")

_DELTA_PFV = "delta_pfv_h120_vs_no_control"
_DELTA_TFV = "delta_tfv_h120_vs_dynamic_internal"
_DELTA_PEAK = "delta_peak_h120_vs_dynamic_internal"
_STATE_KEYS = ["event_id", "checkpoint_id"]

# Deterministic rainfall-phase proxy over episode minutes (H120 windows sit
# inside 0-300 min events); used for recall reporting only, never gating.
_PHASE_EDGES = ((0.0, 120.0, "early"), (120.0, 240.0, "mid"))


def _rainfall_phase(checkpoint_min: float) -> str:
    for low, high, name in _PHASE_EDGES:
        if low <= checkpoint_min < high:
            return name
    return "late"


def combine_state_samples(
    v2_samples: pd.DataFrame, feasibility_samples: pd.DataFrame | None
) -> pd.DataFrame:
    """Frozen v2 responsive samples + accepted feasibility-phase samples."""
    v2 = v2_samples[
        v2_samples["checkpoint_role"].astype(str) == "responsive"
    ].copy()
    v2["evidence_source"] = "pilot_v2"
    frames = [v2]
    if feasibility_samples is not None and len(feasibility_samples):
        new = feasibility_samples.copy()
        new["evidence_source"] = FEASIBILITY_PHASE
        frames.append(new)
    return pd.concat(frames, ignore_index=True, sort=False)


def _within_boundary_band(
    group: pd.DataFrame,
    *,
    scientific_margin: dict,
    boundary_band: dict,
) -> pd.Series:
    """Rows whose three continuous deltas all sit within margin + band."""
    pfv = pd.to_numeric(group[_DELTA_PFV], errors="coerce")
    tfv = pd.to_numeric(group[_DELTA_TFV], errors="coerce")
    peak = pd.to_numeric(group[_DELTA_PEAK], errors="coerce")
    return (
        (pfv <= float(scientific_margin["pfv_m3"]) + float(boundary_band["pfv_m3"]))
        & (tfv <= float(scientific_margin["tfv_m3"]) + float(boundary_band["tfv_m3"]))
        & (
            peak
            <= float(scientific_margin["peak_m3s"])
            + float(boundary_band["peak_m3s"])
        )
    )


def classify_feasibility_states(
    catalog: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    scientific_margin: dict,
    boundary_band: dict,
    unaccounted_by_state: dict[tuple[str, str], int] | None = None,
) -> pd.DataFrame:
    """One frozen-vocabulary class per responsive state.

    ``samples`` is the combined evidence (v2 + feasibility phase);
    ``unaccounted_by_state`` counts planned feasibility rows without a
    closed accept/reject/duplicate outcome -- any shortfall forces
    ``execution_unresolved`` fail-closed.
    """
    unaccounted_by_state = unaccounted_by_state or {}
    rows: list[dict] = []
    for _, state in catalog.iterrows():
        event_id = str(state["event_id"])
        checkpoint_id = str(state["checkpoint_id"])
        group = samples[
            (samples["event_id"].astype(str) == event_id)
            & (samples["checkpoint_id"].astype(str) == checkpoint_id)
        ]
        new_rows = group[
            group["evidence_source"].astype(str) == FEASIBILITY_PHASE
        ]
        joint = _bool(group.get("joint_noninferior"), group.index)
        pfv_safe = _bool(group.get("pfv_safe"), group.index)
        joint_families = sorted(
            group.loc[joint, "candidate_family"].astype(str).unique()
        )
        boundary_mask = _within_boundary_band(
            group,
            scientific_margin=scientific_margin,
            boundary_band=boundary_band,
        )
        unaccounted = int(
            unaccounted_by_state.get((event_id, checkpoint_id), 0)
        )
        if unaccounted > 0:
            state_class = "execution_unresolved"
        elif joint.any():
            state_class = (
                "joint_feasible_robust"
                if len(joint_families) >= 2
                else "joint_feasible_found"
            )
        elif bool((boundary_mask & ~joint).any()):
            state_class = "joint_boundary_found"
        elif not pfv_safe.any():
            state_class = "no_pfv_safe_found"
        else:
            state_class = "no_joint_found_under_budget"
        new_joint = _bool(
            new_rows.get("joint_noninferior"), new_rows.index
        )
        rows.append(
            {
                "event_id": event_id,
                "checkpoint_id": checkpoint_id,
                "state_id": str(state["state_id"]),
                "split": str(state["split"]),
                "checkpoint_min": float(state.get("checkpoint_min", float("nan"))),
                "rainfall_phase": _rainfall_phase(
                    float(state.get("checkpoint_min", 0.0) or 0.0)
                ),
                "positive_control_state": bool(state["positive_control_state"]),
                "online_joint_found": bool(state["positive_control_state"]),
                "state_feasibility_class": state_class,
                "exact_joint_found": bool(joint.any()),
                "new_joint_from_search": bool(new_joint.any()),
                "joint_family_count": int(len(joint_families)),
                "joint_families_json": json.dumps(joint_families),
                "boundary_candidate_count": int((boundary_mask & ~joint).sum()),
                "samples_total": int(len(group)),
                "feasibility_samples": int(len(new_rows)),
                "unaccounted_feasibility_rows": unaccounted,
                "oracle_revealed": bool(
                    state["oracle_revealed_flag_required"] and len(new_rows)
                ),
                "search_result_training_eligible": bool(
                    state["search_result_training_eligible"]
                ),
                "candidate_generator_hit": bool(
                    state["positive_control_state"] and joint.any()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(_STATE_KEYS).reset_index(drop=True)


def compute_generator_recall(map_frame: pd.DataFrame) -> dict:
    """Online candidate-generator state recall over exact-feasible states."""
    feasible = map_frame[
        map_frame["state_feasibility_class"].isin(
            ["joint_feasible_found", "joint_feasible_robust"]
        )
    ]
    online = feasible[feasible["online_joint_found"]]
    missed = feasible[~feasible["online_joint_found"]]
    fallback_only = map_frame[
        map_frame["state_feasibility_class"].isin(
            ["no_joint_found_under_budget", "no_pfv_safe_found"]
        )
    ]

    def _by(frame_all: pd.DataFrame, frame_hit: pd.DataFrame, key: str) -> dict:
        result: dict[str, dict] = {}
        for value in sorted(frame_all[key].astype(str).unique()):
            total = int((frame_all[key].astype(str) == value).sum())
            hit = int((frame_hit[key].astype(str) == value).sum())
            result[value] = {
                "exact_feasible": total,
                "online_found": hit,
                "recall": (hit / total) if total else None,
            }
        return result

    exact_count = int(len(feasible))
    online_count = int(len(online))
    return {
        "exact_joint_feasible_states": exact_count,
        "online_generator_joint_states": online_count,
        "candidate_generator_state_recall": (
            online_count / exact_count if exact_count else None
        ),
        "recall_by_event": _by(feasible, online, "event_id"),
        "recall_by_rainfall_phase": _by(feasible, online, "rainfall_phase"),
        "recall_by_split": _by(feasible, online, "split"),
        "missed_feasible_states": missed[_STATE_KEYS + ["state_id"]].to_dict(
            "records"
        ),
        "fallback_only_states": fallback_only[
            _STATE_KEYS + ["state_id", "state_feasibility_class"]
        ].to_dict("records"),
        "event_support": int(feasible["event_id"].nunique()),
    }


def build_best_candidates(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-state best joint / PFV-safe / TFV / Peak rows across all evidence."""
    rows: list[dict] = []
    for (event_id, checkpoint_id), group in samples.groupby(_STATE_KEYS):
        joint = _bool(group.get("joint_noninferior"), group.index)
        pfv_safe = _bool(group.get("pfv_safe"), group.index)

        def _pick(metric: str, mask: pd.Series | None) -> dict:
            values = pd.to_numeric(group[metric], errors="coerce")
            if mask is not None:
                values = values.where(mask)
            if values.notna().sum() == 0:
                return {"sample_id": "", "value": None, "family": "", "source": ""}
            index = values.idxmin()
            return {
                "sample_id": str(group.loc[index, "sample_id"]),
                "value": float(values.loc[index]),
                "family": str(group.loc[index, "candidate_family"]),
                "source": str(group.loc[index, "evidence_source"]),
            }

        best_joint = _pick(_DELTA_TFV, joint if joint.any() else None)
        if not joint.any():
            best_joint = {"sample_id": "", "value": None, "family": "", "source": ""}
        best_safe = _pick(_DELTA_PFV, pfv_safe if pfv_safe.any() else None)
        if not pfv_safe.any():
            best_safe = {"sample_id": "", "value": None, "family": "", "source": ""}
        best_tfv = _pick(_DELTA_TFV, None)
        best_peak = _pick(_DELTA_PEAK, None)
        rows.append(
            {
                "event_id": str(event_id),
                "checkpoint_id": str(checkpoint_id),
                **{f"best_joint_{k}": v for k, v in best_joint.items()},
                **{f"best_pfv_safe_{k}": v for k, v in best_safe.items()},
                **{f"best_tfv_{k}": v for k, v in best_tfv.items()},
                **{f"best_peak_{k}": v for k, v in best_peak.items()},
            }
        )
    return pd.DataFrame(rows).sort_values(_STATE_KEYS).reset_index(drop=True)


def plan_feasibility_round_b_directives(
    catalog: pd.DataFrame,
    samples: pd.DataFrame,
    candidate_plan: pd.DataFrame,
    *,
    scientific_margin: dict,
    boundary_band: dict,
) -> pd.DataFrame:
    """Near-boundary Round B directives for joint-missing states only.

    Excluded fail-closed: states holding joint candidates from >= 2 distinct
    families, and states without remaining budget under the frozen 32-run
    per-state cap.  Trigger names are recorded for the audit trail.
    """
    planned_per_state = candidate_plan.groupby(_STATE_KEYS).size()
    rows: list[dict] = []
    for _, state in catalog.iterrows():
        if not bool(state["joint_missing_state"]):
            continue
        event_id = str(state["event_id"])
        checkpoint_id = str(state["checkpoint_id"])
        group = samples[
            (samples["event_id"].astype(str) == event_id)
            & (samples["checkpoint_id"].astype(str) == checkpoint_id)
        ]
        joint = _bool(group.get("joint_noninferior"), group.index)
        joint_families = group.loc[joint, "candidate_family"].astype(str).nunique()
        if joint_families >= 2:
            continue
        planned = int(
            planned_per_state.get((event_id, checkpoint_id), 0)
        )
        budget = min(
            ROUND_B_MAX_PER_STATE, TOTAL_BUDGET_PER_MISSING_STATE - planned
        )
        if budget <= 0:
            continue
        pfv_safe = _bool(group.get("pfv_safe"), group.index)
        tfv_ok = _bool(group.get("tfv_noninferior"), group.index)
        peak_ok = _bool(group.get("peak_noninferior"), group.index)
        pfv = pd.to_numeric(group[_DELTA_PFV], errors="coerce")
        tfv = pd.to_numeric(group[_DELTA_TFV], errors="coerce")
        peak = pd.to_numeric(group[_DELTA_PEAK], errors="coerce")
        tfv_margin = float(scientific_margin["tfv_m3"])
        peak_margin = float(scientific_margin["peak_m3s"])
        pfv_margin = float(scientific_margin["pfv_m3"])
        tfv_near = (tfv > tfv_margin) & (
            tfv <= tfv_margin + float(boundary_band["tfv_m3"])
        )
        peak_near = (peak > peak_margin) & (
            peak <= peak_margin + float(boundary_band["peak_m3s"])
        )
        pfv_near = (pfv > pfv_margin) & (
            pfv <= pfv_margin + float(boundary_band["pfv_m3"])
        )
        new_rows = group[
            group["evidence_source"].astype(str) == FEASIBILITY_PHASE
        ]
        old_rows = group[
            group["evidence_source"].astype(str) != FEASIBILITY_PHASE
        ]
        improving = False
        if len(new_rows) and len(old_rows):
            improving = bool(
                pd.to_numeric(new_rows[_DELTA_TFV], errors="coerce").min()
                < pd.to_numeric(old_rows[_DELTA_TFV], errors="coerce").min()
            )
        triggers = {
            "pfv_safe_tfv_near_boundary": bool((pfv_safe & tfv_near).any()),
            "pfv_safe_peak_near_boundary": bool((pfv_safe & peak_near).any()),
            "two_noninferior_one_slight_fail": bool(
                (pfv_safe & tfv_ok & peak_near).any()
                or (pfv_safe & peak_ok & tfv_near).any()
                or (tfv_ok & peak_ok & pfv_near).any()
            ),
            "continuous_delta_near_margin": bool(
                (tfv_near | peak_near | pfv_near).any()
            ),
            "search_still_improving": improving,
        }
        if not any(triggers.values()):
            continue
        rows.append(
            {
                "event_id": event_id,
                "checkpoint_id": checkpoint_id,
                "state_id": str(state["state_id"]),
                "round_b_budget": int(budget),
                "round_a_planned": planned,
                "triggers_json": json.dumps(
                    [name for name, hit in triggers.items() if hit]
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "checkpoint_id",
            "state_id",
            "round_b_budget",
            "round_a_planned",
            "triggers_json",
        ],
    )


def evaluate_p3_gate(recall_report: dict, map_frame: pd.DataFrame) -> dict:
    """Advisory P3 gate over the completed feasibility map (spec section 8)."""
    exact = int(recall_report["exact_joint_feasible_states"])
    online = int(recall_report["online_generator_joint_states"])
    recall = recall_report["candidate_generator_state_recall"]
    unresolved = int(
        (
            map_frame["state_feasibility_class"] == "execution_unresolved"
        ).sum()
    )
    gates = {
        "exact_joint_feasible_states_at_least_8": exact >= 8,
        "event_support_at_least_3": int(recall_report["event_support"]) >= 3,
        "generator_recall_at_least_0p80": (
            recall is not None and recall >= 0.80
        ),
        "execution_unresolved_zero": unresolved == 0,
    }
    if exact == online and recall == 1.0:
        recommendation = (
            "generator_recall_perfect"
            if exact >= 8
            else "exact_matches_online_original_30pct_gate_inappropriate"
        )
    elif recall is not None and recall < 0.80:
        recommendation = "fix_candidate_generator_before_train1600"
    elif exact < 8:
        recommendation = "insufficient_exact_feasible_states"
    else:
        recommendation = "proceed_to_learning_task_redesign"
    return {
        "gates": gates,
        "recommendation": recommendation,
        "unresolved_states": unresolved,
    }


def audit_feasibility_map(
    map_frame: pd.DataFrame,
    samples: pd.DataFrame,
    accounting: dict,
    *,
    catalog: pd.DataFrame,
    candidate_plan: pd.DataFrame,
    hard_columns: tuple[str, ...],
    actual_duplicates: int,
) -> dict:
    """Mechanical hard gate over the built feasibility map (spec section 6).

    Scientific interpretation stays in Gate v3; this audit only proves the
    evidence is real, complete, budget-compliant and replay-consistent.
    """
    new_samples = samples[
        samples["evidence_source"].astype(str) == FEASIBILITY_PHASE
    ]
    hard_ok = all(
        bool(_bool(new_samples.get(column), new_samples.index).all())
        for column in hard_columns
        if len(new_samples)
    )
    replay_plan_rows = candidate_plan[
        candidate_plan["expected_replay_of"].fillna("").astype(str) != ""
    ]
    replay_expected = int(len(replay_plan_rows))
    replay_success = 0
    for _, row in replay_plan_rows.iterrows():
        match = new_samples[
            new_samples["sample_id"].astype(str) == str(row["sample_id"])
        ]
        if len(match) == 1 and str(
            match.iloc[0]["actual_schedule_sha256"]
        ) == str(row["expected_actual_sha"]):
            replay_success += 1
    replay_rate = (
        replay_success / replay_expected if replay_expected else None
    )
    classes = set(map_frame["state_feasibility_class"].astype(str))
    class_counts = (
        map_frame["state_feasibility_class"].value_counts().to_dict()
    )
    unresolved = int(class_counts.get("execution_unresolved", 0))
    recall_report = compute_generator_recall(map_frame)
    p3_gate = evaluate_p3_gate(recall_report, map_frame)
    checks = {
        "classes_in_frozen_vocabulary": classes.issubset(set(STATE_CLASSES)),
        "no_forbidden_class_terms": not any(
            term in cls for cls in classes for term in FORBIDDEN_CLASS_TERMS
        ),
        "all_catalog_states_classified": int(len(map_frame))
        == int(len(catalog)),
        "hard_authenticity_all_true": bool(hard_ok),
        "actual_duplicates_zero": int(actual_duplicates) == 0,
        "accounting_closed": bool(accounting.get("accounting_closed", False)),
        "missing_zero": int(accounting.get("missing", 1)) == 0,
        "execution_unresolved_zero": unresolved == 0,
        "replay_expected_nine": replay_expected
        == int(catalog["positive_control_state"].sum()),
        "replay_success_rate_100": replay_rate == 1.0,
        "state_ids_match_catalog": set(
            map_frame["state_id"].astype(str)
        )
        == set(catalog["state_id"].astype(str)),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "replay_expected": replay_expected,
        "replay_success": replay_success,
        "replay_success_rate": replay_rate,
        "feasibility_samples": int(len(new_samples)),
        "recall_report": recall_report,
        "p3_gate": p3_gate,
        "round_counts": {
            str(k): int(v)
            for k, v in candidate_plan["search_round"].value_counts().items()
        },
    }
