from __future__ import annotations

import pandas as pd


PRIORITY_COLUMNS = (
    "pfv_boundary_score",
    "peak_boundary_score",
    "ranking_conflict_score",
    "uncertainty",
    "coverage_gap",
    "new_action_combination_score",
    "hard_negative_deficit_score",
    "ood_edge_score",
)

MAX_ACCEPTED_PER_STATE = 5


def filter_selectable_candidates(
    candidates: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    accepted_shas: pd.DataFrame | None = None,
    completed_case_ids: set[str] | None = None,
    current_plan_sha: str | None = None,
) -> pd.DataFrame:
    """Pre-filter for Rounds 1-3: never resample states already satisfied.

    Removes states that reached their accepted target, states with an
    exhausted candidate budget, candidates whose projected schedule repeats
    an already-accepted actual schedule for the same state, candidates whose
    case already completed, and rows planned against a stale plan SHA.
    """
    required = {"case_id", "event_id", "checkpoint_id"}
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    ledger_required = {
        "event_id",
        "checkpoint_id",
        "target_met",
        "budget_exhausted",
    }
    ledger_missing = ledger_required - set(ledger)
    if ledger_missing:
        raise ValueError(f"ledger missing columns: {sorted(ledger_missing)}")
    key = ["event_id", "checkpoint_id"]
    source = candidates.merge(
        ledger[key + ["target_met", "budget_exhausted"]],
        on=key,
        how="left",
        validate="many_to_one",
    )
    source["target_met"] = source["target_met"].fillna(False).astype(bool)
    source["budget_exhausted"] = (
        source["budget_exhausted"].fillna(False).astype(bool)
    )
    source = source[~source["target_met"] & ~source["budget_exhausted"]]
    if (
        accepted_shas is not None
        and len(accepted_shas)
        and "projected_schedule_sha256" in source
        and "actual_schedule_sha256" in accepted_shas
    ):
        taken = accepted_shas.groupby(key)["actual_schedule_sha256"].apply(
            lambda values: set(map(str, values))
        )
        duplicate = source.apply(
            lambda row: str(row["projected_schedule_sha256"])
            in taken.get((row["event_id"], row["checkpoint_id"]), set()),
            axis=1,
        )
        source = source[~duplicate]
    if completed_case_ids:
        done = {str(item) for item in completed_case_ids}
        source = source[~source["case_id"].astype(str).isin(done)]
    if current_plan_sha is not None and "plan_sha256" in source:
        source = source[
            source["plan_sha256"].astype(str) == str(current_plan_sha)
        ]
    return source.drop(
        columns=["target_met", "budget_exhausted"]
    ).reset_index(drop=True)


def select_active_learning_cases(
    candidates: pd.DataFrame,
    *,
    limit: int,
    per_checkpoint: int = 5,
) -> pd.DataFrame:
    required = {
        "case_id",
        "event_id",
        "checkpoint_id",
        "uncertainty",
        "coverage_gap",
        "boundary_distance",
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    source = candidates.copy()
    source["_score"] = (
        pd.to_numeric(source["uncertainty"])
        + pd.to_numeric(source["coverage_gap"])
        + 1.0 / (1.0 + pd.to_numeric(source["boundary_distance"]).abs())
    )
    source = source.sort_values("_score", ascending=False)
    selected = (
        source.groupby(["event_id", "checkpoint_id"], group_keys=False)
        .head(int(per_checkpoint))
        .head(int(limit))
        .drop(columns="_score")
    )
    return selected


def select_next_round_candidates(
    candidates: pd.DataFrame,
    accepted_counts: pd.DataFrame,
    *,
    target: int,
    per_state_cap: int = MAX_ACCEPTED_PER_STATE,
) -> pd.DataFrame:
    """Priority-ordered next-round selection with the 5-per-state cap.

    ``accepted_counts`` carries one row per state with the count of already
    accepted actual-unique candidates; the final accepted total per state can
    never exceed ``per_state_cap``.  Priority order: PFV safety boundary,
    Peak safety boundary, ranking conflict, uncertainty, facility/family
    coverage gap, new action combinations, hard-negative deficit, OOD edge.
    """
    required = {"case_id", "event_id", "checkpoint_id", *PRIORITY_COLUMNS}
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    counts_required = {"event_id", "checkpoint_id", "accepted_actual_unique"}
    counts_missing = counts_required - set(accepted_counts)
    if counts_missing:
        raise ValueError(
            f"accepted counts missing columns: {sorted(counts_missing)}"
        )
    source = candidates.merge(
        accepted_counts[list(counts_required)],
        on=["event_id", "checkpoint_id"],
        how="left",
        validate="many_to_one",
    )
    source["accepted_actual_unique"] = (
        pd.to_numeric(source["accepted_actual_unique"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    source["_headroom"] = (
        int(per_state_cap) - source["accepted_actual_unique"]
    ).clip(lower=0)
    source = source[source["_headroom"] > 0].copy()
    source["_priority"] = 0.0
    for weight, column in enumerate(reversed(PRIORITY_COLUMNS), start=1):
        source["_priority"] += float(weight) * pd.to_numeric(
            source[column], errors="coerce"
        ).fillna(0.0)
    source = source.sort_values(
        ["_priority", "case_id"], ascending=[False, True]
    )
    selected_rows = []
    remaining: dict[tuple[str, str], int] = {}
    for _, row in source.iterrows():
        key = (str(row["event_id"]), str(row["checkpoint_id"]))
        if key not in remaining:
            remaining[key] = int(row["_headroom"])
        if remaining[key] <= 0:
            continue
        remaining[key] -= 1
        selected_rows.append(row)
        if len(selected_rows) >= int(target):
            break
    if not selected_rows:
        return source.head(0).drop(
            columns=["_priority", "_headroom"], errors="ignore"
        )
    return pd.DataFrame(selected_rows).drop(
        columns=["_priority", "_headroom"], errors="ignore"
    )
