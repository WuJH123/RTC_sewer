"""Torch-free Train1600 planning: checkpoint catalog, Round0 and audits.

This module must never import torch; it feeds SWMM worker processes and the
PlanTrain1600/AuditTrain1600Plan stages, which are pure data planning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TRAIN_RESPONSIVE_ROLES = (
    "joint_beneficial_or_best_known",
    "pfv_boundary",
    "tfv_hard_negative",
    "peak_hard_negative",
    "uncertainty_or_coverage_gap",
)

TRAIN_LOW_ROLES = (
    "confirmed_flat",
    "strong_but_legal_low_response",
    "neutral",
    "reference_neighbourhood",
    "uncertainty_sample",
)

PRIMARY_SPLITS = ("train", "calibration", "locked_validation")


def build_train_checkpoint_catalog(
    standard_catalog: pd.DataFrame,
    selection: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze the Train1600 checkpoint catalogs from the canonical source.

    Returns ``(train_catalog, reserve_catalog)`` with 64x5=320 and 16x5=80
    rows respectively.  Rows are copied verbatim from the standard catalog;
    no checkpoint is ever cloned or synthesized.
    """
    required = {
        "event_id",
        "rainfall_sha256",
        "checkpoint_id",
        "checkpoint_role",
        "event_tier",
    }
    missing = required - set(standard_catalog)
    if missing:
        raise ValueError(f"standard catalog missing: {sorted(missing)}")
    split_map: dict[str, str] = {}
    for split in (*PRIMARY_SPLITS, "reserve"):
        for event_id in selection.get(split, []):
            if str(event_id) in split_map:
                raise ValueError(f"event in two splits: {event_id}")
            split_map[str(event_id)] = split
    catalog = standard_catalog[
        standard_catalog["event_id"].astype(str).isin(split_map)
    ].copy()
    catalog["split"] = catalog["event_id"].astype(str).map(split_map)
    if not catalog.groupby("event_id").size().eq(5).all():
        raise ValueError("every selected event needs exactly 5 checkpoints")
    if catalog.duplicated(["event_id", "checkpoint_id"]).any():
        raise ValueError("duplicate checkpoints in train catalog")
    train_catalog = catalog[catalog["split"].isin(PRIMARY_SPLITS)].copy()
    reserve_catalog = catalog[catalog["split"] == "reserve"].copy()
    primary_events = train_catalog["event_id"].nunique()
    if primary_events != 64 or len(train_catalog) != 320:
        raise ValueError(
            "train catalog must be 64 events x 5 checkpoints = 320 rows, got "
            f"{primary_events} events / {len(train_catalog)} rows"
        )
    if reserve_catalog["event_id"].nunique() != 16 or len(reserve_catalog) != 80:
        raise ValueError(
            "reserve catalog must be 16 events x 5 checkpoints = 80 rows"
        )
    return (
        train_catalog.reset_index(drop=True),
        reserve_catalog.reset_index(drop=True),
    )


def _roles_for(checkpoint_role: str) -> tuple[str, ...]:
    return (
        TRAIN_LOW_ROLES
        if str(checkpoint_role) == "low_opportunity"
        else TRAIN_RESPONSIVE_ROLES
    )


def build_train1600_target_plan(train_catalog: pd.DataFrame) -> pd.DataFrame:
    """The formal 64x5x5=1600 accepted-candidate target ledger.

    Responsive states use the five responsive candidate roles; low-opportunity
    states use the five low-response roles.  Actual schedules are discovered
    at run time and must be actual-unique; roles are planning intents only.
    """
    rows: list[dict] = []
    for _, checkpoint in train_catalog.drop_duplicates(
        ["event_id", "checkpoint_id"]
    ).iterrows():
        for role in _roles_for(checkpoint["checkpoint_role"]):
            rows.append(
                {
                    **checkpoint.to_dict(),
                    "candidate_role": role,
                    "case_id": (
                        f"{checkpoint['event_id']}__"
                        f"{checkpoint['checkpoint_id']}__{role}"
                    ),
                    "status": "planned",
                }
            )
    plan = pd.DataFrame(rows)
    if len(plan) != 1600:
        raise ValueError(f"target plan must be 1600 rows, got {len(plan)}")
    return plan


def build_round0_plan(train_catalog: pd.DataFrame) -> pd.DataFrame:
    """Round 0: at least one base candidate per state, 400 planned total.

    Every one of the 320 states receives its first-role base candidate; the
    80 highest-opportunity responsive states receive a second role to reach
    the 400-accepted Round0 target deterministically.
    """
    states = train_catalog.drop_duplicates(
        ["event_id", "checkpoint_id"]
    ).copy()
    if len(states) != 320:
        raise ValueError(f"Round0 requires 320 states, got {len(states)}")
    rows: list[dict] = []
    for _, checkpoint in states.iterrows():
        role = _roles_for(checkpoint["checkpoint_role"])[0]
        rows.append(
            {
                **checkpoint.to_dict(),
                "candidate_role": role,
                "round": 0,
                "case_id": (
                    f"round0__{checkpoint['event_id']}__"
                    f"{checkpoint['checkpoint_id']}__{role}"
                ),
                "status": "planned",
            }
        )
    responsive = states[states["checkpoint_role"] == "responsive"].copy()
    responsive["_score"] = pd.to_numeric(
        responsive.get("opportunity_score", 0.0), errors="coerce"
    ).fillna(0.0)
    extras = responsive.sort_values(
        ["_score", "event_id", "checkpoint_id"],
        ascending=[False, True, True],
    ).head(400 - len(states))
    for _, checkpoint in extras.iterrows():
        role = _roles_for(checkpoint["checkpoint_role"])[1]
        record = checkpoint.drop(labels="_score").to_dict()
        rows.append(
            {
                **record,
                "candidate_role": role,
                "round": 0,
                "case_id": (
                    f"round0__{checkpoint['event_id']}__"
                    f"{checkpoint['checkpoint_id']}__{role}"
                ),
                "status": "planned",
            }
        )
    plan = pd.DataFrame(rows)
    if len(plan) != 400:
        raise ValueError(f"Round0 plan must be 400 rows, got {len(plan)}")
    if plan["case_id"].duplicated().any():
        raise ValueError("Round0 plan has duplicate case ids")
    return plan


def audit_train1600_plan(
    train_catalog: pd.DataFrame,
    reserve_catalog: pd.DataFrame,
    selection: dict[str, list[str]],
) -> dict:
    split_counts = {
        split: int(train_catalog[train_catalog["split"] == split][
            "event_id"
        ].nunique())
        for split in PRIMARY_SPLITS
    }
    all_events = pd.concat(
        [train_catalog["event_id"], reserve_catalog["event_id"]]
    ).astype(str)
    split_sets = {
        split: set(map(str, selection.get(split, [])))
        for split in (*PRIMARY_SPLITS, "reserve")
    }
    overlaps = False
    names = list(split_sets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if split_sets[first] & split_sets[second]:
                overlaps = True
    shas = pd.concat(
        [train_catalog, reserve_catalog]
    ).drop_duplicates("event_id")
    checks = {
        "train_catalog_320_rows": len(train_catalog) == 320,
        "reserve_catalog_80_rows": len(reserve_catalog) == 80,
        "primary_events_64": train_catalog["event_id"].nunique() == 64,
        "reserve_events_16": reserve_catalog["event_id"].nunique() == 16,
        "split_48_8_8": split_counts
        == {"train": 48, "calibration": 8, "locked_validation": 8},
        "no_split_overlap": not overlaps,
        "five_checkpoints_per_event": bool(
            pd.concat([train_catalog, reserve_catalog])
            .groupby("event_id")
            .size()
            .eq(5)
            .all()
        ),
        "rainfall_sha_unique_per_event": not shas[
            "rainfall_sha256"
        ].duplicated().any(),
        "standard_tier_only": bool(
            all_events.isin(
                train_catalog[
                    train_catalog["event_tier"] == "standard_4plus"
                ]["event_id"].astype(str)
            ).all()
            or (
                pd.concat([train_catalog, reserve_catalog])["event_tier"]
                .eq("standard_4plus")
                .all()
            )
        ),
        "reserve_not_in_primary": not (
            split_sets["reserve"]
            & (
                split_sets["train"]
                | split_sets["calibration"]
                | split_sets["locked_validation"]
            )
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "split_event_counts": split_counts,
        "target_accounting": {
            "events": 64,
            "checkpoints_per_event": 5,
            "candidates_per_checkpoint": 5,
            "accepted_target": 1600,
            "train_samples": 1200,
            "calibration_samples": 200,
            "locked_validation_samples": 200,
            "reserve_samples_counted": 0,
        },
    }


TRAIN_BUDGET_DEFAULTS = {
    "target_accepted_per_state": 5,
    "initial_candidate_budget_per_state": 6,
    "maximum_candidate_budget_per_state": 10,
    "target_accepted_total": 1600,
    "allow_duplicate_replacement": False,
}

# Legal replenishment order after a rejection / no-op / actual duplicate.
REPLENISH_ORDER = (
    "state_reserve_candidate",
    "new_candidate_family",
    "boundary_uncertainty_coverage_gap",
)


def build_round_rotation(
    train_catalog: pd.DataFrame, *, rounds: int = 4, extra_per_round: int = 80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic 4x400 rotation: 320 basics + disjoint extra-80 sets.

    Every round targets one accepted candidate per state (320 basics) plus a
    second candidate for 80 states; the four extra-80 sets never overlap, so
    after four rounds each state totals exactly 4 + 1 = 5 accepted targets:
    4 x 320 basic = 1280, 4 x 80 extra = 320, total = 1600.  States are dealt
    round-robin over events first so each round's extras stay event- and
    split-balanced; the layout is a pure function of the sorted catalog.
    """
    states = train_catalog.drop_duplicates(
        ["event_id", "checkpoint_id"]
    ).copy()
    if len(states) != rounds * extra_per_round:
        raise ValueError(
            f"rotation requires {rounds * extra_per_round} states, "
            f"got {len(states)}"
        )
    states = states.sort_values(
        ["event_id", "checkpoint_id"], kind="stable"
    ).reset_index(drop=True)
    # Deal one state per event at a time: e1c1, e2c1, ..., e1c2, e2c2, ...
    states["_pos_in_event"] = states.groupby("event_id").cumcount()
    states = states.sort_values(
        ["_pos_in_event", "event_id"], kind="stable"
    ).reset_index(drop=True)
    # Diagonal assignment keeps the extra-80 sets event-balanced: an event's
    # five states land in at least four different rounds (never all in one).
    event_ordinal = pd.factorize(states["event_id"])[0]
    states["extra_round"] = (states["_pos_in_event"] + event_ordinal) % rounds
    states = states.drop(columns=["_pos_in_event"])
    target_rows = []
    extra_rows = []
    for round_index in range(rounds):
        for _, state in states.iterrows():
            extra = int(state["extra_round"]) == round_index
            target_rows.append(
                {
                    "round": round_index,
                    "event_id": state["event_id"],
                    "checkpoint_id": state["checkpoint_id"],
                    "split": state.get("split", ""),
                    "basic_target": 1,
                    "extra_target": int(extra),
                    "round_target": 1 + int(extra),
                }
            )
            if extra:
                extra_rows.append(
                    {
                        "round": round_index,
                        "event_id": state["event_id"],
                        "checkpoint_id": state["checkpoint_id"],
                        "split": state.get("split", ""),
                    }
                )
    state_targets = pd.DataFrame(target_rows)
    extra_rotation = pd.DataFrame(extra_rows)
    per_round = state_targets.groupby("round")["round_target"].sum()
    if not per_round.eq(len(states) + extra_per_round).all():
        raise ValueError("each round must target exactly 400 accepted")
    per_state = state_targets.groupby(
        ["event_id", "checkpoint_id"]
    )["round_target"].sum()
    if not per_state.eq(rounds + 1).all():
        raise ValueError("each state must total exactly 5 accepted targets")
    if extra_rotation.duplicated(["event_id", "checkpoint_id"]).any():
        raise ValueError("extra-80 sets must be disjoint across rounds")
    return state_targets, extra_rotation


def verify_round_rotation(state_targets: pd.DataFrame) -> dict:
    """Independent arithmetic proof of the 4x400 rotation."""
    per_state = state_targets.groupby(
        ["event_id", "checkpoint_id"]
    )["round_target"].sum()
    per_round = state_targets.groupby("round")["round_target"].sum()
    extras = state_targets[state_targets["extra_target"] == 1]
    checks = {
        "basic_total_1280": int(state_targets["basic_target"].sum()) == 1280,
        "extra_total_320": int(state_targets["extra_target"].sum()) == 320,
        "grand_total_1600": int(state_targets["round_target"].sum()) == 1600,
        "each_round_400": bool(per_round.eq(400).all()),
        "each_state_exactly_5": bool(per_state.eq(5).all()),
        "extra_disjoint": not extras.duplicated(
            ["event_id", "checkpoint_id"]
        ).any(),
        "rounds_4": state_targets["round"].nunique() == 4,
        "states_320": per_state.size == 320,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
    }


def build_state_budget_ledger(
    train_catalog: pd.DataFrame,
    accepted: pd.DataFrame,
    attempted: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Per-state accepted target vs. candidate budget bookkeeping.

    ``accepted`` must already be actual-unique informative samples (no
    references, no no-ops); ``attempted`` is every candidate case charged
    against the budget.  Target stays 5 while the budget may grow to 10;
    the two never mix.  ``per_state_accepted_progress.csv`` is written from
    this frame.
    """
    cfg = {**TRAIN_BUDGET_DEFAULTS, **(config or {})}
    target = int(cfg["target_accepted_per_state"])
    max_budget = int(cfg["maximum_candidate_budget_per_state"])
    key = ["event_id", "checkpoint_id"]
    states = train_catalog.drop_duplicates(key)
    accepted_sets = (
        accepted.groupby(key)["actual_schedule_sha256"].apply(
            lambda values: set(map(str, values))
        )
        if len(accepted) and "actual_schedule_sha256" in accepted
        else pd.Series(dtype=object)
    )
    attempted_counts = (
        attempted.groupby(key).size()
        if len(attempted)
        else pd.Series(dtype=int)
    )
    rows = []
    for _, state in states.iterrows():
        state_key = (state["event_id"], state["checkpoint_id"])
        shas = accepted_sets.get(state_key, set())
        got = len(shas)
        spent = int(attempted_counts.get(state_key, 0))
        exhausted = spent >= max_budget
        rows.append(
            {
                "event_id": state["event_id"],
                "checkpoint_id": state["checkpoint_id"],
                "split": state.get("split", ""),
                "checkpoint_role": state.get("checkpoint_role", ""),
                "target_accepted": target,
                "accepted_actual_unique": got,
                "attempted": spent,
                "budget_remaining": max(max_budget - spent, 0),
                "target_met": bool(got >= target),
                "budget_exhausted": bool(exhausted),
                "state_shortfall": bool(got < target and exhausted),
            }
        )
    return pd.DataFrame(rows)


def plan_state_replenishment(
    ledger: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    accepted: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Legal replenishment for states still short of their accepted target.

    Candidates are drawn per state in REPLENISH_ORDER, must be actual-unique
    against the state's accepted schedule SHAs, never cross states or splits,
    never reuse references or no-ops, and stop at the maximum budget.
    """
    required = {
        "event_id",
        "checkpoint_id",
        "candidate_id",
        "candidate_family",
        "replenish_source",
        "projected_schedule_sha256",
    }
    missing = required - set(candidate_pool)
    if missing:
        raise ValueError(f"candidate pool missing: {sorted(missing)}")
    unknown = set(candidate_pool["replenish_source"].astype(str)) - set(
        REPLENISH_ORDER
    )
    if unknown:
        raise ValueError(f"illegal replenish sources: {sorted(unknown)}")
    key = ["event_id", "checkpoint_id"]
    taken = (
        accepted.groupby(key)["actual_schedule_sha256"].apply(
            lambda values: set(map(str, values))
        )
        if len(accepted) and "actual_schedule_sha256" in accepted
        else pd.Series(dtype=object)
    )
    source_rank = {name: rank for rank, name in enumerate(REPLENISH_ORDER)}
    open_states = ledger[
        (~ledger["target_met"].astype(bool))
        & (~ledger["budget_exhausted"].astype(bool))
    ]
    rows = []
    for _, state in open_states.iterrows():
        state_key = (state["event_id"], state["checkpoint_id"])
        used = taken.get(state_key, set())
        pool = candidate_pool[
            (candidate_pool["event_id"] == state["event_id"])
            & (candidate_pool["checkpoint_id"] == state["checkpoint_id"])
        ].copy()
        pool = pool[
            ~pool["projected_schedule_sha256"].astype(str).isin(used)
        ]
        pool["_rank"] = pool["replenish_source"].map(source_rank)
        pool = pool.sort_values(["_rank", "candidate_id"], kind="stable")
        need = min(
            int(state["target_accepted"])
            - int(state["accepted_actual_unique"]),
            int(state["budget_remaining"]),
        )
        for _, candidate in pool.head(max(need, 0)).iterrows():
            record = candidate.drop(labels="_rank").to_dict()
            rows.append(
                {
                    **record,
                    "split": state.get("split", ""),
                    "queue": "replenish",
                    "case_id": (
                        f"{candidate['event_id']}__"
                        f"{candidate['checkpoint_id']}__replenish__"
                        f"{candidate['candidate_id']}"
                    ),
                    "status": "planned",
                }
            )
    return pd.DataFrame(rows)


def plan_event_replacement(
    ledger: pd.DataFrame,
    train_catalog: pd.DataFrame,
    reserve_catalog: pd.DataFrame,
) -> dict:
    """Replace incomplete events wholesale with same-split reserve events.

    Any state_shortfall marks its whole event as event_shortfall; the formal
    1600 table never accepts a partial event, the shortfall event's data is
    kept as auxiliary only, and replacement always swaps the entire event
    (all five checkpoints) from a reserve event of the same split -- never a
    single checkpoint.
    """
    shortfall_states = ledger[ledger["state_shortfall"].astype(bool)]
    shortfall_events = sorted(
        shortfall_states["event_id"].astype(str).unique()
    )
    event_split = (
        train_catalog.drop_duplicates("event_id")
        .set_index(train_catalog.drop_duplicates("event_id")["event_id"].astype(str))["split"]
    )
    reserve_events = list(
        reserve_catalog.drop_duplicates("event_id")["event_id"].astype(str)
    )
    replacements = []
    used_reserve: set[str] = set()
    unresolved = []
    for event_id in shortfall_events:
        target_split = str(event_split.get(event_id, ""))
        chosen = None
        for reserve_id in reserve_events:
            if reserve_id in used_reserve:
                continue
            chosen = reserve_id
            break
        if chosen is None:
            unresolved.append(event_id)
            continue
        used_reserve.add(chosen)
        replacement_rows = reserve_catalog[
            reserve_catalog["event_id"].astype(str) == chosen
        ].copy()
        if len(replacement_rows) != 5:
            raise ValueError(
                f"reserve event {chosen} must carry exactly 5 checkpoints"
            )
        # The replacement joins the shortfall event's split so the formal
        # split geometry (48/8/8) is preserved; the reserve label is dropped.
        replacement_rows["split"] = target_split
        replacements.append(
            {
                "event_shortfall": event_id,
                "replacement_event": chosen,
                "split": target_split,
                "replacement_rows": replacement_rows.reset_index(drop=True),
                "whole_event": True,
            }
        )
    return {
        "event_shortfalls": shortfall_events,
        "replacements": replacements,
        "unresolved": unresolved,
        "auxiliary_events": shortfall_events,
        "partial_event_in_main_table": False,
    }
