from __future__ import annotations

import pandas as pd

from .event_splits import select_pilot_events


PILOT_PLANNING_FILES = (
    "pilot_event_selection.csv",
    "pilot_checkpoint_catalog.csv",
    "pilot_candidate_plan.csv",
    "pilot_reference_plan.csv",
    "pilot_candidate_coverage.csv",
    "pilot_split_manifest.csv",
    "pilot_plan_audit.json",
    "completion.json",
)

PILOT_REFERENCE_BRANCHES = (
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)


PILOT_ROLES = (
    "joint_beneficial_a",
    "joint_beneficial_b",
    "pfv_boundary",
    "PFV_hard_negative",
    "TFV_hard_negative",
    "peak_boundary",
    "PFV_safe_Peak_hard_negative",
    "neutral",
    "coverage_gap",
    "uncertainty",
)


def build_pilot400_plan(checkpoints: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id",
        "rainfall_sha256",
        "checkpoint_id",
        "checkpoint_role",
    }
    missing = required - set(checkpoints)
    if missing:
        raise ValueError(f"checkpoints missing columns: {sorted(missing)}")
    if checkpoints["event_id"].nunique() != 8:
        raise ValueError("Pilot400 requires exactly 8 events")
    if not checkpoints.groupby("event_id").size().eq(5).all():
        raise ValueError("Pilot400 requires five checkpoints per event")
    events = sorted(checkpoints["event_id"].astype(str).unique())
    split_names = ["pilot_train"] * 5 + [
        "pilot_calibration",
        "pilot_validation",
        "pilot_challenge",
    ]
    split_map = dict(zip(events, split_names))
    rows = []
    for _, checkpoint in checkpoints.iterrows():
        for role in PILOT_ROLES:
            rows.append(
                {
                    **checkpoint.to_dict(),
                    "candidate_role": role,
                    "split": split_map[str(checkpoint["event_id"])],
                    "case_id": (
                        f"{checkpoint['event_id']}__"
                        f"{checkpoint['checkpoint_id']}__{role}"
                    ),
                    "status": "planned",
                }
            )
    return pd.DataFrame(rows)


def bind_pilot_candidates(
    role_plan: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "event_id",
        "checkpoint_id",
        "candidate_role",
        "candidate_id",
        "family",
        "projected_schedule_sha256",
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidate catalog missing: {sorted(missing)}")
    candidate_source = candidates.drop_duplicates(
        ["event_id", "checkpoint_id", "candidate_role"],
        keep=False,
    )
    bound = role_plan.merge(
        candidate_source,
        on=["event_id", "checkpoint_id", "candidate_role"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_candidate"),
    )
    missing_rows = bound["candidate_id"].isna()
    if missing_rows.any():
        missing_roles = sorted(bound.loc[missing_rows, "candidate_role"].unique())
        raise ValueError(f"missing candidate role: {missing_roles}")
    duplicate = bound.duplicated(
        ["event_id", "checkpoint_id", "projected_schedule_sha256"]
    )
    if duplicate.any():
        raise ValueError("candidate binding contains projected schedule duplicates")
    return bound


def audit_pilot_plan(plan: pd.DataFrame) -> dict:
    required = {
        "event_id",
        "rainfall_sha256",
        "checkpoint_id",
        "checkpoint_role",
        "candidate_role",
        "split",
        "case_id",
        "candidate_id",
        "family",
        "projected_schedule_sha256",
    }
    missing = required - set(plan)
    if missing:
        return {"status": "blocked", "missing_columns": sorted(missing)}
    checks = {
        "planned_400": len(plan) == 400,
        "events_8": plan["event_id"].nunique() == 8,
        "five_checkpoints_per_event": plan.groupby("event_id")[
            "checkpoint_id"
        ].nunique().eq(5).all(),
        "ten_candidates_per_checkpoint": plan.groupby(
            ["event_id", "checkpoint_id"]
        ).size().eq(10).all(),
        "roles_complete": set(PILOT_ROLES).issubset(set(plan["candidate_role"])),
        "event_split_isolated": not plan.groupby("event_id")["split"].nunique().gt(1).any(),
        "rainfall_split_isolated": not plan.groupby("rainfall_sha256")[
            "split"
        ].nunique().gt(1).any(),
        "case_ids_unique": not plan["case_id"].duplicated().any(),
        "projected_schedules_unique_within_checkpoint": not plan.duplicated(
            ["event_id", "checkpoint_id", "projected_schedule_sha256"]
        ).any(),
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
    }


def audit_pilot_dataset(samples: pd.DataFrame) -> dict:
    required = {
        "event_id",
        "rainfall_sha256",
        "checkpoint_id",
        "checkpoint_role",
        "actual_schedule_sha256",
        "state_hash_match",
        "readback_ok",
        "locally_responsive",
        "confirmed_flat",
        "materially_beneficial",
        "joint_noninferior",
        "pfv_safe",
        "tfv_noninferior",
        "peak_noninferior",
    }
    missing = required - set(samples)
    if missing:
        return {"status": "blocked", "missing_columns": sorted(missing)}
    responsive = samples[samples["checkpoint_role"] == "responsive"]
    flat = samples[samples["confirmed_flat"].astype(bool)]
    peak_degraded = samples[~samples["peak_noninferior"].astype(bool)]
    safe_peak_negative = peak_degraded[peak_degraded["pfv_safe"].astype(bool)]
    checkpoint_joint = responsive.groupby(
        ["event_id", "checkpoint_id"]
    )["joint_noninferior"].any()

    def cross_event_both_sides(column: str) -> bool:
        values = samples.groupby("event_id")[column].agg(["any", "all"])
        positive_events = int(values["any"].sum())
        negative_events = int((~values["all"]).sum())
        return positive_events >= 3 and negative_events >= 3

    checks = {
        "accepted_events_8": samples["event_id"].nunique() == 8,
        "responsive_checkpoints_at_least_32": responsive.groupby(
            ["event_id", "checkpoint_id"]
        ).ngroups
        >= 32,
        "confirmed_flat_at_least_8": len(flat) >= 8,
        "informative_actual_unique_at_least_300": (
            len(
                samples.drop_duplicates(
                    ["event_id", "checkpoint_id", "actual_schedule_sha256"]
                )
            )
            >= 300
        ),
        "responsive_local_response_at_least_70pct": (
            float(responsive["locally_responsive"].mean()) >= 0.70
            if len(responsive)
            else False
        ),
        "flat_fraction_10_to_20pct": 0.10
        <= float(samples["confirmed_flat"].mean())
        <= 0.20,
        "same_state_100pct": bool(samples["state_hash_match"].all()),
        "readback_100pct": bool(samples["readback_ok"].all()),
        "actual_duplicates_0": not samples.duplicated(
            ["event_id", "checkpoint_id", "actual_schedule_sha256"]
        ).any(),
        "material_benefit_3_events": samples[
            samples["materially_beneficial"].astype(bool)
        ]["event_id"].nunique()
        >= 3,
        "joint_at_30pct_responsive_checkpoints": (
            float(checkpoint_joint.mean()) >= 0.30 if len(checkpoint_joint) else False
        ),
        "pfv_both_sides_3_events": cross_event_both_sides("pfv_safe"),
        "tfv_both_sides_3_events": cross_event_both_sides("tfv_noninferior"),
        "peak_both_sides_3_events": cross_event_both_sides("peak_noninferior"),
        "peak_degraded_at_least_30": len(peak_degraded) >= 30,
        "pfv_safe_peak_negative_at_least_10": len(safe_peak_negative) >= 10,
        "rainfall_sha_split_isolated": (
            "split" in samples
            and not samples.groupby("rainfall_sha256")["split"].nunique().gt(1).any()
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "checks": checks,
    }


def build_pilot_planning_bundle(
    standard_catalog: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    peak_anchor_library: pd.DataFrame,
    gate5r_classification: pd.DataFrame,
    count: int = 8,
    seed: int = 20260727,
) -> dict:
    """Plan Pilot400 from the canonical catalog and the event usage ledger.

    Inputs are restricted to the canonical standard checkpoint catalog, the
    event usage ledger, the Peak-boundary anchor library and the existing
    Gate5R classification.  Returns the eight planning artifacts as frames
    plus the audit dict; the Opportunity stage never writes these files.
    """
    if peak_anchor_library.empty:
        raise ValueError("peak boundary anchor library is empty")
    if gate5r_classification.empty:
        raise ValueError("existing Gate5R classification is empty")
    if not standard_catalog["event_tier"].eq("standard_4plus").all():
        raise ValueError("pilot planning only accepts standard_4plus events")
    selected = select_pilot_events(
        standard_catalog, ledger, count=count, seed=seed
    )
    checkpoints = standard_catalog[
        standard_catalog["event_id"].astype(str).isin(selected)
    ].copy()
    if len(checkpoints) != count * 5:
        raise ValueError(
            f"pilot checkpoint catalog must be {count * 5} rows, "
            f"got {len(checkpoints)}"
        )
    selection_rows = []
    for rank, event_id in enumerate(selected):
        event = checkpoints[
            checkpoints["event_id"].astype(str) == event_id
        ].iloc[0]
        selection_rows.append(
            {
                "event_id": event_id,
                "rainfall_sha256": str(event["rainfall_sha256"]),
                "rainfall_family": str(event.get("rainfall_family", "")),
                "risk_level": str(event.get("risk_level", "")),
                "selection_rank": rank,
                "split": "pilot",
            }
        )
    event_selection = pd.DataFrame(selection_rows)
    candidate_plan = build_pilot400_plan(checkpoints)
    reference_rows = []
    for _, checkpoint in checkpoints.iterrows():
        for branch in PILOT_REFERENCE_BRANCHES:
            reference_rows.append(
                {
                    "event_id": checkpoint["event_id"],
                    "rainfall_sha256": checkpoint["rainfall_sha256"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "branch_role": branch,
                    "counted_as_sample": False,
                }
            )
    reference_plan = pd.DataFrame(reference_rows)
    coverage = (
        candidate_plan.groupby(["event_id", "candidate_role"])
        .size()
        .rename("planned")
        .reset_index()
    )
    split_manifest = event_selection[
        ["event_id", "rainfall_sha256", "split"]
    ].copy()
    audit = audit_pilot_plan_bundle(
        event_selection, checkpoints, candidate_plan, reference_plan
    )
    return {
        "pilot_event_selection": event_selection,
        "pilot_checkpoint_catalog": checkpoints.reset_index(drop=True),
        "pilot_candidate_plan": candidate_plan,
        "pilot_reference_plan": reference_plan,
        "pilot_candidate_coverage": coverage,
        "pilot_split_manifest": split_manifest,
        "pilot_plan_audit": audit,
        "selected_events": selected,
    }


def audit_pilot_plan_bundle(
    event_selection: pd.DataFrame,
    checkpoints: pd.DataFrame,
    candidate_plan: pd.DataFrame,
    reference_plan: pd.DataFrame,
) -> dict:
    roles = checkpoints.groupby("event_id")["checkpoint_role"].value_counts()
    responsive_ok = bool(
        roles.unstack(fill_value=0)
        .get("responsive", pd.Series(dtype=int))
        .eq(4)
        .all()
    )
    low_ok = bool(
        roles.unstack(fill_value=0)
        .get("low_opportunity", pd.Series(dtype=int))
        .eq(1)
        .all()
    )
    checks = {
        "events_8": len(event_selection) == 8,
        "rainfall_sha_unique": not event_selection[
            "rainfall_sha256"
        ].duplicated().any(),
        "families_at_least_3": event_selection["rainfall_family"].nunique()
        >= 3,
        "risk_levels_at_least_3": event_selection["risk_level"].nunique() >= 3,
        "checkpoints_40": len(checkpoints) == 40,
        "four_responsive_per_event": responsive_ok,
        "one_low_per_event": low_ok,
        "candidates_400": len(candidate_plan) == 400,
        "references_not_counted": not reference_plan[
            "counted_as_sample"
        ].astype(bool).any(),
        "reference_rows_120": len(reference_plan) == 120,
        "standard_tier_only": checkpoints["event_tier"].eq(
            "standard_4plus"
        ).all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
    }


PILOT_BUDGET_DEFAULTS = {
    "primary_candidates_per_state": 10,
    "reserve_candidates_per_state": 5,
    "min_accepted_informative_total": 300,
    "min_accepted_per_responsive_state": 6,
    "max_candidate_budget_per_state": 15,
}


def audit_pilot_state_progress(
    accepted: pd.DataFrame,
    attempted: pd.DataFrame,
    checkpoints: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Per-state accepted / budget ledger after the primary 400 finish.

    ``accepted`` holds actual-unique informative samples; ``attempted``
    holds every candidate case already spent against the state budget.
    Reserve is enabled only for responsive states short of the accepted
    floor; a state that cannot reach it within the max budget is marked
    ``state_shortfall`` instead of relaxing any constraint.
    """
    cfg = {**PILOT_BUDGET_DEFAULTS, **(config or {})}
    floor = int(cfg["min_accepted_per_responsive_state"])
    max_budget = int(cfg["max_candidate_budget_per_state"])
    key = ["event_id", "checkpoint_id"]
    states = checkpoints[key + ["checkpoint_role"]].drop_duplicates(key)
    accepted_counts = (
        accepted.groupby(key).size() if len(accepted) else pd.Series(dtype=int)
    )
    attempted_counts = (
        attempted.groupby(key).size()
        if len(attempted)
        else pd.Series(dtype=int)
    )
    rows = []
    for _, state in states.iterrows():
        state_key = (state["event_id"], state["checkpoint_id"])
        got = int(accepted_counts.get(state_key, 0))
        spent = int(attempted_counts.get(state_key, 0))
        responsive = str(state["checkpoint_role"]) == "responsive"
        short = responsive and got < floor
        exhausted = spent >= max_budget
        rows.append(
            {
                "event_id": state["event_id"],
                "checkpoint_id": state["checkpoint_id"],
                "checkpoint_role": state["checkpoint_role"],
                "accepted": got,
                "attempted": spent,
                "budget_remaining": max(max_budget - spent, 0),
                "reserve_eligible": bool(short and not exhausted),
                "budget_exhausted": bool(exhausted),
                "state_shortfall": bool(short and exhausted),
            }
        )
    return pd.DataFrame(rows)


def plan_pilot_reserve(
    state_progress: pd.DataFrame,
    reserve_catalog: pd.DataFrame,
    accepted: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Draw reserve candidates only for eligible states, actual-unique only.

    Reserve rows never copy an already-accepted actual schedule, never move
    across states, never change the event split, and stop at the per-state
    max budget.  A single rejection never triggers replenishment here; the
    caller runs this only after the primary plan has fully finished.
    """
    cfg = {**PILOT_BUDGET_DEFAULTS, **(config or {})}
    required = {
        "event_id",
        "checkpoint_id",
        "candidate_id",
        "family",
        "projected_schedule_sha256",
    }
    missing = required - set(reserve_catalog)
    if missing:
        raise ValueError(f"reserve catalog missing: {sorted(missing)}")
    key = ["event_id", "checkpoint_id"]
    eligible = state_progress[
        state_progress["reserve_eligible"].astype(bool)
    ]
    taken = (
        accepted.groupby(key)["actual_schedule_sha256"].apply(set)
        if len(accepted) and "actual_schedule_sha256" in accepted
        else pd.Series(dtype=object)
    )
    floor = int(cfg["min_accepted_per_responsive_state"])
    rows = []
    for _, state in eligible.iterrows():
        state_key = (state["event_id"], state["checkpoint_id"])
        used = taken.get(state_key, set())
        pool = reserve_catalog[
            (reserve_catalog["event_id"] == state["event_id"])
            & (reserve_catalog["checkpoint_id"] == state["checkpoint_id"])
        ]
        pool = pool[
            ~pool["projected_schedule_sha256"].astype(str).isin(used)
        ]
        need = min(
            floor - int(state["accepted"]),
            int(state["budget_remaining"]),
            int(cfg["reserve_candidates_per_state"]),
        )
        for _, candidate in pool.head(max(need, 0)).iterrows():
            rows.append(
                {
                    **candidate.to_dict(),
                    "queue": "reserve",
                    "case_id": (
                        f"{candidate['event_id']}__"
                        f"{candidate['checkpoint_id']}__"
                        f"reserve__{candidate['candidate_id']}"
                    ),
                    "status": "planned",
                }
            )
    return pd.DataFrame(rows)


def evaluate_pilot_gate(
    dataset_audit: dict, baseline_report: dict
) -> dict:
    """Combine the dataset audit and baseline report into the pilot verdict.

    PlanTrain1600 reads this verdict and requires ``scientific_pass=true``
    and ``exit_code=0``; anything else fails closed.
    """
    dataset_pass = dataset_audit.get("status") == "pass"
    baselines = baseline_report.get("models", {})
    baseline_checks = {
        f"{name}_beats_trivial": bool(
            payload.get("beats_zero_prediction", False)
            and payload.get("beats_majority_class", True)
        )
        for name, payload in baselines.items()
    }
    required_models = {"ridge", "logistic", "hist_gradient_boosting"}
    checks = {
        "dataset_audit_pass": dataset_pass,
        "required_models_present": required_models.issubset(set(baselines)),
        **baseline_checks,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    scientific_pass = all(checks.values())
    return {
        "status": "pass" if scientific_pass else "scientific_fail",
        "scientific_pass": scientific_pass,
        "exit_code": 0 if scientific_pass else 5,
        "checks": checks,
        "dataset_checks": {
            key: bool(value)
            for key, value in dataset_audit.get("checks", {}).items()
        },
    }
