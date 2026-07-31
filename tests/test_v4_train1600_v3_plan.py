"""Train1600 V3 plan: 48/8/8/16 split, 3200-row role plan, frozen audit."""
from __future__ import annotations

import json

from sewerrtc.v4.train1600_v3 import (
    TRAIN_V3_BUDGET,
    V3_ROLES_BY_STRATUM,
    V3_STRATA,
    audit_train1600_plan_v3,
)
from train_v3_helpers import (
    SPLIT_COUNTS,
    make_ledger,
    make_plan_chain,
    make_standard_catalog,
)
from sewerrtc.v4.event_splits import select_train1600_events


def _fake_freeze() -> dict:
    return {
        "calibration_plan_sha256": "sha_cal",
        "locked_validation_plan_sha256": "sha_locked",
    }


def test_selection_48_8_8_16_excludes_pilot_p3_and_gate_tuning_events() -> None:
    catalog = make_standard_catalog(86)
    ledger = make_ledger(catalog)
    exclusions = {
        "ev080": ("used_pilot", "pilot"),        # Pilot400 / P3 exact search
        "ev081": ("used_gate5r", ""),            # Gate5R / Oracle tuning
        "ev082": ("used_peak_boundary", ""),     # Peak Boundary
        "ev083": ("policy_tuned_on_event", ""),  # tuning event
        "ev084": ("used_challenge", "challenge"),
        "ev085": ("used_formal", "formal"),
    }
    for event, (flag, split) in exclusions.items():
        mask = ledger["event_id"] == event
        ledger.loc[mask, flag] = True
        if split:
            ledger.loc[mask, "assigned_split"] = split
    # A formal_eligible row with development usage would break the ledger
    # invariant; excluded events stay ineligible.
    ledger.loc[ledger["event_id"].isin(exclusions), "formal_eligible"] = False

    selection = select_train1600_events(catalog, ledger, counts=SPLIT_COUNTS)

    assert {k: len(v) for k, v in selection.items()} == SPLIT_COUNTS
    chosen = [event for events in selection.values() for event in events]
    assert len(set(chosen)) == 80
    assert not set(exclusions) & set(chosen)


def test_role_plan_3200_rows_10_per_state_5_primary_unique_cases() -> None:
    chain = make_plan_chain()
    role_plan = chain["role_plan"]

    assert len(role_plan) == 3200
    per_state = role_plan.groupby(["event_id", "checkpoint_id"]).size()
    assert len(per_state) == 320
    assert per_state.eq(10).all()
    primary = role_plan[role_plan["plan_tier"] == "primary"]
    assert primary.groupby(["event_id", "checkpoint_id"]).size().eq(5).all()
    assert not role_plan["case_id"].duplicated().any()
    # State-reserve rows only replenish inside the 10-candidate budget.
    reserve = role_plan[role_plan["plan_tier"] == "state_reserve"]
    assert (reserve["replenish_source"] == "state_reserve_candidate").all()
    assert TRAIN_V3_BUDGET["maximum_candidate_budget_per_state"] == 10


def test_stratum_menus_match_contract_and_fallback_never_forced_joint() -> None:
    for stratum in V3_STRATA:
        assert len(V3_ROLES_BY_STRATUM[stratum]) == 5
    fallback_roles = {
        role for role, _mat in V3_ROLES_BY_STRATUM["predicted_fallback_likely"]
    }
    # The fallback-likely menu never contains a joint-seeking family.
    assert not any(role.startswith("joint_seeking") for role in fallback_roles)
    assert "near_reference_neutral" in fallback_roles
    low_roles = {role for role, _mat in V3_ROLES_BY_STRATUM["low_opportunity"]}
    assert {"k1_strong_legal_probe", "k2_strong_legal_probe"} <= low_roles


def test_plan_audit_passes_with_freeze_and_blocks_without() -> None:
    chain = make_plan_chain()

    audit = audit_train1600_plan_v3(
        chain["train_catalog"],
        chain["reserve_catalog"],
        chain["selection"],
        chain["role_plan"],
        chain["rotation"],
        plan_freeze=_fake_freeze(),
    )
    assert audit["status"] == "pass"
    assert audit["checks"]["stratification_online_features_only"]
    assert audit["checks"]["low_opportunity_one_per_event"]
    assert audit["informational"][
        "fallback_likely_states_never_forced_joint"
    ] is True
    json.dumps(audit, allow_nan=False)

    unfrozen = audit_train1600_plan_v3(
        chain["train_catalog"],
        chain["reserve_catalog"],
        chain["selection"],
        chain["role_plan"],
        chain["rotation"],
        plan_freeze=None,
    )
    assert unfrozen["status"] == "blocked"
    assert not unfrozen["checks"]["calibration_plan_sha_frozen"]
    assert not unfrozen["checks"]["locked_validation_plan_sha_frozen"]


def test_exact_search_labels_never_enter_the_role_plan() -> None:
    chain = make_plan_chain()
    # Exact feasibility classes are offline labels only; they must never be
    # columns of the online planning surface.
    assert "state_feasibility_class" not in chain["role_plan"].columns
    assert "state_feasibility_class" not in chain["stratified"].columns
