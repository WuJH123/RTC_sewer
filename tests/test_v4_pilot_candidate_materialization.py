from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.pilot_candidates import (
    HORIZON_STEPS,
    LOW_OPPORTUNITY_ROLES,
    MAX_K,
    RESPONSIVE_ROLES,
    audit_pilot_materialized_plan,
    build_pilot_branch_plan,
    build_pilot_role_plan,
    materialize_pilot_candidates,
)


N_FACILITIES = 36
BINARY_IDS = ("ADD301.2", "ADD301.3")
VSP_ID = "add350.1"


def _facility_ids() -> list[str]:
    extra = [f"gate{i:02d}" for i in range(N_FACILITIES - 3)]
    return list(BINARY_IDS) + [VSP_ID] + extra


def _facility_semantics(facility_ids: list[str]) -> pd.DataFrame:
    rows = []
    for facility in facility_ids:
        rows.append(
            {
                "facility_id": facility,
                "binary_or_continuous": (
                    "binary" if facility in BINARY_IDS else "continuous"
                ),
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "rate_limit": 1.0,
                "min_hold_steps": 1,
                "actuator_type": (
                    "pump"
                    if facility in BINARY_IDS or facility == VSP_ID
                    else "gate"
                ),
            }
        )
    return pd.DataFrame(rows)


def _checkpoint_catalog(facility_ids: list[str]) -> pd.DataFrame:
    anchor = [
        0.0 if facility in BINARY_IDS else 0.5 for facility in facility_ids
    ]
    rows = []
    for event_index in range(8):
        event_id = f"e{event_index}"
        for checkpoint_index in range(5):
            rows.append(
                {
                    "event_id": event_id,
                    "rainfall_sha256": f"rain-{event_id}",
                    "checkpoint_id": f"{event_id}_c{checkpoint_index}",
                    "checkpoint_role": (
                        "responsive" if checkpoint_index < 4 else "low"
                    ),
                    "checkpoint_min": 60.0 + 10.0 * checkpoint_index,
                    "anchor_action_json": json.dumps(anchor),
                    "active_facility_ids_json": json.dumps(
                        facility_ids[:6]
                    ),
                    "source_runner_kwargs": json.dumps(
                        {"inp_path": "network.inp"}
                    ),
                    "network_sha256": "netsha",
                }
            )
    return pd.DataFrame(rows)


def _peak_anchor_library(facility_ids: list[str]) -> pd.DataFrame:
    anchor = np.array(
        [
            [0.0 if f in BINARY_IDS else 0.5 for f in facility_ids]
            for _ in range(HORIZON_STEPS)
        ]
    )
    projected = anchor.copy()
    projected[:4, 0] = 1.0
    projected[:4, 2] = 0.9
    rows = []
    for event_index in range(3):
        rows.append(
            {
                "event_id": f"e{event_index}",
                "sample_id": f"peak_anchor_{event_index}",
                "projected_schedule_json": json.dumps(projected.tolist()),
                "anchor_schedule_json": json.dumps(anchor.tolist()),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> dict:
    facility_ids = _facility_ids()
    catalog = _checkpoint_catalog(facility_ids)
    role_plan = build_pilot_role_plan(catalog)
    schedule_dir = tmp_path_factory.mktemp("schedules")
    candidate_plan, coverage_missing = materialize_pilot_candidates(
        role_plan,
        catalog,
        facility_ids=facility_ids,
        facility_semantics=_facility_semantics(facility_ids),
        peak_boundary_anchor_library=_peak_anchor_library(facility_ids),
        contract_sha256="contractsha",
        config_sha256="configsha",
        code_sha256="codesha",
        schedule_dir=schedule_dir,
        schedule_dir_relative_to=schedule_dir.parent,
    )
    branch_plan = build_pilot_branch_plan(
        candidate_plan, contract_sha256="contractsha"
    )
    return {
        "facility_ids": facility_ids,
        "role_plan": role_plan,
        "candidate_plan": candidate_plan,
        "coverage_missing": coverage_missing,
        "branch_plan": branch_plan,
    }


def test_role_plan_has_400_rows_with_leakage_free_split() -> None:
    catalog = _checkpoint_catalog(_facility_ids())
    role_plan = build_pilot_role_plan(catalog)

    assert len(role_plan) == 400
    assert role_plan["event_id"].nunique() == 8
    # 40 states x 10 roles, responsive and low-opportunity menus.
    per_state = role_plan.groupby(["event_id", "checkpoint_id"]).size()
    assert per_state.eq(10).all()
    responsive = role_plan[role_plan["checkpoint_role"] == "responsive"]
    assert set(responsive["candidate_role"]) == set(RESPONSIVE_ROLES)
    low = role_plan[role_plan["checkpoint_role"] != "responsive"]
    assert set(low["candidate_role"]) == set(LOW_OPPORTUNITY_ROLES)
    # sorted events: first five are pilot_train, last three held out.
    split_by_event = role_plan.groupby("event_id")["split"].agg("first")
    assert split_by_event.loc[[f"e{i}" for i in range(5)]].eq(
        "pilot_train"
    ).all()
    assert role_plan.groupby("event_id")["split"].nunique().eq(1).all()


def test_materializes_all_400_roles_into_unique_12x36_candidates(
    materialized: dict,
) -> None:
    candidate_plan = materialized["candidate_plan"]
    coverage_missing = materialized["coverage_missing"]

    assert len(candidate_plan) == 400
    assert len(coverage_missing) == 0
    assert candidate_plan["sample_id"].is_unique
    for _, row in candidate_plan.head(20).iterrows():
        schedule = np.asarray(
            json.loads(row["projected_schedule_json"]), dtype=float
        )
        assert schedule.shape == (HORIZON_STEPS, N_FACILITIES)
    groups = candidate_plan.groupby(["event_id", "checkpoint_id"])
    assert groups.ngroups == 40
    assert groups.size().eq(10).all()
    # Requested and projected schedules are unique within each state.
    assert groups["requested_schedule_sha256"].nunique().eq(10).all()
    assert groups["projected_schedule_sha256"].nunique().eq(10).all()


def test_candidates_respect_k_budget_and_actuator_semantics(
    materialized: dict,
) -> None:
    candidate_plan = materialized["candidate_plan"]
    facility_ids = materialized["facility_ids"]

    assert candidate_plan["k_target"].astype(int).le(MAX_K).all()
    assert candidate_plan["k_target"].astype(int).ge(1).all()
    for text in candidate_plan["k_sequence"]:
        assert max(json.loads(str(text)) or [0]) <= MAX_K
    for column in (
        "binary_semantics_ok",
        "vsp_semantics_ok",
        "bounds_ok",
        "rate_limit_ok",
        "dwell_ok",
        "interlock_ok",
        "no_reversal_ok",
        "projection_valid",
    ):
        assert candidate_plan[column].astype(bool).all(), column
    binary_indexes = [facility_ids.index(item) for item in BINARY_IDS]
    vsp_index = facility_ids.index(VSP_ID)
    for _, row in candidate_plan.head(40).iterrows():
        schedule = np.asarray(
            json.loads(row["projected_schedule_json"]), dtype=float
        )
        for index in binary_indexes:
            assert set(np.unique(schedule[:, index])) <= {0.0, 1.0}
        assert np.all(schedule[:, vsp_index] >= -1e-9)
        assert np.all(schedule[:, vsp_index] <= 1.0 + 1e-9)


def test_materialized_plan_audit_passes_end_to_end(
    materialized: dict,
) -> None:
    audit = audit_pilot_materialized_plan(
        materialized["role_plan"],
        materialized["candidate_plan"],
        materialized["branch_plan"],
        materialized["coverage_missing"],
        peak_tuned_event_ids={"e0", "e1", "e2"},
    )

    assert audit["status"] == "pass", audit


def test_saturated_active_pool_still_materializes_up_roles(
    tmp_path: Path,
) -> None:
    # Regression: checkpoints whose active facilities all sit at the upper
    # bound made every "up" request from the active pool collapse onto the
    # anchor, so pfv_boundary/uncertainty/temporal_pulse ended in
    # coverage_missing; the widened second-half attempt pool must find the
    # unique legal candidate among the remaining facilities.
    facility_ids = _facility_ids()
    catalog = _checkpoint_catalog(facility_ids)
    saturated = [
        1.0 if facility not in BINARY_IDS else 0.0
        for facility in facility_ids
    ]
    catalog["anchor_action_json"] = json.dumps(saturated)
    role_plan = build_pilot_role_plan(catalog)
    role_plan = role_plan[
        role_plan["candidate_role"].isin(
            ["pfv_boundary", "uncertainty", "temporal_pulse"]
        )
    ].reset_index(drop=True)
    assert len(role_plan) > 0

    candidate_plan, coverage_missing = materialize_pilot_candidates(
        role_plan,
        catalog,
        facility_ids=facility_ids,
        facility_semantics=_facility_semantics(facility_ids),
        peak_boundary_anchor_library=_peak_anchor_library(facility_ids),
        contract_sha256="contractsha",
        config_sha256="configsha",
        code_sha256="codesha",
        schedule_dir=tmp_path / "schedules",
        schedule_dir_relative_to=tmp_path,
    )

    assert len(coverage_missing) == 0, coverage_missing
    assert len(candidate_plan) == len(role_plan)
    # Mirror the audit contract: schedules are unique within each state.
    per_state = candidate_plan.groupby(["event_id", "checkpoint_id"])
    assert (
        per_state["projected_schedule_sha256"].nunique()
        == per_state.size()
    ).all()
