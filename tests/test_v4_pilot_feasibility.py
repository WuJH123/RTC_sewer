"""Gate P3 feasibility state catalog: pure-function regression tests."""

import json

import pandas as pd

from sewerrtc.v4.pilot_feasibility import (
    REFERENCE_BRANCHES,
    audit_feasibility_state_catalog,
    build_feasibility_state_catalog,
)

FACILITIES = ["F1", "F2", "F3"]


def _sample(
    sample_id: str,
    event: str,
    checkpoint: str,
    *,
    split: str = "pilot_train",
    role: str = "responsive",
    joint: bool = False,
    pfv_safe: bool = True,
    tfv_ok: bool = False,
    peak_ok: bool = True,
    family: str = "famA",
    delta_pfv: float = -5.0,
    delta_tfv: float = 10.0,
    delta_peak: float = 0.0,
    moved: tuple[int, ...] = (0,),
) -> dict:
    anchor = [[0.0] * len(FACILITIES)] * 2
    projected = [
        [1.0 if i in moved else 0.0 for i in range(len(FACILITIES))]
    ] * 2
    return {
        "sample_id": sample_id,
        "event_id": event,
        "checkpoint_id": checkpoint,
        "checkpoint_role": role,
        "checkpoint_state_sha256": f"sha_{event}_{checkpoint}",
        "checkpoint_min": 60.0,
        "split": split,
        "candidate_family": family,
        "pfv_safe": pfv_safe,
        "tfv_noninferior": tfv_ok,
        "peak_noninferior": peak_ok,
        "joint_noninferior": joint,
        "local_response_magnitude": 3.5,
        "delta_pfv_h120_vs_no_control": delta_pfv,
        "delta_tfv_h120_vs_dynamic_internal": delta_tfv,
        "delta_peak_h120_vs_dynamic_internal": delta_peak,
        "anchor_schedule_json": json.dumps(anchor),
        "projected_schedule_json": json.dumps(projected),
    }


def _manifest() -> pd.DataFrame:
    rows = [
        # State A: positive control, joints from two families.
        _sample("a1", "ev1", "ck1", joint=True, tfv_ok=True, family="famA"),
        _sample(
            "a2",
            "ev1",
            "ck1",
            joint=True,
            tfv_ok=True,
            family="famB",
            delta_pfv=-2.0,
            moved=(1,),
        ),
        # State B: tfv_always_degraded, non-train split.
        _sample(
            "b1",
            "ev2",
            "ck2",
            split="pilot_validation",
            tfv_ok=False,
            delta_tfv=30.0,
            moved=(0, 2),
        ),
        # State C: no PFV-safe candidate at all.
        _sample(
            "c1",
            "ev1",
            "ck3",
            pfv_safe=False,
            tfv_ok=False,
            delta_pfv=40.0,
        ),
        # Low-opportunity checkpoint must be excluded from the catalog.
        _sample("d1", "ev2", "ck9", role="low_opportunity"),
    ]
    return pd.DataFrame(rows)


def test_catalog_partition_and_flags() -> None:
    catalog = build_feasibility_state_catalog(
        _manifest(), facility_ids=FACILITIES
    )
    assert len(catalog) == 3  # low_opportunity excluded
    by_state = catalog.set_index(["event_id", "checkpoint_id"])
    state_a = by_state.loc[("ev1", "ck1")]
    assert bool(state_a["positive_control_state"])
    assert not bool(state_a["joint_missing_state"])
    assert state_a["dominant_failure_reason"] == ""
    assert state_a["joint_family_count"] == 2
    assert json.loads(state_a["joint_families_json"]) == ["famA", "famB"]
    state_b = by_state.loc[("ev2", "ck2")]
    assert bool(state_b["joint_missing_state"])
    assert state_b["dominant_failure_reason"] == "tfv_always_degraded"
    state_c = by_state.loc[("ev1", "ck3")]
    assert state_c["dominant_failure_reason"] == "no_pfv_safe_candidate"


def test_catalog_best_candidates_and_actives() -> None:
    catalog = build_feasibility_state_catalog(
        _manifest(), facility_ids=FACILITIES
    )
    by_state = catalog.set_index(["event_id", "checkpoint_id"])
    state_a = by_state.loc[("ev1", "ck1")]
    # Best PFV-safe candidate is the most negative delta among pfv_safe.
    assert state_a["best_pfv_safe_sample_id"] == "a1"
    assert state_a["best_pfv_safe_delta_pfv"] == -5.0
    assert json.loads(state_a["active_facility_ids_json"]) == ["F1", "F2"]
    assert state_a["active_facility_count"] == 2
    # State C has no pfv_safe rows: best pfv-safe candidate must be empty.
    state_c = by_state.loc[("ev1", "ck3")]
    assert state_c["best_pfv_safe_sample_id"] == ""
    assert pd.isna(state_c["best_pfv_safe_delta_pfv"])


def test_catalog_eval_policy_and_reference_paths() -> None:
    catalog = build_feasibility_state_catalog(
        _manifest(), facility_ids=FACILITIES
    )
    by_state = catalog.set_index(["event_id", "checkpoint_id"])
    train_state = by_state.loc[("ev1", "ck1")]
    eval_state = by_state.loc[("ev2", "ck2")]
    assert bool(train_state["search_result_training_eligible"])
    assert not bool(train_state["oracle_revealed_flag_required"])
    assert not bool(eval_state["search_result_training_eligible"])
    assert bool(eval_state["oracle_revealed_flag_required"])
    for branch in REFERENCE_BRANCHES:
        assert (
            train_state[f"ref_{branch}_path"]
            == f"pilot/references/ev1/ck1/{branch}/detail.csv"
        )


def test_catalog_audit_pass_and_fail() -> None:
    catalog = build_feasibility_state_catalog(
        _manifest(), facility_ids=FACILITIES
    )
    audit = audit_feasibility_state_catalog(
        catalog,
        expected_states=3,
        expected_positive_controls=1,
        expected_joint_missing=2,
    )
    assert audit["status"] == "pass"
    assert all(audit["checks"].values())
    assert audit["dominant_failure_counts"] == {
        "tfv_always_degraded": 1,
        "no_pfv_safe_candidate": 1,
    }
    # Wrong expectations must fail closed, never coerce.
    bad = audit_feasibility_state_catalog(
        catalog,
        expected_states=32,
        expected_positive_controls=9,
        expected_joint_missing=23,
    )
    assert bad["status"] == "blocked"
    assert not bad["checks"]["state_count_matches"]


def test_catalog_missing_columns_fail_closed() -> None:
    frame = _manifest().drop(columns=["joint_noninferior"])
    try:
        build_feasibility_state_catalog(frame, facility_ids=FACILITIES)
    except ValueError as exc:
        assert "joint_noninferior" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
