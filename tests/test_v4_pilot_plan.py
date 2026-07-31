import hashlib

import pandas as pd

import pytest

from sewerrtc.v4.event_splits import assign_split, build_event_usage_ledger
from sewerrtc.v4.pilot import (
    PILOT_ROLES,
    bind_pilot_candidates,
    build_pilot400_plan,
    build_pilot_planning_bundle,
)


def test_pilot_plan_is_8_by_5_by_10_with_event_level_splits() -> None:
    checkpoints = []
    for event_index in range(8):
        for checkpoint_index in range(5):
            checkpoints.append(
                {
                    "event_id": f"e{event_index}",
                    "rainfall_sha256": f"rain-{event_index}",
                    "checkpoint_id": f"c{checkpoint_index}",
                    "checkpoint_role": (
                        "responsive" if checkpoint_index < 4 else "confirmed_flat"
                    ),
                }
            )
    plan = build_pilot400_plan(pd.DataFrame(checkpoints))

    assert len(plan) == 400
    assert plan.groupby("event_id").size().eq(50).all()
    assert not plan.groupby("event_id")["split"].nunique().gt(1).any()
    assert plan["candidate_role"].nunique() == 10


def test_pilot_binding_rejects_missing_role_and_never_fills_with_duplicates() -> None:
    checkpoints = pd.DataFrame(
        [
            {
                "event_id": f"e{event_index}",
                "rainfall_sha256": f"rain-{event_index}",
                "checkpoint_id": f"c{checkpoint_index}",
                "checkpoint_role": (
                    "responsive" if checkpoint_index < 4 else "confirmed_flat"
                ),
            }
            for event_index in range(8)
            for checkpoint_index in range(5)
        ]
    )
    role_plan = build_pilot400_plan(checkpoints)
    candidates = role_plan[
        ["event_id", "checkpoint_id", "candidate_role"]
    ].copy()
    candidates["candidate_id"] = candidates.index.map(lambda value: f"x{value}")
    candidates["family"] = candidates["candidate_role"]
    candidates["projected_schedule_sha256"] = candidates.index.map(
        lambda value: f"sha-{value}"
    )

    bound = bind_pilot_candidates(role_plan, candidates)
    assert len(bound) == 400
    assert not bound.duplicated(
        ["event_id", "checkpoint_id", "projected_schedule_sha256"]
    ).any()

    with pytest.raises(ValueError, match="missing candidate role"):
        bind_pilot_candidates(
            role_plan,
            candidates[
                candidates["candidate_role"] != PILOT_ROLES[-1]
            ],
        )


FAMILIES = ("frontal", "convective", "typhoon")
RISKS = ("low", "medium", "high")


def make_standard_catalog(num_events: int) -> pd.DataFrame:
    rows = []
    for index in range(num_events):
        event = f"ev{index:03d}"
        sha = hashlib.sha256(event.encode()).hexdigest()
        for checkpoint_index in range(5):
            responsive = checkpoint_index < 4
            rows.append(
                {
                    "event_id": event,
                    "rainfall_sha256": sha,
                    "checkpoint_id": f"{event}_cp{checkpoint_index}",
                    "elapsed_min": (
                        30 + checkpoint_index * 40
                        if responsive
                        else 240 + (index % 3) * 120
                    ),
                    "checkpoint_role": (
                        "responsive" if responsive else "low_opportunity"
                    ),
                    "rainfall_phase": "rising",
                    "opportunity_score": float(10 - checkpoint_index),
                    "event_tier": "standard_4plus",
                    "checkpoint_state_source": "cold_start_prefix_replay",
                    "network_sha256": "net",
                    "config_sha256": "cfg",
                    "source_run_uuid": "uuid",
                    "rainfall_family": FAMILIES[index % 3],
                    "risk_level": RISKS[index % 3],
                }
            )
    return pd.DataFrame(rows)


def make_ledger(catalog: pd.DataFrame) -> pd.DataFrame:
    return build_event_usage_ledger(
        catalog[["event_id", "rainfall_sha256", "event_tier"]],
        scanned_event_ids=set(catalog["event_id"].astype(str)),
    )


def test_pilot_planning_bundle_selects_8_development_events() -> None:
    catalog = make_standard_catalog(12)
    ledger = make_ledger(catalog)
    anchors = pd.DataFrame({"event_id": ["ev000"]})

    bundle = build_pilot_planning_bundle(
        catalog,
        ledger,
        peak_anchor_library=anchors,
        gate5r_classification=anchors,
    )

    assert len(bundle["selected_events"]) == 8
    assert len(bundle["pilot_checkpoint_catalog"]) == 40
    assert len(bundle["pilot_candidate_plan"]) == 400
    assert len(bundle["pilot_reference_plan"]) == 120
    assert not bundle["pilot_reference_plan"]["counted_as_sample"].any()
    assert bundle["pilot_plan_audit"]["status"] == "pass"
    assert all(
        type(value) is bool
        for value in bundle["pilot_plan_audit"]["checks"].values()
    )
    selection = bundle["pilot_event_selection"]
    assert not selection["rainfall_sha256"].duplicated().any()
    assert selection["rainfall_family"].nunique() >= 3
    assert selection["risk_level"].nunique() >= 3


def test_pilot_planning_never_uses_locked_challenge_or_formal_events() -> None:
    catalog = make_standard_catalog(9)
    ledger = make_ledger(catalog)
    ledger = assign_split(
        ledger, ["ev008"], "locked_validation", assignment_run_uuid="u"
    )
    anchors = pd.DataFrame({"event_id": ["ev000"]})

    bundle = build_pilot_planning_bundle(
        catalog,
        ledger,
        peak_anchor_library=anchors,
        gate5r_classification=anchors,
    )

    assert "ev008" not in bundle["selected_events"]


def test_pilot_planning_requires_nonempty_anchor_libraries() -> None:
    catalog = make_standard_catalog(12)
    ledger = make_ledger(catalog)

    with pytest.raises(ValueError, match="anchor library is empty"):
        build_pilot_planning_bundle(
            catalog,
            ledger,
            peak_anchor_library=pd.DataFrame(),
            gate5r_classification=pd.DataFrame({"event_id": ["ev000"]}),
        )
