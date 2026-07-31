import json

import pandas as pd

from sewerrtc.v4.opportunity import (
    audit_opportunity_coverage,
    classify_event_tiers,
    robust_normalize,
)

PHASES = ["rising", "pre_peak", "peak", "late_rain"]


def _event_rows(
    event_id: str,
    responsive_count: int,
    low_count: int = 1,
    family: str = "family-0",
    risk: str = "low",
) -> list[dict]:
    rows = []
    for index in range(responsive_count):
        rows.append(
            {
                "event_id": event_id,
                "checkpoint_min": 60 + index * 30,
                "opportunity_class": "responsive",
                "phase": PHASES[index % 4],
                "rainfall_family": family,
                "risk_level": risk,
            }
        )
    for index in range(low_count):
        rows.append(
            {
                "event_id": event_id,
                "checkpoint_min": 360 + index * 30,
                "opportunity_class": "low_opportunity",
                "phase": "late_rain",
                "rainfall_family": family,
                "risk_level": risk,
            }
        )
    return rows


def test_robust_normalization_does_not_permanently_saturate_static_values() -> None:
    values = pd.Series([10.0, 10.5, 11.0, 100.0])
    normalized = robust_normalize(values)

    assert normalized.between(0.0, 1.0).all()
    assert normalized.iloc[0] < normalized.iloc[-1]


def test_opportunity_coverage_requires_8_events_32_responsive_and_diversity() -> None:
    rows = []
    phases = ["rising", "pre_peak", "peak", "late_rain"]
    for event_index in range(8):
        for checkpoint_index, phase in enumerate(phases):
            rows.append(
                {
                    "event_id": f"e{event_index}",
                    "checkpoint_min": checkpoint_index * 30,
                    "opportunity_class": "responsive",
                    "phase": phase,
                    "rainfall_family": f"family-{event_index % 3}",
                    "risk_level": f"risk-{event_index % 3}",
                }
            )
        rows.append(
            {
                "event_id": f"e{event_index}",
                "checkpoint_min": 150,
                "opportunity_class": "low_opportunity",
                "phase": "late_rain",
                "rainfall_family": f"family-{event_index % 3}",
                "risk_level": f"risk-{event_index % 3}",
            }
        )

    audit = audit_opportunity_coverage(
        pd.DataFrame(rows),
        {"opportunity": {"min_standard_eligible_events": 8}},
    )
    assert audit["status"] == "pass"
    assert audit["responsive_checkpoints"] == 32


def test_duration_aware_targets_pass_for_5_4_and_3_planned_events() -> None:
    frame = pd.DataFrame(
        _event_rows("e5", responsive_count=4, low_count=1)
        + _event_rows("e4", responsive_count=3, low_count=1)
        + _event_rows("e3", responsive_count=2, low_count=1)
    )
    tiers = classify_event_tiers(frame).set_index("event_id")

    assert tiers.loc["e5", "planned_checkpoint_count"] == 5
    assert tiers.loc["e5", "required_responsive"] == 4
    assert tiers.loc["e4", "planned_checkpoint_count"] == 4
    assert tiers.loc["e4", "required_responsive"] == 3
    assert tiers.loc["e3", "planned_checkpoint_count"] == 3
    assert tiers.loc["e3", "required_responsive"] == 2
    assert tiers["meets_duration_aware_target"].all()
    assert tiers.loc["e5", "event_tier"] == "standard_4plus"
    assert tiers.loc["e4", "event_tier"] == "short_3"
    assert tiers.loc["e3", "event_tier"] == "short_2"


def test_event_below_feasible_target_fails_duration_aware_check() -> None:
    # planned=4 with 2 low controls: max_feasible=3 > responsive=2.
    rows = _event_rows("short", responsive_count=2, low_count=2)
    for index in range(7):
        rows += _event_rows(
            f"ok{index}",
            responsive_count=4,
            low_count=1,
            family=f"family-{index % 3}",
            risk=["low", "medium", "high"][index % 3],
        )
    audit = audit_opportunity_coverage(
        pd.DataFrame(rows),
        {"opportunity": {"min_standard_eligible_events": 7}},
    )

    assert audit["checks"]["all_events_meet_duration_aware_target"] is False
    assert audit["status"] == "scientific_fail"
    assert audit["events_below_duration_aware_target"] == ["short"]


def test_insufficient_standard_eligible_events_fails_gate() -> None:
    rows = []
    for index in range(10):
        rows += _event_rows(
            f"e{index}",
            responsive_count=4,
            low_count=1,
            family=f"family-{index % 3}",
            risk=["low", "medium", "high"][index % 3],
        )
    audit = audit_opportunity_coverage(pd.DataFrame(rows))

    assert audit["downstream_requirements"]["min_standard_eligible_events"] == 88
    assert audit["standard_eligible_event_count"] == 10
    assert audit["checks"]["standard_eligible_at_least_required"] is False
    assert audit["status"] == "scientific_fail"


def test_current_182_56_6_distribution_passes_gate() -> None:
    rows = []
    risks = ["low", "medium", "high"]
    index = 0
    for count, responsive_count in ((182, 4), (56, 3), (6, 2)):
        for _ in range(count):
            rows += _event_rows(
                f"e{index}",
                responsive_count=responsive_count,
                low_count=1,
                family=f"family-{index % 3}",
                risk=risks[index % 3],
            )
            index += 1
    audit = audit_opportunity_coverage(pd.DataFrame(rows))

    assert audit["status"] == "pass"
    assert audit["standard_eligible_event_count"] == 182
    assert audit["event_tiers"] == {
        "standard_4plus": 182,
        "short_3": 56,
        "short_2": 6,
        "ineligible": 0,
    }
    assert audit["checks"]["all_events_meet_duration_aware_target"] is True
    assert audit["legacy_four_responsive_all_events"] is False
    assert audit["legacy_check_deprecated"] is True
    assert audit["deprecated_reason"] == "planner_auditor_contract_conflict"


def test_audit_checks_are_native_bool() -> None:
    rows = []
    for index in range(8):
        rows += _event_rows(
            f"e{index}",
            responsive_count=4,
            low_count=1,
            family=f"family-{index % 3}",
            risk=["low", "medium", "high"][index % 3],
        )
    audit = audit_opportunity_coverage(pd.DataFrame(rows))

    assert all(type(value) is bool for value in audit["checks"].values())
    assert type(audit["legacy_four_responsive_all_events"]) is bool
    assert type(audit["legacy_check_deprecated"]) is bool


def test_audit_json_has_no_stringified_booleans() -> None:
    rows = []
    for index in range(8):
        rows += _event_rows(
            f"e{index}",
            responsive_count=4,
            low_count=1,
            family=f"family-{index % 3}",
            risk=["low", "medium", "high"][index % 3],
        )
    audit = audit_opportunity_coverage(pd.DataFrame(rows))
    text = json.dumps(audit)

    assert '"True"' not in text
    assert '"False"' not in text
