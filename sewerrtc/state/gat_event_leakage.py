from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_CHECK_STATUSES = {"pass", "fail", "incomplete", "not_applicable"}


@dataclass(frozen=True)
class LeakageDecision:
    status: str
    rows: list[dict[str, Any]]
    blocking_reason: str


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _key(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = _norm(row.get(name))
        if value:
            return value
    return ""


def _event_identity_complete(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if not _key(row, "event_id", "canonical_event_id"):
            return False
        if not _key(row, "split"):
            return False
    return True


def compare_training_and_validation_events(
    training_events: list[dict[str, Any]],
    validation_events: list[dict[str, Any]],
    *,
    training_complete: bool,
    validation_complete: bool,
) -> LeakageDecision:
    """Four-state leakage audit.

    This is intentionally conservative: absence of an overlap is not enough for
    PASS unless both event sets are complete and event identities are usable.
    """
    rows: list[dict[str, Any]] = []
    if not training_complete or not validation_complete or not _event_identity_complete(training_events) or not _event_identity_complete(validation_events):
        rows.append(
            {
                "validation_event": "",
                "matching_training_event": "",
                "match_type": "event_set_incomplete",
                "evidence": "training or validation event identity/split is incomplete",
                "similarity": "",
                "decision": "incomplete",
                "blocking_status": "blocking",
            }
        )
        return LeakageDecision(
            status="incomplete",
            rows=rows,
            blocking_reason="training or validation event set is incomplete",
        )

    training_by_event = {_key(row, "event_id", "canonical_event_id"): row for row in training_events}
    training_by_rain = {
        _key(row, "rainfall_series_sha256", "rainfall_file_sha256"): row
        for row in training_events
        if _key(row, "rainfall_series_sha256", "rainfall_file_sha256")
    }
    training_by_trajectory = {
        _key(row, "trajectory_sha256"): row
        for row in training_events
        if _key(row, "trajectory_sha256")
    }
    training_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in training_events:
        family = _key(row, "storm_family_id")
        if family:
            training_by_family.setdefault(family, []).append(row)

    failed = False
    for val in validation_events:
        val_event = _key(val, "event_id", "canonical_event_id")
        val_rain = _key(val, "rainfall_series_sha256", "rainfall_file_sha256")
        val_traj = _key(val, "trajectory_sha256")
        val_family = _key(val, "storm_family_id")

        matches: list[tuple[str, dict[str, Any], str]] = []
        if val_event in training_by_event:
            matches.append(("exact_event_id", training_by_event[val_event], val_event))
        if val_rain and val_rain in training_by_rain:
            matches.append(("rainfall_series_or_file_hash", training_by_rain[val_rain], val_rain))
        if val_traj and val_traj in training_by_trajectory:
            matches.append(("trajectory_hash", training_by_trajectory[val_traj], val_traj))
        if val_family and val_family in training_by_family:
            for train_row in training_by_family[val_family]:
                matches.append(("storm_family_overlap", train_row, val_family))

        if matches:
            failed = True
            for match_type, train, evidence in matches:
                rows.append(
                    {
                        "validation_event": val_event,
                        "matching_training_event": _key(train, "event_id", "canonical_event_id"),
                        "match_type": match_type,
                        "evidence": evidence,
                        "similarity": 1.0,
                        "decision": "fail",
                        "blocking_status": "blocking",
                    }
                )

    if failed:
        return LeakageDecision(status="fail", rows=rows, blocking_reason="training/validation leakage detected")

    rows.append(
        {
            "validation_event": "",
            "matching_training_event": "",
            "match_type": "no_overlap_found_after_complete_search",
            "evidence": "event_id, rainfall hash, trajectory hash, and storm-family checks completed",
            "similarity": "",
            "decision": "pass",
            "blocking_status": "none",
        }
    )
    return LeakageDecision(status="pass", rows=rows, blocking_reason="")


def near_duplicate_rows(training_events: list[dict[str, Any]], validation_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report near-duplicate opportunities without inventing unavailable series analysis."""
    rows: list[dict[str, Any]] = []
    for val in validation_events:
        val_event = _key(val, "event_id", "canonical_event_id")
        val_family = _key(val, "storm_family_id")
        if not val_family:
            continue
        for train in training_events:
            if _key(train, "storm_family_id") == val_family:
                rows.append(
                    {
                        "validation_event": val_event,
                        "matching_training_event": _key(train, "event_id", "canonical_event_id"),
                        "match_type": "storm_family_candidate",
                        "scale_factor": "",
                        "residual_error": "",
                        "time_shift": "",
                        "evidence": val_family,
                        "decision": "review_or_fail_if_forbidden_by_contract",
                    }
                )
    if not rows:
        rows.append(
            {
                "validation_event": "",
                "matching_training_event": "",
                "match_type": "no_near_duplicate_evidence",
                "scale_factor": "",
                "residual_error": "",
                "time_shift": "",
                "evidence": "rainfall series metadata unavailable or no family overlap",
                "decision": "no_candidate",
            }
        )
    return rows

