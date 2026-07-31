from __future__ import annotations

import pandas as pd


def state_key(event_id: str, checkpoint_id: str, state_sha256: str) -> str:
    return f"{event_id}|{checkpoint_id}|{state_sha256}"


def sample_key(event_id: str, checkpoint_id: str, actual_sha256: str) -> str:
    return f"{event_id}|{checkpoint_id}|{actual_sha256}"


def accounting_summary(
    planned: int,
    *,
    accepted: int = 0,
    rejected: int = 0,
    pending: int = 0,
    missing: int = 0,
) -> dict:
    closed = int(planned) == sum(
        map(int, (accepted, rejected, pending, missing))
    )
    return {
        "planned": int(planned),
        "accepted": int(accepted),
        "rejected": int(rejected),
        "pending": int(pending),
        "missing": int(missing),
        "accounting_closed": closed,
    }


def partial_accounting_summary(
    planned: int,
    *,
    accepted: int = 0,
    rejected: int = 0,
    pending: int = 0,
    missing_confirmed: int = 0,
) -> dict:
    """Partial-mode accounting: pending is future work, never missing.

    ``missing_confirmed`` covers only completed cases whose recorded detail
    artifact disappeared; ``scope_complete`` is always False because a
    partial snapshot never certifies the full plan scope.
    """
    summary = accounting_summary(
        planned,
        accepted=accepted,
        rejected=rejected,
        pending=pending,
        missing=missing_confirmed,
    )
    summary["missing_confirmed"] = int(missing_confirmed)
    summary["remaining"] = int(pending)
    summary["scope_complete"] = False
    summary["partial_only"] = True
    return summary


def deduplicate_actual_schedules(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["event_id", "checkpoint_id", "actual_schedule_sha256"]
    missing = set(keys) - set(frame)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    duplicate = frame.duplicated(keys, keep="first")
    accepted = frame[~duplicate].copy()
    rejected = frame[duplicate].copy()
    if len(rejected):
        rejected["rejection_reason"] = "duplicate_actual_schedule"
    return accepted, rejected


def validate_sample_contract(sample: dict) -> dict:
    required_branches = {
        "candidate",
        "no_control",
        "dynamic_internal_rules",
        "hold_previous",
    }
    action_fields = (
        "requested",
        "projected",
        "written",
        "target",
        "current",
        "readback",
    )
    checks = {
        "state_key_complete": all(
            sample.get(key)
            for key in (
                "event_id",
                "checkpoint_id",
                "checkpoint_state_sha256",
            )
        ),
        "sample_key_complete": bool(sample.get("actual_schedule_sha256")),
        "four_branches": set(sample.get("branches", [])) == required_branches,
        "action_stages_separate": all(
            sample.get(f"{field}_schedule_path") for field in action_fields
        ),
        "not_noop": not bool(sample.get("no_op", True)),
        "candidate_differs_from_reference": not bool(
            sample.get("candidate_matches_reference", True)
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
    }
