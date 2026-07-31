"""Event usage ledger and deterministic split selection for Final V4.

Every event/rainfall SHA owns exactly one ledger row.  A rainfall SHA may
belong to at most one formal split, all checkpoints and candidates of one
event share that split, and events are never deleted based on outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


LEDGER_COLUMNS = (
    "event_id",
    "rainfall_sha256",
    "event_tier",
    "opportunity_scanned",
    "used_gate2",
    "used_gate3",
    "used_gate4",
    "used_gate5r",
    "used_peak_boundary",
    "used_pilot",
    "used_train",
    "used_calibration",
    "used_locked_validation",
    "used_challenge",
    "used_formal",
    "oracle_revealed",
    "policy_tuned_on_event",
    "formal_eligible",
    "exclusion_reason",
    "assigned_split",
    "assignment_run_uuid",
)

USAGE_FLAG_COLUMNS = tuple(
    column
    for column in LEDGER_COLUMNS
    if column.startswith("used_")
    or column
    in ("opportunity_scanned", "oracle_revealed", "policy_tuned_on_event")
)

SPLIT_TO_USAGE = {
    "pilot": "used_pilot",
    "train": "used_train",
    "calibration": "used_calibration",
    "locked_validation": "used_locked_validation",
    "challenge": "used_challenge",
    "formal": "used_formal",
}


class EventShortfallError(ValueError):
    """Raised fail-closed when the standard event inventory is insufficient."""

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


def build_event_usage_ledger(
    tier_catalog: pd.DataFrame,
    *,
    scanned_event_ids: set[str],
    assignment_run_uuid: str = "",
) -> pd.DataFrame:
    """One row per event/rainfall SHA with fail-closed formal eligibility.

    All Opportunity-scanned events default to development-only:
    ``opportunity_scanned=True`` forces ``formal_eligible=False``.
    """
    required = {"event_id", "rainfall_sha256", "event_tier"}
    missing = required - set(tier_catalog)
    if missing:
        raise ValueError(f"tier catalog missing columns: {sorted(missing)}")
    events = tier_catalog.drop_duplicates("event_id").copy()
    if events["rainfall_sha256"].astype(str).duplicated().any():
        raise ValueError("duplicate rainfall_sha256 across events")
    ledger = pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str).to_numpy(),
            "rainfall_sha256": events["rainfall_sha256"].astype(str).to_numpy(),
            "event_tier": events["event_tier"].astype(str).to_numpy(),
        }
    )
    scanned = ledger["event_id"].isin({str(item) for item in scanned_event_ids})
    ledger["opportunity_scanned"] = scanned
    for column in USAGE_FLAG_COLUMNS:
        if column not in ledger:
            ledger[column] = False
    ledger["formal_eligible"] = ~ledger["opportunity_scanned"]
    ledger["exclusion_reason"] = np.where(
        ledger["opportunity_scanned"],
        "opportunity_scanned_development_only",
        "",
    )
    ledger["assigned_split"] = ""
    ledger["assignment_run_uuid"] = str(assignment_run_uuid)
    return ledger[list(LEDGER_COLUMNS)].reset_index(drop=True)


def validate_ledger(ledger: pd.DataFrame) -> None:
    missing = set(LEDGER_COLUMNS) - set(ledger)
    if missing:
        raise ValueError(f"ledger missing columns: {sorted(missing)}")
    if ledger["event_id"].astype(str).duplicated().any():
        raise ValueError("ledger has duplicate event_id rows")
    if ledger["rainfall_sha256"].astype(str).duplicated().any():
        raise ValueError("ledger has duplicate rainfall_sha256 rows")
    formal_bad = ledger["formal_eligible"].astype(bool) & (
        ledger["opportunity_scanned"].astype(bool)
        | ledger["oracle_revealed"].astype(bool)
        | ledger["policy_tuned_on_event"].astype(bool)
    )
    if formal_bad.any():
        raise ValueError("formal_eligible rows overlap development usage")


def assign_split(
    ledger: pd.DataFrame,
    event_ids: list[str],
    split: str,
    *,
    assignment_run_uuid: str,
) -> pd.DataFrame:
    """Freeze one split for the given events (idempotent, never reassigns)."""
    if split not in SPLIT_TO_USAGE and split != "reserve":
        raise ValueError(f"unknown split: {split}")
    validate_ledger(ledger)
    result = ledger.copy()
    wanted = result["event_id"].astype(str).isin({str(e) for e in event_ids})
    if int(wanted.sum()) != len(set(map(str, event_ids))):
        missing = set(map(str, event_ids)) - set(
            result.loc[wanted, "event_id"].astype(str)
        )
        raise ValueError(f"events missing from ledger: {sorted(missing)}")
    already = result.loc[wanted, "assigned_split"].astype(str)
    conflict = already[(already != "") & (already != split)]
    if len(conflict):
        raise ValueError(
            "events already frozen in another split: "
            f"{sorted(result.loc[conflict.index, 'event_id'].astype(str))}"
        )
    result.loc[wanted, "assigned_split"] = split
    result.loc[wanted, "assignment_run_uuid"] = str(assignment_run_uuid)
    usage_column = SPLIT_TO_USAGE.get(split)
    if usage_column:
        result.loc[wanted, usage_column] = True
    if split == "formal":
        bad = result.loc[wanted]
        if bad["opportunity_scanned"].astype(bool).any() or not bad[
            "formal_eligible"
        ].astype(bool).all():
            raise ValueError("formal split may only use formal_eligible events")
    return result


def _deterministic_order(event_ids: pd.Series, seed: int) -> list[str]:
    ordered = sorted(event_ids.astype(str).unique())
    permutation = np.random.default_rng(int(seed)).permutation(len(ordered))
    return [ordered[index] for index in permutation]


def _event_features(standard_catalog: pd.DataFrame) -> pd.DataFrame:
    """Per-event diversity features derived from the canonical catalog."""
    grouped = standard_catalog.groupby("event_id")
    features = pd.DataFrame(
        {
            "rainfall_sha256": grouped["rainfall_sha256"].first().astype(str),
            "rainfall_family": (
                grouped["rainfall_family"].first().astype(str)
                if "rainfall_family" in standard_catalog
                else "unknown"
            ),
            "risk_level": (
                grouped["risk_level"]
                .agg(lambda values: values.mode().iloc[0])
                .astype(str)
                if "risk_level" in standard_catalog
                else "unknown"
            ),
            "duration_bucket": (
                (grouped["elapsed_min"].max() // 60.0).astype(int)
            ),
        }
    ).reset_index()
    return features


def select_pilot_events(
    standard_catalog: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    count: int = 8,
    seed: int = 20260727,
) -> list[str]:
    """Deterministically select development events for Pilot400.

    Constraints: never events already frozen in locked_validation, challenge
    or formal; unique rainfall SHA; at least 3 rainfall families, 3 risk
    levels and multiple duration buckets across the selection.
    """
    validate_ledger(ledger)
    features = _event_features(standard_catalog)
    blocked_splits = {"locked_validation", "challenge", "formal"}
    ledger_indexed = ledger.set_index("event_id")
    usable = []
    for event_id in features["event_id"].astype(str):
        if event_id not in ledger_indexed.index:
            continue
        row = ledger_indexed.loc[event_id]
        assigned = str(row["assigned_split"])
        if assigned in blocked_splits:
            continue
        if assigned not in ("", "pilot"):
            continue
        if bool(row["used_locked_validation"]) or bool(
            row["used_challenge"]
        ) or bool(row["used_formal"]):
            continue
        usable.append(event_id)
    if len(usable) < int(count):
        raise ValueError(
            f"need {count} pilot-eligible events, found {len(usable)}"
        )
    features = features[features["event_id"].astype(str).isin(usable)]
    features = features.set_index("event_id")
    order = [
        event_id
        for event_id in _deterministic_order(
            pd.Series(usable), seed
        )
    ]
    selected: list[str] = []
    families: set[str] = set()
    risks: set[str] = set()
    durations: set[int] = set()

    def coverage_gain(event_id: str) -> int:
        row = features.loc[event_id]
        return (
            int(str(row["rainfall_family"]) not in families)
            + int(str(row["risk_level"]) not in risks)
            + int(int(row["duration_bucket"]) not in durations)
        )

    remaining = list(order)
    while len(selected) < int(count) and remaining:
        # Greedy diversity-first, deterministic tie-break by permutation order.
        best = max(remaining, key=lambda item: (coverage_gain(item),))
        if coverage_gain(best) == 0:
            best = remaining[0]
        selected.append(best)
        remaining.remove(best)
        row = features.loc[best]
        families.add(str(row["rainfall_family"]))
        risks.add(str(row["risk_level"]))
        durations.add(int(row["duration_bucket"]))
    if len(selected) != int(count):
        raise ValueError("pilot event selection incomplete")
    shas = features.loc[selected, "rainfall_sha256"].astype(str)
    if shas.duplicated().any():
        raise ValueError("pilot selection has duplicate rainfall SHA")
    if len(families) < 3 or len(risks) < 3 or len(durations) < 2:
        raise ValueError(
            "pilot selection cannot satisfy diversity constraints: "
            f"families={len(families)} risks={len(risks)} "
            f"durations={len(durations)}"
        )
    return selected


def select_train1600_events(
    standard_catalog: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    counts: dict[str, int] | None = None,
    seed: int = 20260727,
) -> dict[str, list[str]]:
    """Deterministic Train1600 event split selection with fail-closed shortfall.

    Excludes pilot events, gate-tuning events, challenge/formal reserved
    events, duplicate rainfall SHA, and events without the 4+1 structure.
    Never pads with short events, never reuses pilot events, never clones
    events or checkpoints.
    """
    counts = dict(
        counts
        or {"train": 48, "calibration": 8, "locked_validation": 8, "reserve": 16}
    )
    validate_ledger(ledger)
    features = _event_features(standard_catalog)
    grouped = standard_catalog.groupby("event_id")
    structure_ok = grouped["checkpoint_role"].apply(
        lambda values: int(values.eq("responsive").sum()) == 4
        and int(values.eq("low_opportunity").sum()) == 1
    )
    ledger_indexed = ledger.set_index("event_id")
    excluded: dict[str, str] = {}
    usable: list[str] = []
    seen_sha: set[str] = set()
    for event_id in sorted(features["event_id"].astype(str)):
        row = (
            ledger_indexed.loc[event_id]
            if event_id in ledger_indexed.index
            else None
        )
        if row is None:
            excluded[event_id] = "missing_from_ledger"
            continue
        if not bool(structure_ok.get(event_id, False)):
            excluded[event_id] = "not_4plus1_structure"
            continue
        if str(row["event_tier"]) != "standard_4plus":
            excluded[event_id] = "not_standard_4plus"
            continue
        if bool(row["used_pilot"]) or str(row["assigned_split"]) == "pilot":
            excluded[event_id] = "pilot_event"
            continue
        if bool(row["used_gate5r"]) or bool(row["used_peak_boundary"]) or bool(
            row["policy_tuned_on_event"]
        ):
            excluded[event_id] = "gate_tuning_event"
            continue
        if bool(row["used_challenge"]) or bool(row["used_formal"]) or str(
            row["assigned_split"]
        ) in ("challenge", "formal"):
            excluded[event_id] = "challenge_or_formal_reserved"
            continue
        if str(row["assigned_split"]) not in ("",):
            excluded[event_id] = f"already_{row['assigned_split']}"
            continue
        sha = str(row["rainfall_sha256"])
        if sha in seen_sha:
            excluded[event_id] = "duplicate_rainfall_sha"
            continue
        seen_sha.add(sha)
        usable.append(event_id)
    total = sum(int(value) for value in counts.values())
    if len(usable) < total:
        report = {
            "required_events": int(total),
            "usable_events": int(len(usable)),
            "shortfall": int(total - len(usable)),
            "counts": {key: int(value) for key, value in counts.items()},
            "excluded_events": excluded,
            "policy": {
                "no_short_event_padding": True,
                "no_pilot_reuse": True,
                "no_event_or_checkpoint_cloning": True,
            },
        }
        raise EventShortfallError(
            f"need {total} standard events, found {len(usable)}", report
        )
    order = _deterministic_order(pd.Series(usable), seed)
    ordered_usable = [event_id for event_id in order if event_id in set(usable)]
    result: dict[str, list[str]] = {}
    cursor = 0
    for split in ("train", "calibration", "locked_validation", "reserve"):
        span = int(counts[split])
        result[split] = ordered_usable[cursor : cursor + span]
        cursor += span
    return result


def select_formal_blind_candidates(ledger: pd.DataFrame) -> pd.DataFrame:
    """Only never-scanned, never-revealed, never-used events qualify."""
    validate_ledger(ledger)
    mask = (
        ledger["formal_eligible"].astype(bool)
        & ~ledger["opportunity_scanned"].astype(bool)
        & ~ledger["oracle_revealed"].astype(bool)
        & ~ledger["policy_tuned_on_event"].astype(bool)
    )
    for column in (
        "used_pilot",
        "used_train",
        "used_calibration",
        "used_locked_validation",
        "used_challenge",
        "used_gate2",
        "used_gate3",
        "used_gate4",
        "used_gate5r",
        "used_peak_boundary",
    ):
        mask &= ~ledger[column].astype(bool)
    return ledger[mask].copy()
