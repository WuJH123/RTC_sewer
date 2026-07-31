from __future__ import annotations

from collections import defaultdict
from typing import Any


FORBIDDEN_DEV_SPLITS = {"gat_independent_holdout", "formal_blind", "calibration_a", "locked_validation_b"}


def assign_split(row: dict[str, Any]) -> str:
    if str(row.get("gat_independent_holdout", "")).lower() == "true":
        return "gat_independent_holdout"
    if str(row.get("formal_eligibility", "")).lower() == "true":
        return "formal_blind"
    return "development_fit"


def audit_split_leakage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event: dict[str, set[str]] = defaultdict(set)
    by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split") or row.get("intended_split") or assign_split(row)
        by_event[str(row.get("event_id", ""))].add(str(split))
        by_family[str(row.get("storm_family_id", ""))].add(str(split))
    findings = []
    for event_id, splits in by_event.items():
        if len(splits) > 1:
            findings.append({"level": "event", "id": event_id, "splits": "|".join(sorted(splits)), "status": "leakage"})
    for family, splits in by_family.items():
        major = {s for s in splits if s in {"action_effect_fit", "calibration_a", "locked_validation_b", "formal_blind"}}
        if len(major) > 1:
            findings.append({"level": "storm_family", "id": family, "splits": "|".join(sorted(splits)), "status": "leakage"})
    return findings

