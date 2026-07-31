from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class CoverageCell:
    event_id: str
    storm_family_id: str
    split: str
    checkpoint_id: str
    phase: str
    state_cluster: str
    anchor_type: str
    facility_group: str
    direction: str
    magnitude: str
    duration: str
    concurrency: str
    interaction_type: str
    gat_seen: str
    fallback_disagreement: str
    outcome_class: str
    decision_relevance: str
    unique_event_support: int = 0
    feasibility: str = "planned"
    status: str = "missing"


def rows(cells: Iterable[CoverageCell]) -> list[dict[str, object]]:
    return [asdict(cell) for cell in cells]

