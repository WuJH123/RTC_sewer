"""Native SWMM rule text helpers for Project6 V3.

These helpers deliberately separate preliminary text references from confirmed
THEN/ELSE action clauses. They are not a full SWMM rule engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NativeActionReference:
    object_id: str
    clause: str
    rule_name: str | None = None
    priority: str | None = None


def strip_swmm_comments(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        lines.append(raw.split(";", 1)[0])
    return "\n".join(lines)


def token_regex(token: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(token)}(?![A-Za-z0-9_.-])", re.IGNORECASE)


def preliminary_text_reference(control_text: str, object_id: str) -> bool:
    return bool(token_regex(object_id).search(strip_swmm_comments(control_text)))


def extract_action_clause_references(control_text: str, object_ids: Iterable[str]) -> list[NativeActionReference]:
    clean = strip_swmm_comments(control_text)
    object_ids = list(object_ids)
    references: list[NativeActionReference] = []
    current_rule: str | None = None
    current_priority: str | None = None
    for raw in clean.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].upper()
        if keyword == "RULE" and len(parts) > 1:
            current_rule = parts[1]
            current_priority = None
            continue
        if keyword == "PRIORITY" and len(parts) > 1:
            current_priority = parts[1]
            continue
        if keyword not in {"THEN", "ELSE"}:
            continue
        for object_id in object_ids:
            if token_regex(object_id).search(line):
                references.append(
                    NativeActionReference(
                        object_id=object_id,
                        clause=keyword,
                        rule_name=current_rule,
                        priority=current_priority,
                    )
                )
    return references
