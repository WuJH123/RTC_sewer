from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from sewerrtc.io.inp_parser import read_sections


ACTION_RE = re.compile(r"^\s*(THEN|ELSE|AND)\s+(?:LINK|PUMP|ORIFICE|WEIR|OUTLET)\s+(\S+)\s+(SETTING|STATUS)\s*=\s*(\S+)", re.I)
COND_RE = re.compile(r"^\s*(IF|AND|OR)\s+(.+)$", re.I)
PRIORITY_RE = re.compile(r"^\s*PRIORITY\s+(\S+)", re.I)
RULE_RE = re.compile(r"^\s*RULE\s+(\S+)", re.I)


@dataclass
class ParsedRuleAction:
    rule_name: str
    clause: str
    actuator_id: str
    actuator_type: str
    target_kind: str
    target_value: str
    priority: str
    source_line_start: int
    source_line_end: int
    rule_hash: str


@dataclass
class ParsedRuleCondition:
    rule_name: str
    clause: str
    normalized_condition: str
    source_line: int
    rule_hash: str


def _clean(raw: str) -> str:
    return raw.split(";", 1)[0].strip()


def parse_swmm_controls(inp_path: str | Path) -> dict[str, Any]:
    sections = read_sections(inp_path)
    raw_lines = sections.get("CONTROLS", [])
    actions: list[ParsedRuleAction] = []
    conditions: list[ParsedRuleCondition] = []
    rules: dict[str, dict[str, Any]] = {}
    current_rule = ""
    current_priority = ""
    current_rule_lines: list[str] = []

    def rule_hash(lines: list[str]) -> str:
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    for i, raw in enumerate(raw_lines, start=1):
        line = _clean(raw)
        if not line:
            continue
        m_rule = RULE_RE.match(line)
        if m_rule:
            current_rule = m_rule.group(1)
            current_priority = ""
            current_rule_lines = [line]
            rules.setdefault(current_rule, {"rule_name": current_rule, "source_start": i, "actions": [], "conditions": [], "priority": ""})
            continue
        if not current_rule:
            continue
        current_rule_lines.append(line)
        r_hash = rule_hash(current_rule_lines)
        m_pr = PRIORITY_RE.match(line)
        if m_pr:
            current_priority = m_pr.group(1)
            rules[current_rule]["priority"] = current_priority
            continue
        m_action = ACTION_RE.match(line)
        if m_action:
            clause, actuator_id, target_kind, value = m_action.groups()
            action = ParsedRuleAction(
                rule_name=current_rule,
                clause=clause.upper(),
                actuator_id=actuator_id,
                actuator_type="link",
                target_kind=target_kind.upper(),
                target_value=value,
                priority=current_priority,
                source_line_start=i,
                source_line_end=i,
                rule_hash=r_hash,
            )
            actions.append(action)
            rules[current_rule]["actions"].append(asdict(action))
            continue
        m_cond = COND_RE.match(line)
        if m_cond:
            condition = ParsedRuleCondition(current_rule, m_cond.group(1).upper(), " ".join(m_cond.group(2).split()), i, r_hash)
            conditions.append(condition)
            rules[current_rule]["conditions"].append(asdict(condition))
    return {
        "rules": list(rules.values()),
        "actions": [asdict(a) for a in actions],
        "conditions": [asdict(c) for c in conditions],
        "parser_status": "completed",
    }
