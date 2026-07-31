from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from sewerrtc.contracts.prompt3a import managed_facility_ids, sha256_file, utc_now, write_csv, write_json
from sewerrtc.contracts.swmm_control_parser import parse_swmm_controls
from sewerrtc.io.inp_parser import parse_links, read_sections


CONFLICT_COLUMNS = [
    "facility_id",
    "rule_a",
    "rule_b",
    "action_a",
    "action_b",
    "priority_a",
    "priority_b",
    "source_order_a",
    "source_order_b",
    "condition_relation",
    "conflict_type",
    "resolution_rule",
    "blocking_status",
    "evidence",
]


def _priority_rank(value: str) -> tuple[int, float]:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return (0, float("-inf"))
    try:
        return (1, float(value))
    except ValueError:
        return (1, float("-inf"))


def _action_value(action: dict[str, Any]) -> str:
    return f"{action.get('target_kind', '')}={action.get('target_value', '')}"


def _condition_relation(a: dict[str, Any], b: dict[str, Any]) -> str:
    if a.get("rule_name") == b.get("rule_name") and {a.get("clause"), b.get("clause")} == {"THEN", "ELSE"}:
        return "mutually_exclusive_conditions"
    if a.get("rule_name") == b.get("rule_name"):
        return "same_rule_static_branch"
    return "potential_runtime_overlap"


def _classify_action_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    action_a = _action_value(a)
    action_b = _action_value(b)
    same_action = action_a == action_b
    relation = _condition_relation(a, b)
    pr_a = _priority_rank(str(a.get("priority", "")))
    pr_b = _priority_rank(str(b.get("priority", "")))
    line_a = int(a.get("source_line_start", 0) or 0)
    line_b = int(b.get("source_line_start", 0) or 0)
    if relation == "mutually_exclusive_conditions":
        conflict_type = "mutually_exclusive_conditions"
        resolution = "then_else_branches_are_not_simultaneous"
        blocking = "resolved"
    elif same_action and pr_a == pr_b:
        conflict_type = "same_priority_same_action"
        resolution = "same_action_no_conflict"
        blocking = "resolved"
    elif pr_a != pr_b:
        conflict_type = "different_priority_actions"
        winner = "rule_a" if pr_a > pr_b else "rule_b"
        resolution = f"resolved_by_higher_priority:{winner}"
        blocking = "resolved"
    elif line_a != line_b:
        conflict_type = "same_priority_contradictory_actions" if not same_action else "same_priority_same_action"
        winner = "rule_a" if line_a < line_b else "rule_b"
        resolution = f"resolved_by_source_order:{winner}"
        blocking = "resolved"
    else:
        conflict_type = "unresolved_static_overlap" if not same_action else "same_priority_same_action"
        resolution = "manual_review_required" if not same_action else "same_action_no_conflict"
        blocking = "blocking" if not same_action else "resolved"
    return {
        "facility_id": a.get("actuator_id", ""),
        "rule_a": a.get("rule_name", ""),
        "rule_b": b.get("rule_name", ""),
        "action_a": action_a,
        "action_b": action_b,
        "priority_a": a.get("priority", ""),
        "priority_b": b.get("priority", ""),
        "source_order_a": line_a,
        "source_order_b": line_b,
        "condition_relation": relation,
        "conflict_type": conflict_type,
        "resolution_rule": resolution,
        "blocking_status": blocking,
        "evidence": f"{a.get('rule_name','')}:{a.get('clause','')}:{line_a}|{b.get('rule_name','')}:{b.get('clause','')}:{line_b}",
    }


def audit_native_rules(inp_path: str | Path, out_dir: str | Path) -> tuple[int, dict[str, Any], list[Path]]:
    out_dir = Path(out_dir)
    parsed = parse_swmm_controls(inp_path)
    sections = read_sections(inp_path)
    links = parse_links(sections)
    link_types = {str(row["link_id"]): str(row["link_type"]) for _, row in links.iterrows()} if not links.empty else {}
    managed = managed_facility_ids()
    actions = parsed["actions"]
    conditions = parsed["conditions"]
    by_facility: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        by_facility[action["actuator_id"]].append(action)
    controlled_rows = []
    for fid in managed:
        acts = by_facility.get(fid, [])
        controlled_rows.append(
            {
                "facility_id": fid,
                "actuator_type": link_types.get(fid, ""),
                "native_action_clause_confirmed": bool(acts),
                "native_rule_names": "|".join(sorted({a["rule_name"] for a in acts})),
                "native_action_clause_count": len(acts),
                "native_rule_priorities": "|".join(sorted({a.get("priority", "") for a in acts if a.get("priority")})),
                "native_action_domain": "|".join(sorted({f"{a['target_kind']}={a['target_value']}" for a in acts})),
                "condition_dependencies": "",
                "conflict_status": "multiple_actions" if len(acts) > 1 else "none",
            }
        )
    conflicts = []
    for facility_id, acts in by_facility.items():
        if len(acts) < 2:
            continue
        for a, b in combinations(acts, 2):
            row = _classify_action_pair(a, b)
            row["conflict_type"] = row["conflict_type"] or "multiple_rules_same_facility"
            conflicts.append(row)
    unresolved_same_priority = [row for row in conflicts if row["blocking_status"] == "blocking"]
    potential_conflicts = [row for row in conflicts if row["condition_relation"] == "potential_runtime_overlap"]
    resolved_by_priority = [row for row in conflicts if str(row["resolution_rule"]).startswith("resolved_by_higher_priority")]
    files = [
        write_json(out_dir / "native_rules_parsed.json", {"created_at": utc_now(), "source_inp": str(inp_path), "source_sha256": sha256_file(Path(inp_path)), **parsed}),
        write_csv(out_dir / "native_rule_actions.csv", actions),
        write_csv(out_dir / "native_rule_conditions.csv", conditions),
        write_csv(out_dir / "native_rule_conflicts.csv", conflicts, CONFLICT_COLUMNS),
        write_csv(out_dir / "native_controlled_facilities.csv", controlled_rows),
    ]
    report = {
        "status": "completed",
        "source_inp": str(inp_path),
        "source_sha256": sha256_file(Path(inp_path)),
        "rule_count": len(parsed["rules"]),
        "action_clause_count": len(actions),
        "managed_facilities_with_native_actions": sum(1 for row in controlled_rows if row["native_action_clause_confirmed"]),
        "conflict_count": len(conflicts),
        "facilities_with_multiple_rules": len({row["facility_id"] for row in conflicts}),
        "potential_conflicts": len(potential_conflicts),
        "resolved_by_priority": len(resolved_by_priority),
        "unresolved_same_priority_conflicts": len(unresolved_same_priority),
        "confirmed_runtime_conflicts": 0,
        "internal_fallback_blocked_by_native_rules": bool(unresolved_same_priority),
        "preliminary_text_reference_not_used_for_final_control": True,
    }
    files.append(write_json(out_dir / "native_rule_audit_report.json", report))
    return 0, report, files
