from __future__ import annotations

from collections import deque

import pandas as pd


def _adjacency(link_table: pd.DataFrame, reverse: bool = False) -> dict[str, list[tuple[str, str]]]:
    cols = {c.lower(): c for c in link_table.columns}
    from_col = cols.get("from_node") or cols.get("from")
    to_col = cols.get("to_node") or cols.get("to")
    id_col = cols.get("link_id") or cols.get("id") or cols.get("name")
    if not from_col or not to_col:
        return {}
    adj: dict[str, list[tuple[str, str]]] = {}
    for _, r in link_table.iterrows():
        u = str(r[from_col])
        v = str(r[to_col])
        lid = str(r[id_col]) if id_col else ""
        a, b = (v, u) if reverse else (u, v)
        adj.setdefault(a, []).append((b, lid))
    return adj


def khop_domain(link_table: pd.DataFrame, start: str, k: int, direction: str) -> dict[str, int]:
    adj = _adjacency(link_table, reverse=direction == "upstream")
    seen = {str(start): 0}
    q = deque([str(start)])
    while q:
        u = q.popleft()
        if seen[u] >= int(k):
            continue
        for v, _ in adj.get(u, []):
            if v in seen:
                continue
            seen[v] = seen[u] + 1
            q.append(v)
    return seen


def build_priority_influence_domains(
    link_table: pd.DataFrame,
    actuator_table: pd.DataFrame,
    priority_nodes: list[str],
    k: int = 3,
    fallback_k: int = 12,
    max_candidates_per_priority: int = 24,
    include_global_storage_controls: bool = True,
    include_global_regulators: bool = True,
    include_global_pumps: bool = True,
    max_storage_controls_per_priority: int = 10,
    max_regulators_per_priority: int = 48,
    max_pumps_per_priority: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    act = actuator_table.copy()
    act_node_cols = [c for c in ["from_node", "to_node", "node_id", "upstream_node", "downstream_node"] if c in act.columns]
    domains = []
    candidates = []
    for pn in priority_nodes:
        up = khop_domain(link_table, pn, k, "upstream")
        down = khop_domain(link_table, pn, k, "downstream")
        for node, dist in up.items():
            domains.append({"priority_node": pn, "node_id": node, "direction": "upstream", "hop_distance": dist})
        for node, dist in down.items():
            domains.append({"priority_node": pn, "node_id": node, "direction": "downstream", "hop_distance": dist})
        domain_nodes = set(up) | set(down)
        extended_up = khop_domain(link_table, pn, max(int(k), int(fallback_k)), "upstream")
        extended_down = khop_domain(link_table, pn, max(int(k), int(fallback_k)), "downstream")
        priority_candidates: list[dict] = []

        def _asset_role(row: pd.Series) -> str:
            storage_role = str(row.get("storage_control_type", "") or "").lower()
            if storage_role and storage_role.lower() not in {"nan", "none", "", "not_storage"}:
                return storage_role
            text = " ".join([str(row.get("asset_role", "")), str(row.get("link_type", ""))]).lower()
            if "pump" in text:
                return "pump"
            if "weir" in text:
                return "weir"
            if "orifice" in text or "regulator" in text:
                return "orifice"
            return str(row.get("asset_role", row.get("link_type", "unknown"))).lower()

        def _is_control_relevant(row: pd.Series) -> bool:
            text = " ".join(
                [
                    str(row.get("link_type", "")),
                    str(row.get("asset_role", "")),
                    str(row.get("storage_control_type", "")),
                ]
            ).lower()
            return any(tok in text for tok in ("storage", "pump", "orifice", "weir"))

        def _role_group(role: str) -> str:
            role_text = str(role or "").lower()
            if "storage" in role_text or "inlet" in role_text or "outlet" in role_text:
                return "storage"
            if "pump" in role_text:
                return "pump"
            if "orifice" in role_text or "weir" in role_text or "regulator" in role_text:
                return "regulator"
            return "other"

        def _best_match_for_row(row: pd.Series, max_dist: int | None = None) -> tuple[str, str, int]:
            touched = []
            for c in act_node_cols:
                val = str(row.get(c, ""))
                dist = min(extended_up.get(val, 999), extended_down.get(val, 999))
                if max_dist is None or dist <= int(max_dist):
                    touched.append((c, val, dist))
            if touched:
                return sorted(touched, key=lambda x: (x[2], x[0], x[1]))[0]
            for c in act_node_cols:
                val = str(row.get(c, ""))
                if val and val.lower() != "nan":
                    return (c, val, int(fallback_k) + 1)
            return ("", "", int(fallback_k) + 1)

        def _add_candidate(row: pd.Series, best: tuple[str, str, int], source: str) -> None:
            aid = str(row.get("actuator_id", row.get("link_id", "")))
            if not aid or aid.lower() == "nan":
                return
            if aid in {str(c.get("actuator_id", "")) for c in priority_candidates}:
                return
            role = _asset_role(row)
            priority_candidates.append(
                {
                    "priority_node": pn,
                    "actuator_id": aid,
                    "asset_role": role,
                    "matched_node_field": best[0],
                    "matched_node": best[1],
                    "influence_path_length": int(best[2]),
                    "candidate_source": source,
                    "physical_rationale": f"{role} actuator is within {best[2]} hops of priority node {pn}.",
                }
            )

        for _, a in act.iterrows():
            touched = []
            for c in act_node_cols:
                val = str(a.get(c, ""))
                if val in domain_nodes:
                    touched.append((c, val, min(up.get(val, 999), down.get(val, 999))))
            if not touched:
                continue
            best = sorted(touched, key=lambda x: x[2])[0]
            _add_candidate(a, best, "primary_khop")

        for _, a in act.iterrows():
            aid = str(a.get("actuator_id", a.get("link_id", "")))
            existing = {str(c.get("actuator_id", "")) for c in priority_candidates}
            if aid in existing or not _is_control_relevant(a):
                continue
            touched = []
            for c in act_node_cols:
                val = str(a.get(c, ""))
                dist = min(extended_up.get(val, 999), extended_down.get(val, 999))
                if dist <= int(fallback_k):
                    touched.append((c, val, dist))
            if not touched:
                continue
            best = sorted(touched, key=lambda x: x[2])[0]
            _add_candidate(a, best, "extended_hydraulic_domain")

        def _ranked_pool(role_group: str) -> list[tuple[int, str, pd.Series]]:
            rows: list[tuple[int, str, pd.Series]] = []
            for _, row in act.iterrows():
                role = _asset_role(row)
                if _role_group(role) != role_group:
                    continue
                best = _best_match_for_row(row)
                rows.append((int(best[2]), str(row.get("actuator_id", row.get("link_id", ""))), row))
            return sorted(rows, key=lambda item: (item[0], item[1]))

        def _add_global_pool(role_group: str, limit: int, source: str) -> None:
            if int(limit) <= 0:
                return
            added = 0
            for _, _, row in _ranked_pool(role_group):
                existing = {str(c.get("actuator_id", "")) for c in priority_candidates}
                aid = str(row.get("actuator_id", row.get("link_id", "")))
                if aid in existing:
                    continue
                best = _best_match_for_row(row)
                _add_candidate(row, best, source)
                added += 1
                if added >= int(limit):
                    break

        if include_global_storage_controls:
            _add_global_pool("storage", int(max_storage_controls_per_priority), "global_storage_control_pool")
        if include_global_regulators:
            _add_global_pool("regulator", int(max_regulators_per_priority), "global_regulator_pool")
        if include_global_pumps:
            _add_global_pool("pump", int(max_pumps_per_priority), "global_pump_pool")

        priority_candidates = sorted(
            priority_candidates,
            key=lambda r: (
                0
                if str(r.get("candidate_source", "")) == "primary_khop"
                else (1 if str(r.get("candidate_source", "")) == "extended_hydraulic_domain" else 2),
                0
                if _role_group(str(r.get("asset_role", ""))) == "storage"
                else (1 if _role_group(str(r.get("asset_role", ""))) == "regulator" else 2),
                int(r.get("influence_path_length", 999)),
                str(r.get("actuator_id", "")),
            ),
        )
        if int(max_candidates_per_priority) > 0:
            cap = int(max_candidates_per_priority)
            pump_rows = [
                row for row in priority_candidates
                if _role_group(str(row.get("asset_role", ""))) == "pump"
            ]
            other_rows = [
                row for row in priority_candidates
                if _role_group(str(row.get("asset_role", ""))) != "pump"
            ]
            priority_candidates = (pump_rows + other_rows)[:cap]
        candidates.extend(priority_candidates)
    return pd.DataFrame(domains), pd.DataFrame(candidates).drop_duplicates()
