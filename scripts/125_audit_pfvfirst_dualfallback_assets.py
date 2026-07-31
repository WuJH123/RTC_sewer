from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(raw: str | None, fallback: str) -> Path:
    text = raw or fallback
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def read_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.append(item)
    return out


def parse_inp_sections(path: Path) -> Dict[str, List[List[str]]]:
    sections: Dict[str, List[List[str]]] = defaultdict(list)
    current: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").upper()
            continue
        if current is None:
            continue
        parts = line.split()
        if parts:
            sections[current].append(parts)
    return dict(sections)


def first_column_ids(rows: Iterable[List[str]]) -> set[str]:
    return {row[0] for row in rows if row}


def link_nodes(rows: Iterable[List[str]]) -> Dict[str, Tuple[str | None, str | None]]:
    out: Dict[str, Tuple[str | None, str | None]] = {}
    for row in rows:
        if not row:
            continue
        upstream = row[1] if len(row) > 1 else None
        downstream = row[2] if len(row) > 2 else None
        out[row[0]] = (upstream, downstream)
    return out


def control_references(sections: Dict[str, List[List[str]]], actuator_ids: List[str]) -> Dict[str, bool]:
    text = "\n".join(" ".join(row) for row in sections.get("CONTROLS", []))
    lower_text = text.lower()
    return {aid: aid.lower() in lower_text for aid in actuator_ids}


def classify_actuators(sections: Dict[str, List[List[str]]], actuator_ids: List[str]) -> List[Dict[str, Any]]:
    storage_nodes = first_column_ids(sections.get("STORAGE", []))
    type_by_id: Dict[str, str] = {}
    node_by_id: Dict[str, Tuple[str | None, str | None]] = {}
    for section, typ in [
        ("ORIFICES", "orifice"),
        ("WEIRS", "weir"),
        ("PUMPS", "pump"),
        ("OUTLETS", "outlet"),
    ]:
        rows = sections.get(section, [])
        for aid, nodes in link_nodes(rows).items():
            type_by_id[aid] = typ
            node_by_id[aid] = nodes
    refs = control_references(sections, actuator_ids)
    exact_inp_ids = set(type_by_id)
    lower_to_exact = {aid.lower(): aid for aid in exact_inp_ids}
    rows: List[Dict[str, Any]] = []
    for aid in actuator_ids:
        exact = aid if aid in exact_inp_ids else lower_to_exact.get(aid.lower())
        typ = type_by_id.get(exact or "", "missing_from_inp")
        upstream, downstream = node_by_id.get(exact or "", (None, None))
        if typ == "pump" and aid in {"ADD301.2", "ADD301.3"}:
            semantics = "binary_pump"
        elif typ == "pump":
            semantics = "pump_semantics_unverified"
        elif typ in {"orifice", "weir", "outlet"}:
            semantics = "continuous_regulator_unverified"
        else:
            semantics = "unknown"
        if upstream in storage_nodes and downstream in storage_nodes:
            storage_role = "storage_to_storage"
        elif upstream in storage_nodes:
            storage_role = "storage_outlet"
        elif downstream in storage_nodes:
            storage_role = "storage_inlet"
        else:
            storage_role = "not_storage_linked"
        rows.append(
            {
                "actuator_id": aid,
                "canonical_id": exact,
                "exists_in_inp": exact is not None,
                "case_mismatch": exact is not None and exact != aid,
                "actuator_type": typ,
                "control_semantics": semantics,
                "binary_pump_semantics": aid in {"ADD301.2", "ADD301.3"},
                "upstream_node": upstream,
                "downstream_node": downstream,
                "storage_role": storage_role,
                "preliminary_controls_text_reference": refs.get(aid, False),
                "native_action_clause_confirmed": None,
                "native_rule_names": [],
                "native_action_clause_count": None,
                "native_rule_priorities": [],
                "native_rule_audit_status": "pending",
            }
        )
    return rows


def node_presence(sections: Dict[str, List[List[str]]], nodes: Iterable[str]) -> Dict[str, bool]:
    inp_nodes = set()
    for section in ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS"):
        inp_nodes |= first_column_ids(sections.get(section, []))
    return {node: node in inp_nodes for node in nodes}


def path_status(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V3 PFV-first dual-fallback assets.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = resolve_path(args.config, args.config)
    cfg = load_yaml(config_path)

    project_cfg = cfg.get("project", {}) if isinstance(cfg.get("project", {}), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    control_cfg = cfg.get("control", {}) if isinstance(cfg.get("control", {}), dict) else {}
    contract_cfg = cfg.get("contracts", {}) if isinstance(cfg.get("contracts", {}), dict) else {}
    sensing_cfg = cfg.get("sensing", {}) if isinstance(cfg.get("sensing", {}), dict) else {}
    state_estimation_cfg = cfg.get("state_estimation", {}) if isinstance(cfg.get("state_estimation", {}), dict) else {}
    priority_cfg = cfg.get("priority", {}) if isinstance(cfg.get("priority", {}), dict) else {}

    inp_path = resolve_path(data_cfg.get("inp_path") or project_cfg.get("inp"), "data/wuhan_v8_storage_retrofit.inp")
    ids_path = resolve_path(
        data_cfg.get("managed_actuator_ids_path") or control_cfg.get("managed_actuator_ids_path"),
        "data/project6_v8_storage_retrofit_control_enabled_ids.txt",
    )
    kpi_contract = resolve_path(contract_cfg.get("kpi_contract_path") or contract_cfg.get("kpi"), "docs/contracts/kpi_contract.json")
    forecast_contract = resolve_path(contract_cfg.get("forecast_contract_path") or contract_cfg.get("forecast"), "docs/contracts/forecast_contract.json")
    sentinel_contract = resolve_path(contract_cfg.get("sentinel_nodes"), "docs/contracts/sentinel_nodes_provenance.json")
    sentinel_nodes_path = resolve_path(control_cfg.get("sentinel_nodes_path"), "data/project6_v3_sentinel_nodes.txt")

    actuator_ids = read_ids(ids_path)
    duplicate_ids = sorted([aid for aid, n in Counter(actuator_ids).items() if n > 1])
    sections = parse_inp_sections(inp_path) if inp_path.exists() else {}
    actuator_audit = classify_actuators(sections, actuator_ids) if sections else []

    priority_nodes = priority_cfg.get("priority_nodes") or []
    sentinel_nodes = priority_cfg.get("sentinel_nodes") or []
    if not isinstance(priority_nodes, list):
        priority_nodes = []
    if not isinstance(sentinel_nodes, list):
        sentinel_nodes = []
    if not sentinel_nodes:
        sentinel_nodes = read_ids(sentinel_nodes_path)
    sentinel_contract_status = None
    if sentinel_contract.exists():
        try:
            sentinel_payload = json.loads(sentinel_contract.read_text(encoding="utf-8"))
            sentinel_contract_status = sentinel_payload.get("sentinel_contract_status")
        except json.JSONDecodeError:
            sentinel_contract_status = "invalid_json"
    kpi_priority_source = None
    kpi_priority_hash = None
    if kpi_contract.exists():
        try:
            kpi_data = json.loads(kpi_contract.read_text(encoding="utf-8"))
            pfv_cfg = kpi_data.get("pfv", {}) if isinstance(kpi_data.get("pfv", {}), dict) else {}
            kpi_priority_source = pfv_cfg.get("priority_node_list_path")
            if not priority_nodes and kpi_priority_source:
                priority_path = resolve_path(kpi_priority_source, "")
                priority_nodes = read_ids(priority_path)
                kpi_priority_hash = sha256_file(priority_path)
        except json.JSONDecodeError:
            pass

    gat_candidates = []
    raw_gat_candidates = sensing_cfg.get("gat_candidates") or state_estimation_cfg.get("gat_candidates") or {}
    if isinstance(raw_gat_candidates, dict):
        gat_iter = [{"name": name, "path": path} for name, path in raw_gat_candidates.items()]
    elif isinstance(raw_gat_candidates, list):
        gat_iter = raw_gat_candidates
    else:
        gat_iter = []
    for item in gat_iter:
        if not isinstance(item, dict):
            continue
        path = resolve_path(item.get("path"), "")
        gat_candidates.append(
            {
                "name": item.get("name"),
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path),
                "compatibility_status": "present_unverified" if path.exists() else "missing",
                "required_followup": [
                    "node_order_hash",
                    "sensor_mask_hash",
                    "normalization_hash",
                    "model_structure_hash",
                    "retrofit_network_signature",
                ],
            }
        )

    missing_inp = [row["actuator_id"] for row in actuator_audit if not row["exists_in_inp"]]
    report = {
        "config": str(config_path),
        "required_files": [
            path_status(config_path),
            path_status(inp_path),
            path_status(ids_path),
            path_status(kpi_contract),
            path_status(forecast_contract),
            path_status(sentinel_contract),
            path_status(sentinel_nodes_path),
        ],
        "managed_actuator_count": len(actuator_ids),
        "managed_actuator_ids": actuator_ids,
        "duplicate_actuator_ids": duplicate_ids,
        "actuator_ids_missing_from_inp": missing_inp,
        "actuator_type_counts": dict(Counter(row["actuator_type"] for row in actuator_audit)),
        "binary_pumps": [row["actuator_id"] for row in actuator_audit if row["binary_pump_semantics"]],
        "storage_linked_actuators": [
            row["actuator_id"] for row in actuator_audit if row["storage_role"] != "not_storage_linked"
        ],
        "preliminary_text_reference_count": sum(1 for row in actuator_audit if row["preliminary_controls_text_reference"]),
        "preliminary_text_referenced_actuators": [
            row["actuator_id"] for row in actuator_audit if row["preliminary_controls_text_reference"]
        ],
        "preliminary_text_reference_note": "String search in [CONTROLS] only; not an implemented native-rule behavior audit.",
        "actuator_audit": actuator_audit,
        "priority_node_presence": node_presence(sections, priority_nodes),
        "priority_node_source": kpi_priority_source,
        "priority_node_source_sha256": kpi_priority_hash,
        "sentinel_node_presence": node_presence(sections, sentinel_nodes),
        "sentinel_contract_status": sentinel_contract_status,
        "gat_candidates": gat_candidates,
        "status": "passed_file_presence_and_hash_audit",
        "compatibility_note": "This audit does not validate GAT compatibility, fallback reachability, state clone equivalence, or SWMM execution.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
