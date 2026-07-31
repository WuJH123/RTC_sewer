from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.graph.graph_builder import build_node_link_graph, make_actuator_table, node_feature_matrix
from sewerrtc.io.inp_parser import parse_controls, parse_raingages, parse_timeseries, read_sections, audit_inp
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _configured_actuator_ids(cfg: dict) -> list[str]:
    network = cfg.get("network", {}) or {}
    raw_ids = network.get("control_actuator_ids", network.get("actuator_ids", []))
    ids: list[str] = []
    if isinstance(raw_ids, str):
        ids.extend(x.strip() for x in raw_ids.split(",") if x.strip())
    elif raw_ids:
        ids.extend(str(x).strip() for x in raw_ids if str(x).strip())
    file_key = "control_actuator_ids_file" if network.get("control_actuator_ids_file") else "actuator_ids_file"
    if network.get(file_key):
        path = cfg_path(cfg, f"network.{file_key}")
        for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                ids.append(text.split(",")[0].strip())
    out: list[str] = []
    for actuator_id in ids:
        if actuator_id and actuator_id not in out:
            out.append(actuator_id)
    return out


def _configured_control_enabled_ids(cfg: dict) -> list[str]:
    """Read a control mask without reducing the historical action vector."""
    network = cfg.get("network", {}) or {}
    raw_ids = network.get("control_enabled_actuator_ids", [])
    ids: list[str] = []
    if isinstance(raw_ids, str):
        ids.extend(x.strip() for x in raw_ids.split(",") if x.strip())
    elif raw_ids:
        ids.extend(str(x).strip() for x in raw_ids if str(x).strip())
    if network.get("control_enabled_actuator_ids_file"):
        path = cfg_path(cfg, "network.control_enabled_actuator_ids_file")
        for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                ids.append(text.split(",")[0].strip())
    return list(dict.fromkeys(x for x in ids if x))


def _maybe_cfg_path(cfg: dict, raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = Path(cfg["project_root"]) / path
    return str(path)


def _apply_policy_metadata(cfg: dict, actuators: pd.DataFrame) -> pd.DataFrame:
    """Attach benchmark policy metadata to the actuator table.

    The trajectory and closed-loop workers only receive ``actuator_table.csv``.
    Keeping fixed actions and public schedule paths in this table makes policy
    execution reproducible across multiprocessing jobs and avoids hidden
    config-dependent behavior inside worker processes.
    """
    if actuators.empty:
        return actuators
    out = actuators.copy()
    benchmark = cfg.get("benchmark", {}) or {}
    fixed_actions = benchmark.get("fixed_actions", {}) or {}
    ids = out["actuator_id"].astype(str)
    for policy_id, mapping in fixed_actions.items():
        if not isinstance(mapping, dict):
            continue
        col = f"{str(policy_id).strip()}_setting"
        out[col] = np.nan
        for i, aid in ids.items():
            if aid in mapping:
                out.loc[i, col] = float(mapping[aid])
    schedules = benchmark.get("schedule_policies", {}) or {}
    for policy_id, schedule in schedules.items():
        if not isinstance(schedule, dict):
            continue
        prefix = str(policy_id).strip()
        csv_path = _maybe_cfg_path(cfg, schedule.get("csv", ""))
        if csv_path:
            out[f"{prefix}_schedule_csv"] = csv_path
        out[f"{prefix}_schedule_time_column"] = str(schedule.get("time_column", "simtime (hr)"))
        out[f"{prefix}_schedule_time_unit"] = str(schedule.get("time_unit", "hr"))
    return out


def _annotate_actuator_scope(actuators: pd.DataFrame, configured_actuators: list[str]) -> pd.DataFrame:
    out = actuators.copy()
    has_rule = out.get("has_internal_rule", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    explicitly_configured = out.get("actuator_id", pd.Series("", index=out.index)).astype(str).isin(
        {str(x) for x in configured_actuators}
    )
    out["is_existing_rtc"] = has_rule | explicitly_configured
    out["is_physically_controllable"] = True
    out["control_scope"] = out["is_existing_rtc"].map(
        {True: "existing_rtc", False: "hypothetical_retrofit"}
    )
    return out


def _annotate_control_enabled(actuators: pd.DataFrame, enabled_actuators: list[str]) -> pd.DataFrame:
    """Declare a scenario control mask while preserving table order/indexing."""
    out = actuators.copy()
    enabled = {str(x) for x in enabled_actuators}
    if enabled:
        out["control_enabled"] = out["actuator_id"].astype(str).isin(enabled)
    else:
        out["control_enabled"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inp = cfg_path(cfg, "network.inp")
    out = ensure_dir(cfg_path(cfg, "outputs.audit"))
    sections = read_sections(inp)
    nodes, links, subcatchments, options = audit_inp(inp)
    nodes, links, edge_index = build_node_link_graph(nodes, links)
    configured_actuators = _configured_actuator_ids(cfg)
    control_enabled_actuators = _configured_control_enabled_ids(cfg)
    max_actuators = 0 if configured_actuators else int(cfg["experiment"].get("max_actuators", 0))
    actuators = make_actuator_table(nodes, links, max_actuators)
    missing_configured_actuators: list[str] = []
    if configured_actuators:
        present = set(actuators["actuator_id"].astype(str)) if "actuator_id" in actuators else set()
        missing_configured_actuators = [aid for aid in configured_actuators if aid not in present]
        if missing_configured_actuators:
            raise ValueError(
                "Configured control actuators are absent from audited SWMM actuator table: "
                f"{missing_configured_actuators}"
            )
        order = {aid: i for i, aid in enumerate(configured_actuators)}
        actuators = actuators[actuators["actuator_id"].astype(str).isin(order)].copy()
        actuators["_configured_order"] = actuators["actuator_id"].astype(str).map(order)
        actuators = actuators.sort_values("_configured_order").drop(columns=["_configured_order"])
        actuators["actuator_index"] = range(len(actuators))
    actuators = _annotate_actuator_scope(actuators, configured_actuators)
    actuators = _annotate_control_enabled(actuators, control_enabled_actuators)
    actuators = _apply_policy_metadata(cfg, actuators)
    node_x, node_cols = node_feature_matrix(nodes)
    controls = parse_controls(sections)
    raingages = parse_raingages(sections)
    timeseries = parse_timeseries(sections)
    nodes.to_csv(out / "node_table.csv", index=False)
    links.to_csv(out / "link_table.csv", index=False)
    actuators.to_csv(out / "actuator_table.csv", index=False)
    subcatchments.to_csv(out / "subcatchment_table.csv", index=False)
    controls.to_csv(out / "control_rule_audit.csv", index=False)
    raingages.to_csv(out / "raingage_table.csv", index=False)
    timeseries.to_csv(out / "timeseries_table.csv", index=False)
    np.save(out / "graph_edge_index.npy", edge_index)
    np.save(out / "graph_node_features.npy", node_x)
    report = {
        "inp": str(inp),
        "nodes": int(len(nodes)),
        "links": int(len(links)),
        "subcatchments": int(len(subcatchments)),
        "actuators_total": int(links["is_actuator"].sum()) if "is_actuator" in links else 0,
        "actuators_selected": int(len(actuators)),
        "configured_control_actuators": configured_actuators,
        "control_enabled_actuators": control_enabled_actuators,
        "missing_configured_control_actuators": missing_configured_actuators,
        "internal_rule_lines": int(len(controls)),
        "internal_rule_unique_links": int(controls["link_id"].nunique()) if not controls.empty else 0,
        "raingages": int(len(raingages)),
        "timeseries_rows": int(len(timeseries)),
        "options": options,
        "passed": True,
    }
    (out / "inp_audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
