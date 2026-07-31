from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, load_config


def classify(row: pd.Series, variable_speed_pump_ids: set[str]) -> str:
    link_type = str(row.get("link_type", "")).lower()
    storage = str(row.get("storage_control_type", "")).lower()
    if storage == "storage_inlet":
        return "storage_inlet"
    if storage == "storage_outlet":
        return "storage_outlet"
    if link_type == "orifice":
        return "continuous_orifice"
    if link_type == "weir":
        return "continuous_weir"
    if link_type == "pump" and str(row.get("actuator_id", "")) in variable_speed_pump_ids:
        return "variable_speed_pump"
    if link_type == "pump":
        return "binary_pump"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--out-dir", default="outputs/audits")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    controller_cfg = cfg.get("controller", {}) or {}
    variable_speed_pump_ids = {
        str(value) for value in controller_cfg.get("variable_speed_pump_ids", [])
    }
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    influence_path = cfg_path(cfg, "outputs.network") / "priority_to_actuator_candidates.csv"
    influence = pd.read_csv(influence_path) if influence_path.exists() else pd.DataFrame()
    domains = {}
    if not influence.empty:
        domains = influence.groupby("actuator_id")["priority_node"].agg(lambda values: json.dumps(sorted(set(map(str, values))))).to_dict()
    out = pd.DataFrame()
    out["actuator_id"] = actuators["actuator_id"].astype(str)
    out["link_type"] = actuators["link_type"].astype(str)
    out["real_or_virtual"] = np.where(actuators.get("control_scope", "").astype(str).eq("hypothetical_retrofit"), "virtual_or_planning_asset", "existing_asset")
    out["control_semantics"] = actuators.apply(
        classify, axis=1, variable_speed_pump_ids=variable_speed_pump_ids
    )
    out["continuous_or_binary"] = out["control_semantics"].map({
        "continuous_orifice": "continuous", "continuous_weir": "continuous",
        "storage_inlet": "continuous", "storage_outlet": "continuous",
        "binary_pump": "binary", "variable_speed_pump": "continuous",
    }).fillna("unknown")
    out["min_setting"] = 0.0
    out["max_setting"] = 1.0
    out["rate_limit"] = np.nan
    out["minimum_hold_steps"] = np.nan
    out["minimum_on_steps"] = np.nan
    out["minimum_off_steps"] = np.nan
    out["upstream_node"] = actuators["from_node"].astype(str)
    out["downstream_node"] = actuators["to_node"].astype(str)
    out["storage_association"] = actuators.get("storage_node_id", pd.Series("", index=actuators.index)).fillna("").astype(str)
    out["priority_influence_domains"] = out["actuator_id"].map(domains).fillna("[]")
    out["semantic_evidence"] = "SWMM link type and storage endpoint association from actuator_table.csv."
    out.loc[out["control_semantics"].eq("binary_pump"), "semantic_evidence"] = (
        "No VFD declaration in this scenario configuration; conservatively classified binary."
    )
    out.loc[out["control_semantics"].eq("variable_speed_pump"), "semantic_evidence"] = (
        "Declared as a variable-speed pump in controller.variable_speed_pump_ids; "
        "continuous normalized speed is limited by the configured ramp and dwell constraints."
    )
    out["risk_flag"] = ""
    out.loc[out["real_or_virtual"].eq("virtual_or_planning_asset"), "risk_flag"] = "planning asset; not historical installed RTC"
    out.loc[out["control_semantics"].eq("unknown"), "risk_flag"] = "unknown control semantics"
    out.loc[out["rate_limit"].isna(), "risk_flag"] = out.loc[out["rate_limit"].isna(), "risk_flag"].str.cat(
        pd.Series("physical rate/hold constraints unavailable", index=out.index), sep="; ", na_rep=""
    ).str.strip("; ")
    target = root / args.out_dir / "actuator_control_semantics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(json.dumps({
        "path": str(target), "rows": len(out),
        "semantics": out["control_semantics"].value_counts().to_dict(),
        "real_or_virtual": out["real_or_virtual"].value_counts().to_dict(),
        "unknown": out.loc[out["control_semantics"].eq("unknown"), "actuator_id"].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
