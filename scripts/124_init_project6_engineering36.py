from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _read_id_file(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            ids.append(text.split(",", 1)[0].strip())
    return list(dict.fromkeys(ids))


def _classify_semantics(row: pd.Series, *, binary_pumps: set[str], variable_speed_pumps: set[str]) -> tuple[str, str]:
    aid = str(row["actuator_id"])
    link_type = str(row.get("link_type", "")).lower()
    storage_type = str(row.get("storage_control_type", "")).lower()
    if aid in binary_pumps:
        return "binary_pump", "binary"
    if link_type == "pump" and aid not in variable_speed_pumps:
        return "binary_pump", "binary"
    if link_type == "pump":
        return "variable_speed_pump", "continuous"
    if storage_type == "storage_inlet":
        return "storage_inlet", "continuous"
    if storage_type == "storage_outlet":
        return "storage_outlet", "continuous"
    if link_type in {"orifice", "weir"}:
        return f"continuous_{link_type}", "continuous"
    return "unknown", "unknown"


def _event_split(events: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    frame = events.copy()
    frame["event_id"] = frame["event_id"].astype(str)
    if "rain_id" not in frame:
        frame["rain_id"] = frame["event_id"].str.split("_", n=1).str[0]
    if "pattern" not in frame:
        frame["pattern"] = frame["event_id"].str.extract(r"_D[0-9]+_(.+)$", expand=False).fillna("unknown")
    if "duration_min" not in frame:
        frame["duration_min"] = frame["event_id"].str.extract(r"_D([0-9]+)_", expand=False).astype(float)
    frame["_rank"] = frame["event_id"].map(lambda event: _stable_hash([event, seed]))
    frame = frame.sort_values("_rank").reset_index(drop=True)
    smoke_ids = {
        "T10_D150_chicago_center",
        "T30_D240_block",
        "T50_D300_chicago_late",
        "T75_D240_double_peak",
        "T100_D300_chicago_late",
    }
    roles: list[str] = []
    for i, row in enumerate(frame.itertuples(index=False)):
        event_id = str(row.event_id)
        if event_id in smoke_ids:
            roles.append("smoke")
        elif i % 5 == 0:
            roles.append("formal_blind")
        elif i % 5 == 1:
            roles.append("calibration")
        else:
            roles.append("fit")
    frame["split"] = roles
    frame["is_stress_event"] = frame["rain_id"].astype(str).isin({"T75", "T100"})
    return frame.drop(columns=["_rank"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Project6 Engineering36 experiment contract.")
    parser.add_argument("--config", default="configs/wuhan_project6_engineering36.yaml")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["project_root"])
    engineering = cfg.get("engineering36", {}) or {}
    out = ensure_dir(root / (args.out_dir or engineering.get("frozen_dir", "outputs/project6_engineering36/frozen")))

    enabled_path = cfg_path(cfg, "network.control_enabled_actuator_ids_file")
    enabled_ids = _read_id_file(enabled_path)
    residual_ids = [str(x) for x in engineering.get("residual10_actuator_ids", [])]
    residual_set = set(residual_ids)
    if len(enabled_ids) != 36:
        raise ValueError(f"expected 36 control-enabled actuators, got {len(enabled_ids)} from {enabled_path}")
    missing_residual_order = [aid for aid in residual_ids if aid not in enabled_ids]
    if missing_residual_order:
        raise ValueError(f"residual10 ids missing from 36-facility mask: {missing_residual_order}")
    core_ids = [aid for aid in enabled_ids if aid not in residual_set]
    if len(core_ids) != 26 or len(residual_ids) != 10:
        raise ValueError(f"expected core26/residual10 split, got core={len(core_ids)} residual={len(residual_ids)}")

    audit_dir = root / str(engineering.get("audit_source_dir", "outputs/audit_v8_storage_36"))
    actuator_table = pd.read_csv(audit_dir / "actuator_table.csv")
    node_table = pd.read_csv(audit_dir / "node_table.csv")
    action_order = {aid: i for i, aid in enumerate(enabled_ids)}
    facilities = actuator_table[actuator_table["actuator_id"].astype(str).isin(action_order)].copy()
    facilities["_order"] = facilities["actuator_id"].astype(str).map(action_order)
    facilities = facilities.sort_values("_order").drop(columns=["_order"])
    if len(facilities) != 36:
        present = set(facilities["actuator_id"].astype(str))
        raise ValueError(f"enabled actuator ids absent from actuator table: {sorted(set(enabled_ids) - present)}")

    binary_pumps = {str(x) for x in engineering.get("binary_pump_ids", [])}
    variable_speed_pumps = {str(x) for x in engineering.get("variable_speed_pump_ids", [])}
    semantics_rows = []
    for index, (_, series) in enumerate(facilities.iterrows()):
        aid = str(series["actuator_id"])
        semantic, value_type = _classify_semantics(series, binary_pumps=binary_pumps, variable_speed_pumps=variable_speed_pumps)
        semantics_rows.append(
            {
                "action_index": index,
                "actuator_id": aid,
                "tier": "tier2_residual10" if aid in residual_set else "tier1_core26",
                "link_type": str(series.get("link_type", "")),
                "from_node": str(series.get("from_node", "")),
                "to_node": str(series.get("to_node", "")),
                "storage_node_id": "" if pd.isna(series.get("storage_node_id", "")) else str(series.get("storage_node_id", "")),
                "storage_control_type": str(series.get("storage_control_type", "")),
                "control_semantics": semantic,
                "value_type": value_type,
                "min_setting": 0.0,
                "max_setting": 1.0,
                "max_delta_per_step": 1.0 if value_type == "binary" else float((cfg.get("controller", {}) or {}).get("action_constraints", {}).get("continuous_max_delta_per_step", 0.12)),
                "min_on_steps": int((cfg.get("controller", {}) or {}).get("action_constraints", {}).get("binary_pump_min_on_steps", 2)) if value_type == "binary" else "",
                "min_off_steps": int((cfg.get("controller", {}) or {}).get("action_constraints", {}).get("binary_pump_min_off_steps", 2)) if value_type == "binary" else "",
                "source": "audited INP actuator_table plus Project6 Engineering36 mask",
            }
        )
    semantics = pd.DataFrame(semantics_rows)
    if set(semantics.loc[semantics["value_type"].eq("binary"), "actuator_id"]) != binary_pumps:
        raise ValueError("binary pump declaration does not match frozen semantics")
    semantics.to_csv(out / "facilities_36_semantics.csv", index=False)

    priority_nodes = [str(x) for x in engineering.get("priority_core_nodes", [])]
    sentinel_nodes = [str(x) for x in engineering.get("sentinel_nodes", [])]
    node_ids = set(node_table["node_id"].astype(str))
    missing_nodes = sorted((set(priority_nodes) | set(sentinel_nodes)) - node_ids)
    if missing_nodes:
        raise ValueError(f"priority/sentinel nodes absent from node table: {missing_nodes}")
    pd.DataFrame(
        {
            "node_id": priority_nodes,
            "role": "pfv_core",
            "counts_toward_pfv": True,
            "freeze_basis": "pre-declared Project6 Engineering36 priority core; not selected from controller outcomes",
        }
    ).to_csv(out / "priority_core_nodes.csv", index=False)
    sentinel_meta = node_table[node_table["node_id"].astype(str).isin(sentinel_nodes)].copy()
    sentinel_meta = sentinel_meta.assign(
        role="depth_surcharge_sentinel",
        counts_toward_pfv=False,
        freeze_basis="pre-declared high-water/backwater-sensitive sentinel; safety constraint only",
    )
    sentinel_meta[["node_id", "role", "counts_toward_pfv", "max_depth", "degree_in", "degree_out", "freeze_basis"]].to_csv(
        out / "sentinel_nodes.csv", index=False
    )

    rainfall_path = root / str(engineering.get("rainfall_event_table"))
    events = _event_split(pd.read_csv(rainfall_path), seed=int((cfg.get("experiment", {}) or {}).get("random_seed", 20260716)))
    events.to_csv(out / "event_split.csv", index=False)

    model_rows = []
    for ratio, raw in (engineering.get("sensor_ratio_model_paths", {}) or {}).items():
        path = root / str(raw)
        model_rows.append(
            {
                "sensor_ratio_id": str(ratio),
                "model_path": str(path),
                "exists": bool(path.exists()),
                "sha256": _sha256_file(path) if path.exists() else "",
                "reuse_decision": "reuse_state_reconstruction_only" if path.exists() else "missing_block_stage",
            }
        )
    pd.DataFrame(model_rows).to_csv(out / "gat_model_registry.csv", index=False)

    artifacts = [
        out / "facilities_36_semantics.csv",
        out / "priority_core_nodes.csv",
        out / "sentinel_nodes.csv",
        out / "event_split.csv",
        out / "gat_model_registry.csv",
        enabled_path,
        rainfall_path,
    ]
    manifest = {
        "config": str(Path(args.config).resolve()),
        "frozen_dir": str(out),
        "control_enabled_ids_file": str(enabled_path),
        "action_ids_sha256": _stable_hash(enabled_ids),
        "core26_ids": core_ids,
        "residual10_ids": residual_ids,
        "binary_pump_ids": sorted(binary_pumps),
        "priority_core_nodes": priority_nodes,
        "sentinel_nodes": sentinel_nodes,
        "event_split_counts": events["split"].value_counts().astype(int).to_dict(),
        "stress_event_counts": [
            {"split": str(split), "is_stress_event": bool(is_stress), "events": int(count)}
            for (split, is_stress), count in events.groupby(["split", "is_stress_event"]).size().items()
        ],
        "gat_models_all_present": bool(pd.DataFrame(model_rows)["exists"].all()) if model_rows else False,
        "artifact_hashes": {path.name: _sha256_file(path) for path in artifacts if path.exists()},
        "trajectory_reuse_policy": engineering.get("trajectory_reuse_policy", {}),
        "passed": bool(model_rows and pd.DataFrame(model_rows)["exists"].all()),
    }
    (out / "contract_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
