from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.data.historical_trajectory_planning import (
    build_action_learning_plan,
    build_gat_mixing_plan,
    build_sensor_coverage_plan,
    canonical_action_ids_from_order,
    node_signature_from_cache,
    scan_trajectory_roots,
)
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _split_paths(value: str) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def _default_trajectory_roots(project6_root: Path) -> list[Path]:
    project5_root = project6_root.parent / "Project5"
    candidates = [
        project6_root / "outputs" / "data_bank_train_paired_no_controls" / "trajectories",
        project6_root / "outputs" / "data_bank_train_v8_storage_variablepump" / "trajectories",
        project6_root / "outputs" / "data_bank_train_v8_storage_36" / "trajectories",
        project6_root / "outputs" / "data_bank_train_v8_storage_retrained" / "trajectories",
        project6_root / "outputs" / "data_bank_train_v8_storage_full_retrain" / "trajectories",
        project5_root / "outputs" / "data_bank_train_paired_no_controls" / "trajectories",
        project5_root / "outputs" / "pystorms_beta" / "data_bank_train" / "trajectories",
        project5_root / "outputs" / "pystorms_beta" / "trajectories",
    ]
    return [path for path in candidates if path.exists()]


def _relative_or_absolute(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Project5/Project6 historical trajectories and build no-training manifests for GAT and 36-action learning."
    )
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--trajectory-roots", default="", help="Comma/semicolon separated trajectory directories. Defaults to known Project5/6 banks.")
    parser.add_argument("--canonical-action-order", default="outputs/project6_36_fulltrain_v1/canonical_action_order/canonical_36_actuator_order.csv")
    parser.add_argument("--base-cache", default="outputs/cache_v8_storage_variablepump/transition_cache.npz")
    parser.add_argument("--sensor-ratios", default="0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--out-dir", default="outputs/research_reuse_plan")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out_dir = ensure_dir(_relative_or_absolute(root, args.out_dir))
    canonical_path = _relative_or_absolute(root, args.canonical_action_order)
    base_cache_path = _relative_or_absolute(root, args.base_cache)
    action_ids = canonical_action_ids_from_order(canonical_path)
    roots = _split_paths(args.trajectory_roots) or _default_trajectory_roots(root)
    roots = [path if path.is_absolute() else root / path for path in roots]

    inventory = scan_trajectory_roots(roots, canonical_action_ids=action_ids)
    inventory_path = out_dir / "historical_trajectory_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    base_signature = node_signature_from_cache(base_cache_path)
    gat_plan = build_gat_mixing_plan(inventory, base_node_signature=base_signature)
    gat_plan_path = out_dir / "gat_mixed_trajectory_manifest.csv"
    gat_plan.to_csv(gat_plan_path, index=False)

    action_plan = build_action_learning_plan(inventory, canonical_action_ids=action_ids, horizon_steps=int(args.horizon_steps))
    action_plan_path = out_dir / "temporal_action_learning_manifest.csv"
    action_plan.to_csv(action_plan_path, index=False)

    ratios = [float(item.strip()) for item in args.sensor_ratios.replace(";", ",").split(",") if item.strip()]
    sensor_plan = build_sensor_coverage_plan(ratios, include_priority_nodes=True)
    sensor_plan_path = out_dir / "sensor_coverage_plan.csv"
    sensor_plan.to_csv(sensor_plan_path, index=False)

    gat_used = gat_plan[gat_plan["gat_use"].fillna(False)]
    action_used = action_plan[action_plan["action_learning_use"].fillna(False)]
    summary = {
        "config": str(Path(args.config).resolve()),
        "canonical_action_order": str(canonical_path),
        "base_cache": str(base_cache_path),
        "base_node_signature": base_signature,
        "trajectory_roots_requested": [str(path) for path in roots],
        "inventory_rows": int(len(inventory)),
        "inventory_files_existing": int(inventory.get("exists", pd.Series(dtype=bool)).fillna(False).sum()) if len(inventory) else 0,
        "node_signature_count": int(inventory["node_signature"].nunique()) if "node_signature" in inventory else 0,
        "gat_manifest_rows": int(len(gat_used)),
        "gat_events": int(gat_used["event_id"].nunique()) if len(gat_used) else 0,
        "gat_policies": sorted(gat_used["policy_id"].dropna().astype(str).unique().tolist()) if len(gat_used) else [],
        "action_learning_rows": int(len(action_used)),
        "action_learning_events": int(action_used["event_id"].nunique()) if len(action_used) else 0,
        "action_learning_policies": sorted(action_used["policy_id"].dropna().astype(str).unique().tolist()) if len(action_used) else [],
        "same_state_effect_rows": int(action_used["effect_label_role"].astype(str).eq("same_state_candidate_vs_no_control_effect").sum()) if len(action_used) else 0,
        "observational_pretraining_rows": int(action_used["effect_label_role"].astype(str).str.contains("pretraining", na=False).sum()) if len(action_used) else 0,
        "sensor_ratios": ratios,
        "outputs": {
            "inventory": str(inventory_path),
            "gat_manifest": str(gat_plan_path),
            "action_learning_manifest": str(action_plan_path),
            "sensor_coverage_plan": str(sensor_plan_path),
        },
        "interpretation": {
            "gat": "Use all rows in gat_manifest for state reconstruction only when node_signature matches the base Wuhan network.",
            "action_learning": "Use observational trajectories for action/state dynamics pretraining; use same-state paired rows only for safety effect labels.",
            "mpc": "Closed-loop gate should compare candidate sequence against online predicted no-control reference with PFV non-inferiority, peak safety, and TFV reduction ranking.",
        },
    }
    summary_path = out_dir / "research_reuse_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
