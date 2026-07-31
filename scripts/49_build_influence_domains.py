from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.network.influence_domain import build_priority_influence_domains


def _output_dir(cfg: dict) -> Path:
    raw = (cfg.get("outputs", {}) or {}).get("network", "outputs/network")
    path = Path(raw)
    if not path.is_absolute():
        path = cfg_path(cfg, "project_root") / path
    return ensure_dir(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--khop", type=int, default=3)
    ap.add_argument("--fallback-khop", type=int, default=12)
    ap.add_argument("--max-candidates-per-priority", type=int, default=24)
    ap.add_argument("--max-storage-controls-per-priority", type=int, default=10)
    ap.add_argument("--max-regulators-per-priority", type=int, default=48)
    ap.add_argument("--max-pumps-per-priority", type=int, default=32)
    ap.add_argument("--no-global-storage-controls", action="store_true")
    ap.add_argument("--no-global-regulators", action="store_true")
    ap.add_argument("--no-global-pumps", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    link_path = cfg_path(cfg, "outputs.audit") / "link_table.csv"
    act_path = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    priority_path = cfg_path(cfg, "outputs.design") / "priority_nodes.txt"
    link_table = pd.read_csv(link_path)
    actuator_table = pd.read_csv(act_path)
    actuator_scope = str((cfg.get("controller", {}) or {}).get("actuator_scope", "existing_rtc"))
    actuator_table = select_actuators_for_scope(actuator_table, actuator_scope)
    if actuator_table.empty:
        raise ValueError(f"No actuators available for controller.actuator_scope={actuator_scope}")
    priority_nodes = [x.strip() for x in priority_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    domains, candidates = build_priority_influence_domains(
        link_table,
        actuator_table,
        priority_nodes,
        k=int(args.khop),
        fallback_k=int(args.fallback_khop),
        max_candidates_per_priority=int(args.max_candidates_per_priority),
        include_global_storage_controls=not bool(args.no_global_storage_controls),
        include_global_regulators=not bool(args.no_global_regulators),
        include_global_pumps=not bool(args.no_global_pumps),
        max_storage_controls_per_priority=int(args.max_storage_controls_per_priority),
        max_regulators_per_priority=int(args.max_regulators_per_priority),
        max_pumps_per_priority=int(args.max_pumps_per_priority),
    )
    out_dir = _output_dir(cfg)
    domains.to_csv(out_dir / "influence_domains.csv", index=False)
    candidates.to_csv(out_dir / "priority_to_actuator_candidates.csv", index=False)
    summary = {
        "priority_nodes": len(priority_nodes),
        "domain_rows": int(len(domains)),
        "candidate_rows": int(len(candidates)),
        "unique_actuators": int(candidates["actuator_id"].nunique()) if not candidates.empty else 0,
        "primary_khop": int(args.khop),
        "fallback_khop": int(args.fallback_khop),
        "max_candidates_per_priority": int(args.max_candidates_per_priority),
        "max_storage_controls_per_priority": int(args.max_storage_controls_per_priority),
        "max_regulators_per_priority": int(args.max_regulators_per_priority),
        "max_pumps_per_priority": int(args.max_pumps_per_priority),
        "candidate_rows_by_role": candidates["asset_role"].value_counts().to_dict() if not candidates.empty else {},
        "unique_actuators_by_role": candidates.groupby("asset_role")["actuator_id"].nunique().to_dict() if not candidates.empty else {},
        "outputs": {
            "domains": str(out_dir / "influence_domains.csv"),
            "candidates": str(out_dir / "priority_to_actuator_candidates.csv"),
        },
    }
    (out_dir / "influence_domain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
