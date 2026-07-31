from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.data.historical_trajectory_planning import canonical_action_ids_from_order
from sewerrtc.data.three_step_research_builders import build_temporal_action_pretrain_dataset
from sewerrtc.io.project_paths import cfg_path, load_config


def _node_cols_from_cache(path: Path) -> list[str]:
    with np.load(path, allow_pickle=True) as data:
        return [str(x) for x in data["node_cols"].tolist()]


def _read_nodes_txt(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines() if line.strip()]


def _local_node_cols(root: Path, cfg: dict, base_node_cols: list[str], canonical_path: Path) -> list[str]:
    node_ids = {c.split(":", 1)[1] for c in base_node_cols if c.startswith("h:")}
    selected: list[str] = []
    design = cfg_path(cfg, "outputs.design")
    for filename in ["priority_nodes.txt", "priority_sentinel_nodes.txt", "priority_domain_nodes.txt"]:
        for node in _read_nodes_txt(design / filename):
            if node in node_ids and node not in selected:
                selected.append(node)
    if canonical_path.exists():
        table = pd.read_csv(canonical_path)
        for col in ["upstream_node", "downstream_node", "storage_association"]:
            if col in table:
                for node in table[col].dropna().astype(str):
                    if node and node in node_ids and node not in selected:
                        selected.append(node)
    # Keep the local target bounded; priority/storage/actuator-neighbour
    # coverage is enough for action pretraining and avoids [N,H,932].
    return [f"h:{node}" for node in selected[:128]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build observational [H,36] temporal action pretraining data from historical trajectories.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--manifest", default="outputs/research_reuse_plan/temporal_action_learning_manifest.csv")
    parser.add_argument("--base-cache", default="outputs/cache_v8_storage_variablepump/transition_cache.npz")
    parser.add_argument("--canonical-action-order", default="outputs/project6_36_fulltrain_v1/canonical_action_order/canonical_36_actuator_order.csv")
    parser.add_argument("--out-npz", default="outputs/cache_temporal_action_pretrain_36/temporal_action_pretrain_36.npz")
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--target-mode", choices=["risk_local", "full_state"], default="risk_local")
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--chunk-size-samples",
        type=int,
        default=0,
        help="Write bounded NPZ shards of this many samples; 0 preserves the legacy single-file format.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    manifest = root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    base_cache = root / args.base_cache if not Path(args.base_cache).is_absolute() else Path(args.base_cache)
    canonical_path = root / args.canonical_action_order if not Path(args.canonical_action_order).is_absolute() else Path(args.canonical_action_order)
    out_npz = root / args.out_npz if not Path(args.out_npz).is_absolute() else Path(args.out_npz)
    node_cols = _node_cols_from_cache(base_cache)
    priority_nodes = _read_nodes_txt(cfg_path(cfg, "outputs.design") / "priority_nodes.txt")
    report = build_temporal_action_pretrain_dataset(
        manifest_path=manifest,
        out_npz=out_npz,
        base_node_cols=node_cols,
        canonical_action_ids=canonical_action_ids_from_order(canonical_path),
        priority_nodes=priority_nodes,
        local_node_cols=_local_node_cols(root, cfg, node_cols, canonical_path),
        horizon_steps=int(args.horizon_steps),
        target_mode=str(args.target_mode),
        time_stride=int(args.time_stride),
        max_files=int(args.max_files),
        max_samples=int(args.max_samples),
        chunk_size_samples=int(args.chunk_size_samples),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
