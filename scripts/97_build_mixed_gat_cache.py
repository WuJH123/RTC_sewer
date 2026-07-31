from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sewerrtc.data.three_step_research_builders import build_mixed_gat_cache
from sewerrtc.io.project_paths import cfg_path, load_config


def _node_cols_from_cache(path: Path) -> list[str]:
    with np.load(path, allow_pickle=True) as data:
        return [str(x) for x in data["node_cols"].tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GAT state-reconstruction cache from same-network historical trajectories.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--manifest", default="outputs/research_reuse_plan/gat_mixed_trajectory_manifest.csv")
    parser.add_argument("--base-cache", default="outputs/cache_v8_storage_variablepump/transition_cache.npz")
    parser.add_argument("--out-npz", default="outputs/cache_research_mixed_gat/transition_cache.npz")
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    manifest = root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    base_cache = root / args.base_cache if not Path(args.base_cache).is_absolute() else Path(args.base_cache)
    out_npz = root / args.out_npz if not Path(args.out_npz).is_absolute() else Path(args.out_npz)
    report = build_mixed_gat_cache(
        manifest_path=manifest,
        out_npz=out_npz,
        base_node_cols=_node_cols_from_cache(base_cache),
        time_stride=int(args.time_stride),
        max_files=int(args.max_files),
        max_samples=int(args.max_samples),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
