from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.influence_candidate_generator import generate_influence_candidates
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--max-delta", type=float, default=0.08)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    act = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    cand_path = root / "outputs" / "network" / "priority_to_actuator_candidates.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"Missing influence candidate table: {cand_path}. Run scripts/49_build_influence_domains.py")
    cand = pd.read_csv(cand_path)
    native = np.ones(len(act), dtype=float)
    generated = generate_influence_candidates(native, cand, act, max_delta=float(args.max_delta))
    rows = [{k: v for k, v in g.items() if k != "action"} for g in generated]
    out = pd.DataFrame(rows)
    out_dir = ensure_dir(root / "outputs" / "network")
    out.to_csv(out_dir / "influence_candidate_generation_audit.csv", index=False)
    summary = {
        "input_candidate_rows": int(len(cand)),
        "generated_action_candidates": int(len(out)),
        "max_delta": float(args.max_delta),
        "unique_priority_nodes": int(out["target_priority_nodes"].nunique()) if not out.empty else 0,
        "unique_templates": int(out["label"].str.split("|").str[0].nunique()) if not out.empty else 0,
        "output": str(out_dir / "influence_candidate_generation_audit.csv"),
    }
    (out_dir / "influence_candidate_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
