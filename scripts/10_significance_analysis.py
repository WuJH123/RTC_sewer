from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.evaluation.significance import paired_stats
from sewerrtc.io.project_paths import cfg_path, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="formal")
    ap.add_argument("--run-tag", default="", help="Optional diagnostics subdirectory.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    diag = cfg_path(cfg, "outputs.diagnostics") / args.mode
    if args.run_tag:
        diag = diag / args.run_tag
    comp = pd.read_csv(diag / "strategy_comparison.csv")
    if "baseline_policy" in comp:
        frames = []
        for policy, sub in comp.groupby("baseline_policy"):
            st = paired_stats(sub)
            st.insert(0, "baseline_policy", policy)
            frames.append(st)
        stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        stats = paired_stats(comp)
    out = diag / "paired_bootstrap_wilcoxon.csv"
    stats.to_csv(out, index=False)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
