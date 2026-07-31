from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    args = ap.parse_args()
    cfg = load_config(args.config)
    comp_path = cfg_path(cfg, "outputs.diagnostics") / args.mode / "strategy_comparison.csv"
    if not comp_path.exists():
        print(f"missing {comp_path}")
        return
    comp = pd.read_csv(comp_path)
    out = ensure_dir(cfg_path(cfg, "outputs.figures"))
    fig, ax = plt.subplots(figsize=(7, 4))
    comp[["PFV_reduction_pct", "TFV_reduction_pct", "peak_TFV_rate_reduction_pct"]].boxplot(ax=ax)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylabel("Reduction vs auto-RBC (%)")
    ax.set_title("Project4 PFV-first safety-constrained MPC")
    fig.tight_layout()
    fig.savefig(out / f"fig_reduction_boxplot_{args.mode}.svg")
    print(f"saved {out / f'fig_reduction_boxplot_{args.mode}.svg'}")


if __name__ == "__main__":
    main()
