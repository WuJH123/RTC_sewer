from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.generic_initial_policy import AutoRBCPolicy, SafeHeuristicPolicy, StorageEqualizationPolicy
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    native = np.ones(len(actuators), dtype=float)
    policies = {
        "auto_rbc": AutoRBCPolicy().action(actuators, priority_depth_max=0.9, current_action=native),
        "storage_equalization": StorageEqualizationPolicy().action(actuators, storage_fill_ratio=0.85, current_action=native),
        "safe_heuristic": SafeHeuristicPolicy().action(actuators, downstream_peak_risk=0.8, current_action=native),
    }
    rows = []
    for name, action in policies.items():
        rows.append(
            {
                "policy_id": name,
                "actuator_count": int(len(action)),
                "changed_actuator_count": int(np.sum(np.abs(action - native) > 1e-9)),
                "mean_setting": float(np.mean(action)),
                "min_setting": float(np.min(action)),
                "max_setting": float(np.max(action)),
            }
        )
    out_dir = ensure_dir(root / "outputs" / "generic_rtc")
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "generic_vs_passive_comparison.csv", index=False)
    summary = {
        "status": "generic_smoke_ready",
        "does_not_require_native_rules": True,
        "policies": list(policies.keys()),
        "actuator_count": int(len(actuators)),
        "comparison": str(out_dir / "generic_vs_passive_comparison.csv"),
    }
    (out_dir / "generic_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
