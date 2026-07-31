from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--python", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    py = args.python or str(cfg.get("python", "python"))
    diag = ensure_dir(cfg_path(cfg, "outputs.diagnostics") / "supplemental_experiments")

    sensor_rows = []
    for ratio in [0.05, 0.10, 0.20]:
        tag = f"sensor_sr{int(ratio * 100):02d}"
        sensor_rows.append(
            {
                "experiment_type": "sensor_sensitivity",
                "sensor_ratio": ratio,
                "run_tag": tag,
                "purpose": "Test sparse-sensing robustness of priority-zone reconstruction and closed-loop control.",
                "commands": "\n".join(
                    [
                        f"# Create a temporary copy of configs/wuhan.yaml with experiment.sensor_ratio={ratio}.",
                        f"{py} scripts/02_select_priority_and_sensors.py --config configs/wuhan.yaml",
                        f"{py} scripts/05_train_gat.py --config configs/wuhan.yaml --device cuda --epochs 60",
                        f"powershell -ExecutionPolicy Bypass -File .\\RUN_PROJECT4_WR_EXTENDED_BENCHMARK.ps1 -Python {py} -Mode debug -Device cuda -Workers 16 -ProposedBase native -RunTag {tag} -SkipExisting",
                    ]
                ),
                "acceptance": "priority_NSE stable; PFV_worse_frac <= 0.34; TFV/peak guards not worse than main run.",
            }
        )

    ablations = [
        ("native_shield_full", "Full Proposed-NativeShield; residual value + safety guard + native rules."),
        ("no_residual_value", "Use native rules only or force residual_value_path missing; quantifies action-value contribution."),
        ("generic_clean", "No internal rules; clean INP with graph surrogate and Auto-RBC/heuristic start."),
        ("no_pfv_first_guard", "Relax PFV-first safety guard for diagnostic only; should show risk-transfer failure."),
        ("no_cooldown", "Reduce min_control_interval_steps to zero for diagnostic only; quantifies action smoothness role."),
    ]
    ablation_rows = []
    for tag, purpose in ablations:
        base = "clean" if tag == "generic_clean" else "native"
        ablation_rows.append(
            {
                "experiment_type": "ablation",
                "run_tag": tag,
                "proposed_base": base,
                "purpose": purpose,
                "command": (
                    f"powershell -ExecutionPolicy Bypass -File .\\RUN_PROJECT4_WR_EXTENDED_BENCHMARK.ps1 "
                    f"-Python {py} -Mode debug -Device cuda -Workers 16 -ProposedBase {base} -RunTag {tag} -SkipExisting"
                ),
                "acceptance": "Full model should dominate ablations on PFV with TFV/peak guards and smoother actions.",
            }
        )

    pd.DataFrame(sensor_rows).to_csv(diag / "sensor_sensitivity_plan.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(diag / "ablation_plan.csv", index=False)
    report = {
        "out_dir": str(diag),
        "sensor_sensitivity_cases": len(sensor_rows),
        "ablation_cases": len(ablation_rows),
        "note": (
            "This file is a reproducible supplemental experiment matrix. "
            "Run cases only after the native-shield preflight gate passes."
        ),
    }
    (diag / "supplemental_experiment_plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
