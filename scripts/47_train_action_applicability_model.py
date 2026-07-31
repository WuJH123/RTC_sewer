from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out_dir = ensure_dir(root / "outputs" / "models_paired_no_controls")
    model_card = {
        "model_type": "heuristic_action_applicability",
        "status": "baseline_ready",
        "features": [
            "priority_risk_score",
            "priority_depth_trend",
            "rain_phase",
            "upstream_storage_available",
            "downstream_capacity_margin",
            "uncertainty_score",
        ],
        "outputs": [
            "applicability_prob",
            "expected_benefit",
            "risk_penalty",
            "uncertainty_penalty",
            "recommended_phase",
            "disallow_reason",
        ],
    }
    path = out_dir / "action_applicability_model_card.json"
    path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(path), **model_card}, indent=2))


if __name__ == "__main__":
    main()
