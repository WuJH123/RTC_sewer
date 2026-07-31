from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.action_applicability import heuristic_action_applicability


def _read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--dataset", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    hcfg = cfg.get("horizon_surrogate", {}) or {}
    dataset_path = Path(args.dataset) if args.dataset else root / hcfg.get("output_dataset", "data/surrogate/horizon_mpc_dataset.parquet")
    if not dataset_path.exists():
        dataset_path = root / hcfg.get("fallback_output_dataset", "data/surrogate/horizon_mpc_dataset.csv")
    df = _read_dataset(dataset_path).head(5000)
    rows = []
    for _, r in df.iterrows():
        res = heuristic_action_applicability(
            {
                "phase": r.get("phase", "unknown"),
                "priority_depth_max": r.get("priority_depth_max", 0.0),
                "priority_depth_trend": r.get("priority_depth_trend", 0.0),
                "upstream_storage_available": max(0.0, 1.0 - float(r.get("action_mean", 1.0) or 1.0)),
                "downstream_capacity_margin": 1.0 - float(r.get("current_depth_p95", 0.0) or 0.0),
                "uncertainty_score": 0.0,
            }
        )
        rows.append({**res.__dict__, "action_value": res.action_value})
    out = pd.DataFrame(rows)
    out_dir = ensure_dir(root / "outputs" / "surrogate")
    out.to_csv(out_dir / "action_applicability_validation.csv", index=False)
    summary = {
        "samples": int(len(out)),
        "mean_applicability_prob": float(out["applicability_prob"].mean()) if not out.empty else 0.0,
        "pass_rate_prob_ge_0p5": float((out["applicability_prob"] >= 0.5).mean()) if not out.empty else 0.0,
        "output": str(out_dir / "action_applicability_validation.csv"),
    }
    (out_dir / "action_applicability_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
