"""Gate PFV-first dual-fallback effect model before Smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_GATE_KEYS = [
    "tfv_unsafe_recall",
    "tfv_false_safe",
    "peak_unsafe_recall",
    "peak_false_safe",
    "pfv_pairwise_ranking_accuracy",
    "top5_realized_pfv_improvement_fraction",
    "interval_90_coverage",
    "interval_90_width_median",
    "interval_sharpness_score",
    "per_key_class_independent_events",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def evaluate_gate(config_path: Path, metrics_path: Path, out_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    metrics = load_json(metrics_path)
    thresholds = cfg.get("model_gate", {})
    missing = [key for key in REQUIRED_GATE_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"metrics missing keys: {missing}")

    checks = {
        "tfv_unsafe_recall": metrics["tfv_unsafe_recall"] >= thresholds["tfv_unsafe_recall_min"],
        "tfv_false_safe": metrics["tfv_false_safe"] <= thresholds["tfv_false_safe_max"],
        "peak_unsafe_recall": metrics["peak_unsafe_recall"] >= thresholds["peak_unsafe_recall_min"],
        "peak_false_safe": metrics["peak_false_safe"] <= thresholds["peak_false_safe_max"],
        "pfv_pairwise_ranking_accuracy": metrics["pfv_pairwise_ranking_accuracy"] >= thresholds["pfv_pairwise_ranking_accuracy_min"],
        "top5_realized_pfv_improvement_fraction": metrics["top5_realized_pfv_improvement_fraction"] >= thresholds["top5_realized_pfv_improvement_fraction_min"],
        "interval_90_coverage_lower_bound": metrics["interval_90_coverage"] >= thresholds["interval_90_coverage_min"],
        "interval_width_reported": metrics["interval_90_width_median"] is not None,
        "interval_sharpness_reported": metrics["interval_sharpness_score"] is not None,
        "per_key_class_independent_events": metrics["per_key_class_independent_events"] >= thresholds["per_key_class_independent_events_min"],
    }
    passed = all(checks.values())
    report = {"passed": passed, "checks": checks, "metrics": metrics, "thresholds": thresholds}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pfvfirst_model_gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate_gate(Path(args.config), Path(args.metrics), Path(args.out_dir)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
