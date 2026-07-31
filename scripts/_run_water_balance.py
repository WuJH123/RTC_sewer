"""Run water balance baseline training and evaluation directly."""
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from sewerrtc.v4.v42_water_balance import (
    train_water_balance_baseline,
    evaluate_water_balance_baseline,
)
from sewerrtc.v4.runtime import atomic_write_json, working_code_sha
import numpy as np


def sanitize(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def main():
    project_root = Path("E:/RTC_sewer/Project6")
    output_root = Path("E:/RTC_sewer/Project6/outputs/project6_dual_reference_v4/final_v4")
    config_path = project_root / "configs" / "wuhan_project6_v4_final.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    code_sha = working_code_sha(project_root)

    # ---- Train + CV ----
    print("=" * 60)
    print("Training Water Balance Baseline (5-fold event-grouped CV)")
    print("=" * 60)
    cv_results = train_water_balance_baseline(
        project_root=project_root, output_root=output_root, config=config
    )

    model_dir = output_root / "models" / "v42_water_balance"
    model_dir.mkdir(parents=True, exist_ok=True)

    cv_sanitized = sanitize(cv_results)
    cv_sanitized["code_sha256"] = code_sha
    cv_path = model_dir / "water_balance_baseline_cv.json"
    atomic_write_json(cv_path, cv_sanitized)
    print(f"\nCV results written to: {cv_path}")

    agg = cv_results.get("aggregate_metrics", {})
    print("\n=== CV Results (averaged over 5 folds) ===")
    for target, metrics in agg.items():
        print(
            f"  {target:25s}: R2 = {metrics['r2_mean']:+.4f} +/- {metrics['r2_std']:.4f}, "
            f"MAE = {metrics['mae_mean']:.4f}, "
            f"Sign Acc = {metrics['sign_accuracy_mean']:.3f}"
        )
    print(f"\n  n_folds={cv_results['n_folds']}, n_events={cv_results['n_events']}, n_samples={cv_results['n_samples']}")

    # ---- Evaluate ----
    print("\n" + "=" * 60)
    print("Evaluating Water Balance Baseline (per-event)")
    print("=" * 60)
    eval_results = evaluate_water_balance_baseline(
        project_root=project_root, output_root=output_root, config=config
    )

    eval_sanitized = sanitize(eval_results)
    eval_sanitized["code_sha256"] = code_sha
    eval_path = model_dir / "water_balance_evaluation.json"
    atomic_write_json(eval_path, eval_sanitized)
    print(f"\nEvaluation results written to: {eval_path}")

    overall = eval_results.get("overall_metrics", {})
    print("\n=== Overall Evaluation (train=all, eval=all) ===")
    for target, metrics in overall.items():
        print(
            f"  {target:25s}: R2 = {metrics['r2']:+.4f}, "
            f"MAE = {metrics['mae']:.4f}, "
            f"Sign Acc = {metrics['sign_accuracy']:.3f}"
        )

    # ---- Model coefficients ----
    print("\n=== Model Coefficients ===")
    coeffs = eval_results.get("model_coefficients", {})
    features = eval_results.get("feature_names", [])
    print(f"  Features: {features}")
    for target, c in coeffs.items():
        print(f"  {target}: intercept={c['intercept']:.4f}, coef={[f'{x:.4f}' for x in c['coefficients']]}")

    print("\nDone! Exit code 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
