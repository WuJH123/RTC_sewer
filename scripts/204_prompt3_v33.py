from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.prompt3 import action_effect_v33 as v33


def main() -> int:
    parser = argparse.ArgumentParser(description="Project6 V3.3 champion-V31 repair, Round5, and formal flow.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--target-samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract-dry-run", action="store_true")
    args = parser.parse_args()

    dispatch = {
        "DiagnoseV32RegressionV33": lambda: v33.diagnose_v32_regression_v33(args.config),
        "RunModuleAblationV33": lambda: v33.run_module_ablation_v33(args.config, args.max_events or 12, args.workers, args.resume),
        "PlanRound5HardNegativesV33": lambda: v33.plan_round5_hard_negatives_v33(args.config, args.target_samples, args.seed),
        "GenerateRound5HardNegativesV33": lambda: v33.generate_round5_hard_negatives_v33(args.config, args.max_samples, args.smoke, args.resume),
        "BuildRound5DatasetV33": lambda: v33.build_round5_dataset_v33(args.config, args.smoke),
        "AuditRound5DatasetV33": lambda: v33.audit_round5_dataset_v33(args.config, args.smoke),
        "TrainActionEffectV33": lambda: v33.train_action_effect_v33(args.config, args.epochs, args.ensemble_size, args.max_samples, args.smoke),
        "CalibrateUncertaintyV33": lambda: v33.calibrate_uncertainty_v33(args.config, args.smoke),
        "TrainOODSafetyFallbackV33": lambda: v33.train_ood_safety_fallback_v33(args.config, args.smoke),
        "EvaluateModelGateV33": lambda: v33.evaluate_model_gate_v33(args.config, args.smoke),
        "RunClosedLoopDevV33": lambda: v33.run_closed_loop_dev_v33(args.config, args.max_events or 3, args.workers, args.resume),
        "BuildEvaluationRainfallAssetsV33": lambda: v33.build_evaluation_rainfall_assets_v33(args.config),
        "BuildEvaluationSplitsV33": lambda: v33.build_evaluation_splits_v33(args.config),
        "AuditEvaluationSplitsV33": lambda: v33.audit_evaluation_splits_v33(args.config),
        "CalibrationAV33": lambda: v33.calibration_a_v33(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "LockedValidationBV33": lambda: v33.locked_validation_b_v33(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "PolicyLockV33": lambda: v33.policy_lock_v33(args.config),
        "AuditPolicyLockV33": lambda: v33.audit_policy_lock_v33(args.config),
        "FormalBlindV33": lambda: v33.formal_blind_v33(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "RunFormalExtraBaselinesV33": lambda: v33.run_formal_extra_baselines_v33(args.config, args.max_events, args.workers, args.resume),
        "BuildFormalComparisonV33": lambda: v33.build_formal_comparison_v33(args.config),
        "EvaluateFormalPerformanceV33": lambda: v33.evaluate_formal_performance_v33(args.config),
        "ExportFormalTablesV33": lambda: v33.export_formal_tables_v33(args.config),
    }
    if args.stage not in dispatch:
        print(json.dumps({"status": "failed", "error": f"unknown_stage:{args.stage}"}, indent=2))
        return 7
    code, outputs = dispatch[args.stage]()
    status = "pass" if code == 0 else "blocked" if code == 3 else "failed_gate" if code == 5 else "contract_mismatch" if code == 6 else "failed"
    print(json.dumps({"status": status, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
