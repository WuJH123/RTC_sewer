from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.prompt3 import action_effect_v31 as v31


def main() -> int:
    parser = argparse.ArgumentParser(description="Project6 V3.1 hard-negative repair, stricter gates, and independent formal flow.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--target-samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract-dry-run", action="store_true")
    args = parser.parse_args()

    dispatch = {
        "DiagnoseFormalFailuresV31": lambda: v31.diagnose_formal_failures_v31(args.config, args.max_events),
        "PlanRound3HardNegativesV31": lambda: v31.plan_round3_hard_negatives_v31(args.config, args.target_samples, args.seed),
        "GenerateRound3HardNegativesV31": lambda: v31.generate_round3_hard_negatives_v31(args.config, args.max_samples, args.smoke, args.resume),
        "BuildRound3DatasetV31": lambda: v31.build_round3_dataset_v31(args.config, args.smoke),
        "AuditRound3DatasetV31": lambda: v31.audit_round3_dataset_v31(args.config, args.smoke),
        "TrainActionEffectV31": lambda: v31.train_action_effect_v31(args.config, args.epochs, args.ensemble_size, args.max_samples, args.smoke),
        "CalibrateUncertaintyV31": lambda: v31.calibrate_uncertainty_v31(args.config, args.smoke),
        "TrainOODSafetyFallbackV31": lambda: v31.train_ood_safety_fallback_v31(args.config, args.smoke),
        "EvaluateModelGateV31": lambda: v31.evaluate_model_gate_v31(args.config, args.smoke),
        "RunClosedLoopDevV31": lambda: v31.run_closed_loop_dev_v31(args.config, args.max_events, args.workers, args.resume),
        "BuildEvaluationRainfallAssetsV31": lambda: v31.build_evaluation_rainfall_assets_v31(args.config),
        "BuildEvaluationSplitsV31": lambda: v31.build_evaluation_splits_v31(args.config),
        "AuditEvaluationSplitsV31": lambda: v31.audit_evaluation_splits_v31(args.config),
        "CalibrationAV31": lambda: v31.calibration_a_v31(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "LockedValidationBV31": lambda: v31.locked_validation_b_v31(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "PolicyLockV31": lambda: v31.policy_lock_v31(args.config),
        "AuditPolicyLockV31": lambda: v31.audit_policy_lock_v31(args.config),
        "FormalBlindV31": lambda: v31.formal_blind_v31(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "BuildFormalComparisonV31": lambda: v31.build_formal_comparison_v31(args.config),
        "ExportFormalTablesV31": lambda: v31.export_formal_tables_v31(args.config),
        "EvaluateFormalPerformanceV31": lambda: v31.evaluate_formal_performance_v31(args.config),
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
