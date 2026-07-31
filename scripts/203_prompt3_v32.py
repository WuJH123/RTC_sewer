from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.prompt3 import action_effect_v32 as v32


def main() -> int:
    parser = argparse.ArgumentParser(description="Project6 V3.2 event-budget, adaptive-K, Round4, and split flow.")
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
        "DiagnoseFormalFailuresV32": lambda: v32.diagnose_v31_failures_v32(args.config, args.max_events),
        "PlanRound4HardNegativesV32": lambda: v32.plan_round4_hard_negatives_v32(args.config, args.target_samples, args.seed),
        "GenerateRound4HardNegativesV32": lambda: v32.generate_round4_hard_negatives_v32(args.config, args.max_samples, args.smoke, args.resume),
        "BuildRound4DatasetV32": lambda: v32.build_round4_dataset_v32(args.config, args.smoke),
        "AuditRound4DatasetV32": lambda: v32.audit_round4_dataset_v32(args.config, args.smoke),
        "TrainActionEffectV32": lambda: v32.train_action_effect_v32(args.config, args.epochs, args.ensemble_size, args.max_samples, args.smoke),
        "CalibrateUncertaintyV32": lambda: v32.calibrate_uncertainty_v32(args.config, args.smoke),
        "TrainOODSafetyFallbackV32": lambda: v32.train_ood_safety_fallback_v32(args.config, args.smoke),
        "EvaluateModelGateV32": lambda: v32.evaluate_model_gate_v32(args.config, args.smoke),
        "RunClosedLoopDevV32": lambda: v32.run_closed_loop_dev_v32(args.config, args.max_events, args.workers, args.resume),
        "BuildEvaluationRainfallAssetsV32": lambda: v32.build_evaluation_rainfall_assets_v32(args.config),
        "BuildEvaluationSplitsV32": lambda: v32.build_evaluation_splits_v32(args.config),
        "AuditEvaluationSplitsV32": lambda: v32.audit_evaluation_splits_v32(args.config),
        "CalibrationAV32": lambda: v32.calibration_a_v32(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "LockedValidationBV32": lambda: v32.locked_validation_b_v32(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "PolicyLockV32": lambda: v32.policy_lock_v32(args.config),
        "AuditPolicyLockV32": lambda: v32.audit_policy_lock_v32(args.config),
        "FormalBlindV32": lambda: v32.formal_blind_v32(args.config, args.max_events, args.workers, args.resume, args.contract_dry_run),
        "RunFormalExtraBaselinesV32": lambda: v32.run_formal_extra_baselines_v32(args.config, args.max_events, args.workers, args.resume),
        "BuildFormalComparisonV32": lambda: v32.build_formal_comparison_v32(args.config),
        "EvaluateFormalPerformanceV32": lambda: v32.evaluate_formal_performance_v32(args.config),
        "ExportFormalTablesV32": lambda: v32.export_formal_tables_v32(args.config),
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
