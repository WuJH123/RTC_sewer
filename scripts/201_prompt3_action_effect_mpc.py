from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.prompt3 import action_effect_mpc as p3


def main() -> int:
    parser = argparse.ArgumentParser(description="Project6 V3 Prompt3 action-effect model and MPC stages.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--include-rounds", default="round0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dispatch = {
        "AuditPrompt3Entry": lambda: p3.audit_prompt3_entry(args.config),
        "EvaluatePrompt3EntryGate": lambda: p3.evaluate_prompt3_entry_gate(args.config),
        "BuildActionEffectDataset": lambda: p3.build_action_effect_dataset(args.config, args.include_rounds, args.resume, args.smoke),
        "AuditActionEffectDataset": lambda: p3.audit_action_effect_dataset(args.config),
        "EvaluateActionEffectDatasetGate": lambda: p3.evaluate_action_effect_dataset_gate(args.config, args.smoke),
        "TrainActionEffectBaselineModels": lambda: p3.train_action_effect_baseline_models(args.config, args.smoke, args.max_samples),
        "TrainActionEffectEnsemble": lambda: p3.train_action_effect_ensemble(args.config, args.smoke, args.max_samples, args.epochs, args.ensemble_size, args.seeds),
        "EvaluateActionEffectModelGate": lambda: p3.evaluate_action_effect_model_gate(args.config, args.smoke),
        "CalibrateDevelopmentUncertainty": lambda: p3.calibrate_development_uncertainty(args.config, args.smoke),
        "EvaluateUncertaintyGate": lambda: p3.evaluate_uncertainty_gate(args.config, args.smoke),
        "TrainOODModel": lambda: p3.train_ood_model(args.config, args.smoke),
        "EvaluateOODGate": lambda: p3.evaluate_ood_gate(args.config, args.smoke),
        "TrainSafetyClassifier": lambda: p3.train_safety_classifier(args.config, args.smoke),
        "EvaluateSafetyClassifierGate": lambda: p3.evaluate_safety_classifier_gate(args.config, args.smoke),
        "TrainFallbackSelector": lambda: p3.train_fallback_selector(args.config, args.smoke),
        "EvaluatePrompt3ModelGate": lambda: p3.evaluate_prompt3_model_gate(args.config, args.smoke),
        "BuildPFVFirstDualFallbackMPC": lambda: p3.build_pfvfirst_dualfallback_mpc(args.config, args.smoke),
        "AuditMPCContract": lambda: p3.audit_mpc_contract(args.config, args.smoke),
        "RunMPCUnitSmoke": lambda: p3.run_mpc_unit_smoke(args.config, args.max_cases),
        "EvaluateMPCUnitGate": lambda: p3.evaluate_mpc_unit_gate(args.config),
        "RunMPCShadowSmoke": lambda: p3.run_mpc_shadow_smoke(args.config, args.max_events, args.workers, args.resume),
        "RunMPCShadowDevelopment": lambda: p3.run_mpc_shadow_smoke(args.config, args.max_events, args.workers, args.resume),
        "EvaluateMPCShadowGate": lambda: p3.evaluate_mpc_shadow_gate(args.config),
        "RunMPCClosedLoopSmoke": lambda: p3.run_mpc_closed_loop_smoke(args.config, args.max_events, args.workers, args.resume),
        "EvaluateMPCClosedLoopSmokeGate": lambda: p3.evaluate_mpc_closed_loop_smoke_gate(args.config),
        "AuditAuthoritativeClosedLoopReadiness": lambda: p3.audit_authoritative_closed_loop_readiness(args.config),
        "RunAuthoritativeClosedLoopDev": lambda: p3.run_authoritative_closed_loop_dev(args.config, args.max_events, args.workers, args.resume),
        "EvaluateAuthoritativeClosedLoopDevGate": lambda: p3.evaluate_authoritative_closed_loop_dev_gate(args.config),
        "RunPairedClosedLoopDev": lambda: p3.run_paired_closed_loop_dev(args.config, args.max_events, args.workers, args.resume),
        "EvaluatePairedClosedLoopDevGate": lambda: p3.evaluate_paired_closed_loop_dev_gate(args.config),
        "BuildEvaluationEventSplits": lambda: p3.build_evaluation_event_splits(args.config),
        "AuditEvaluationEventSplits": lambda: p3.audit_evaluation_event_splits(args.config),
        "CalibrationA": lambda: p3.calibration_a(args.config, args.max_events, args.workers, args.resume),
        "EvaluateCalibrationAGate": lambda: p3.evaluate_calibration_a_gate(args.config),
        "LockedValidationB": lambda: p3.locked_validation_b(args.config, args.max_events, args.workers, args.resume),
        "EvaluateLockedValidationBGate": lambda: p3.evaluate_locked_validation_b_gate(args.config),
        "PolicyLock": lambda: p3.policy_lock(args.config),
        "AuditPolicyLock": lambda: p3.audit_policy_lock(args.config),
        "FormalBlind": lambda: p3.formal_blind(args.config, args.max_events, args.workers, args.resume),
        "BuildFormalPairedComparison": lambda: p3.build_formal_paired_comparison(args.config),
        "EvaluateFormalPerformanceGate": lambda: p3.evaluate_formal_performance_gate(args.config),
        "ExportFormalPaperTables": lambda: p3.export_formal_paper_tables(args.config),
        "EvaluatePrompt3Completion": lambda: p3.evaluate_prompt3_completion(args.config),
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
