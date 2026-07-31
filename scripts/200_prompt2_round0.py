from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.data import round0_prompt2 as p2


def main() -> int:
    parser = argparse.ArgumentParser(description="Project6 V3 Prompt2 Round0 control and dataset stages.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-effective-candidates", type=int, default=1800)
    parser.add_argument("--target-fit-events", type=int, default=36)
    parser.add_argument("--target-checkpoints", type=int, default=144)
    parser.add_argument("--max-per-event", type=int, default=6)
    parser.add_argument("--reserve-candidates", type=int, default=400)
    parser.add_argument("--pressure-candidates", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--round0-manifest", default=str(p2.ROUND0_DIR / "paired_manifest_round0.csv"))
    parser.add_argument("--acknowledge-round0-manifest", action="store_true")
    parser.add_argument("--round", default="round0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-existing-only", action="store_true")
    args = parser.parse_args()

    dispatch = {
        "AuditPrompt2Entry": lambda: p2.audit_prompt2_entry(args.config),
        "PlanPrompt2FitEventExpansion": lambda: p2.plan_prompt2_fit_event_expansion(args.config, args.target_fit_events, args.seed),
        "AuditPrompt2FitEventExpansion": lambda: p2.audit_prompt2_fit_event_expansion(args.config),
        "PlanPrompt2BaselineExpansion": lambda: p2.plan_prompt2_baseline_expansion(args.config, args.max_candidates),
        "AuditPrompt2BaselineExpansion": lambda: p2.audit_prompt2_baseline_expansion(args.config),
        "BuildPrompt2ControlCheckpointCandidates": lambda: p2.build_prompt2_control_checkpoint_candidates(args.config),
        "SelectPrompt2ControlCheckpoints": lambda: p2.select_prompt2_control_checkpoints(args.config, args.target_checkpoints, args.max_per_event, args.seed),
        "AuditPrompt2ControlCheckpointSupport": lambda: p2.audit_prompt2_control_checkpoint_support(args.config),
        "BuildPrompt2StateInputManifest": lambda: p2.build_prompt2_state_input_manifest(args.config, args.max_candidates),
        "BuildPrompt2StateFeatures": lambda: p2.build_prompt2_state_features(args.config, args.max_candidates),
        "AuditPrompt2StateCoverage": lambda: p2.audit_prompt2_state_coverage(args.config),
        "EvaluatePrompt2CheckpointSupportGate": lambda: p2.evaluate_prompt2_checkpoint_support_gate(args.config),
        "BuildControlAlignedCheckpointCatalog": lambda: p2.build_control_aligned_checkpoint_catalog(args.config),
        "AuditControlAlignedCheckpointCatalog": lambda: p2.audit_control_aligned_checkpoint_catalog(args.config),
        "BuildRound0CoverageContract": lambda: p2.build_round0_coverage_contract(args.config),
        "PlanRound0": lambda: p2.plan_round0_manifest(args.config, args.target_effective_candidates, args.reserve_candidates, args.pressure_candidates, args.seed),
        "AuditRound0Manifest": lambda: p2.audit_round0_manifest(args.config),
        "PlanRound0HydraulicDryRun": lambda: p2.plan_round0_hydraulic_dryrun(args.config, args.max_candidates),
        "RunRound0HydraulicDryRun": lambda: p2.run_round0_hydraulic_dryrun(args.config, args.max_candidates, args.workers, args.resume),
        "EvaluateRound0HydraulicDryRunGate": p2.evaluate_round0_hydraulic_dryrun_gate,
        "ApproveRound0Manifest": lambda: p2.approve_round0_manifest(args.config, args.round0_manifest, args.acknowledge_round0_manifest),
        "GenerateRound0Pilot": lambda: p2.generate_round0_pilot(args.config, args.max_candidates, args.batch_size, args.workers, args.resume),
        "EvaluateRound0Pilot": p2.evaluate_round0_pilot,
        "ReplanRound0Adaptive": lambda: p2.replan_round0_adaptive(args.config, args.target_effective_candidates),
        "GenerateRound0Batch": lambda: p2.generate_round0_batch(args.config, args.batch_size, args.workers, args.resume, args.refresh_existing_only),
        "BuildRound0Dataset": lambda: p2.build_round0_dataset(args.config, args.round, args.resume),
        "AuditRound0Dataset": lambda: p2.audit_round0_dataset(args.round),
        "EvaluateRound0DataGate": p2.evaluate_round0_data_gate,
        "EvaluateActionEffectTrainingReadiness": p2.evaluate_action_effect_training_readiness,
        "PlanRound1": lambda: p2.plan_round_manifest(args.config, "round1", args.target_effective_candidates, args.seed),
        "AuditRound1Manifest": lambda: p2.audit_round_manifest(args.config, "round1"),
        "ApproveRound1Manifest": lambda: p2.approve_round_manifest(args.config, "round1", args.acknowledge_round0_manifest),
        "GenerateRound1Pilot": lambda: p2.generate_round_pilot(args.config, "round1", args.max_candidates, args.workers, args.resume),
        "EvaluateRound1Pilot": lambda: p2.evaluate_round_pilot("round1"),
        "GenerateRound1Batch": lambda: p2.generate_round_batch(args.config, "round1", args.batch_size, args.workers, args.resume),
        "BuildRound1Dataset": lambda: p2.build_round0_dataset(args.config, "round1", args.resume),
        "AuditRound1Dataset": lambda: p2.audit_round0_dataset("round1"),
        "EvaluateRound1DataGate": lambda: p2.evaluate_round_data_gate("round1"),
        "EvaluateRound1": lambda: p2.evaluate_round_learning("round1"),
        "GenerateRound1": lambda: p2.generate_round_pilot(args.config, "round1", args.max_candidates, args.workers, args.resume),
        "PlanRound2": lambda: p2.plan_round_manifest(args.config, "round2", args.target_effective_candidates, args.seed),
        "AuditRound2Manifest": lambda: p2.audit_round_manifest(args.config, "round2"),
        "ApproveRound2Manifest": lambda: p2.approve_round_manifest(args.config, "round2", args.acknowledge_round0_manifest),
        "GenerateRound2Pilot": lambda: p2.generate_round_pilot(args.config, "round2", args.max_candidates, args.workers, args.resume),
        "EvaluateRound2Pilot": lambda: p2.evaluate_round_pilot("round2"),
        "GenerateRound2Batch": lambda: p2.generate_round_batch(args.config, "round2", args.batch_size, args.workers, args.resume),
        "BuildRound2Dataset": lambda: p2.build_round0_dataset(args.config, "round2", args.resume),
        "AuditRound2Dataset": lambda: p2.audit_round0_dataset("round2"),
        "EvaluateRound2DataGate": lambda: p2.evaluate_round_data_gate("round2"),
        "EvaluateRound2": lambda: p2.evaluate_round_learning("round2"),
        "GenerateRound2": lambda: p2.generate_round_pilot(args.config, "round2", args.max_candidates, args.workers, args.resume),
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
