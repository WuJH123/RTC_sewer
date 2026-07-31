from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_uses_structured_stage_errors() -> None:
    text = runner_text()
    for name in [
        "DisabledStageError",
        "BlockedStageError",
        "RuntimeStageError",
        "GateFailedError",
        "ContractMismatchError",
        "CliContractError",
    ]:
        assert f"class {name}" in text


def test_disabled_stage_writes_status_and_exits_2() -> None:
    text = runner_text()
    assert 'Write-StageStatus -Stage $Name -Status "disabled" -ExitCode 2' in text
    assert "throw [DisabledStageError]::new" in text
    assert "exit 2" in text


def test_blocked_and_cli_exit_codes_are_explicit() -> None:
    text = runner_text()
    assert "exit 3" in text
    assert "exit 7" in text
    assert 'select_exactly_one_stage' in text


def test_enabled_gat_state_stages_are_registered() -> None:
    text = runner_text()
    for stage in ["RegisterGAT", "AuditGAT", "BuildStateFeatures", "StateCloneTest"]:
        assert f'"{stage}"' in text
    assert "scripts\\134_register_project4_gat.py" in text
    assert "scripts\\135_audit_gat_compatibility.py" in text
    assert "scripts\\136_build_augmented_state.py" in text
    assert "scripts\\138_prepare_state_clone_test.py" in text


def test_disabled_research_stages_remain_disabled_by_absence_from_implemented_list() -> None:
    text = runner_text()
    implemented_block = text.split("$ImplementedStages = @(", 1)[1].split(")", 1)[0]
    for stage in ["BuildDataset", "TrainPilot", "MinimalGate", "RunSmoke"]:
        assert f'"{stage}"' not in implemented_block


def test_rebuild_contract_rejects_old_reference_fallback_marker_outputs() -> None:
    text = runner_text()
    assert "Assert-UpstreamMarkerContainsOutput" in text
    assert "reference_roles\\reference_roles_contract.json" in text
    assert "stale_upstream_outputs" in text


def test_runner_registers_same_state_dual_path_stages() -> None:
    text = runner_text()
    for stage in [
        "RunContinuousReplayDeterminismAudit",
        "RunStateCloneDiagnosticMatrix",
        "RunSameStateReplayEquivalence",
        "EvaluateHotstartCloneGate",
        "EvaluateSameStateBranchGate",
    ]:
        assert f'"{stage}"' in text
    assert "scripts\\175_run_continuous_replay_determinism_audit.py" in text
    assert "scripts\\178_run_same_state_replay_equivalence.py" in text
    assert "scripts\\180_evaluate_same_state_branch_gate.py" in text


def test_runner_registers_hotstart_acceleration_stages() -> None:
    text = runner_text()
    for stage in [
        "BuildCanonicalHotstartCache",
        "DiagnoseHotstartFirstDivergence",
        "AuditHotstartCompatibility",
        "RunHotstartSmoke",
        "EvaluateHotstartSmokeGate",
        "RunHotstartFullValidation",
        "EvaluateHotstartFullGate",
        "BenchmarkHotstartAcceleration",
        "CertifyHotstartCheckpoints",
        "EvaluateHotstartAccelerationReadiness",
    ]:
        assert f'"{stage}"' in text
    for script in [
        "scripts\\181_diagnose_hotstart_first_divergence.py",
        "scripts\\183_build_canonical_hotstart_cache.py",
        "scripts\\184_run_hotstart_smoke.py",
        "scripts\\190_evaluate_hotstart_acceleration_readiness.py",
    ]:
        assert script in text


def test_runner_registers_replay_acceleration_stages() -> None:
    text = runner_text()
    for stage in [
        "AuditRunoffCacheEligibility",
        "BuildRainfallInterfaceCache",
        "BuildRunoffInterfaceCache",
        "AuditRunoffInterfaceEquivalence",
        "EvaluateRunoffCacheGate",
        "BuildReferenceBranchCache",
        "RunCandidatePrefilterAudit",
        "BenchmarkReplayAcceleration",
        "EvaluateReplayAccelerationGate",
    ]:
        assert f'"{stage}"' in text
    for script in [
        "scripts\\191_audit_runoff_cache_eligibility.py",
        "scripts\\193_build_runoff_interface_cache.py",
        "scripts\\195_evaluate_runoff_cache_gate.py",
        "scripts\\198_benchmark_replay_acceleration.py",
        "scripts\\199_evaluate_replay_acceleration_gate.py",
    ]:
        assert script in text


def test_runner_registers_prompt2_round0_stages() -> None:
    text = runner_text()
    for stage in [
        "AuditPrompt2Entry",
        "PlanPrompt2FitEventExpansion",
        "AuditPrompt2FitEventExpansion",
        "PlanPrompt2BaselineExpansion",
        "GeneratePrompt2BaselineExpansion",
        "AuditPrompt2BaselineExpansion",
        "BuildPrompt2ControlCheckpointCandidates",
        "SelectPrompt2ControlCheckpoints",
        "AuditPrompt2ControlCheckpointSupport",
        "BuildPrompt2StateInputManifest",
        "BuildPrompt2StateFeatures",
        "AuditPrompt2StateCoverage",
        "EvaluatePrompt2CheckpointSupportGate",
        "BuildControlAlignedCheckpointCatalog",
        "AuditControlAlignedCheckpointCatalog",
        "BuildRound0CoverageContract",
        "PlanRound0",
        "AuditRound0Manifest",
        "PlanRound0HydraulicDryRun",
        "RunRound0HydraulicDryRun",
        "EvaluateRound0HydraulicDryRunGate",
        "ApproveRound0Manifest",
        "GenerateRound0Pilot",
        "EvaluateRound0Pilot",
        "ReplanRound0Adaptive",
        "GenerateRound0Batch",
        "BuildRound0Dataset",
        "AuditRound0Dataset",
        "EvaluateRound0DataGate",
        "EvaluateActionEffectTrainingReadiness",
    ]:
        assert f'"{stage}"' in text
    assert "scripts\\200_prompt2_round0.py" in text
    assert "Invoke-Prompt2Round0Stage" in text
    assert "AcknowledgeRound0Manifest" in text
    assert "TargetFitEvents" in text
    assert "TargetCheckpoints" in text
    assert "MaxPerEvent" in text
    assert "RefreshExistingOnly" in text
    assert "--refresh-existing-only" in text


def test_runner_batch_completion_uses_actual_round0_output_names() -> None:
    text = runner_text()
    assert 'round0\\round0_generation_manifest.csv' in text
    assert 'round0\\round0_branch_audit.csv' in text
    assert 'round0\\round0_action_audit.csv' in text
    assert 'round0\\round0_kpi_audit.csv' in text
    assert 'round0\\round0_fallback_audit.csv' in text
    assert 'round0\\round0_failures.csv' in text
    assert 'round0\\round0_generation_branch_audit.csv' not in text
    assert 'if ($RefreshExistingOnly) { $args += "--refresh-existing-only" }' in text


def test_runner_registers_prompt3_action_effect_and_mpc_stages() -> None:
    text = runner_text()
    for stage in [
        "AuditPrompt3Entry",
        "EvaluatePrompt3EntryGate",
        "BuildActionEffectDataset",
        "AuditActionEffectDataset",
        "EvaluateActionEffectDatasetGate",
        "TrainActionEffectBaselineModels",
        "TrainActionEffectEnsemble",
        "EvaluateActionEffectModelGate",
        "CalibrateDevelopmentUncertainty",
        "EvaluateUncertaintyGate",
        "TrainOODModel",
        "EvaluateOODGate",
        "TrainSafetyClassifier",
        "EvaluateSafetyClassifierGate",
        "TrainFallbackSelector",
        "EvaluatePrompt3ModelGate",
        "BuildPFVFirstDualFallbackMPC",
        "AuditMPCContract",
        "RunMPCUnitSmoke",
        "EvaluateMPCUnitGate",
        "RunMPCShadowSmoke",
        "EvaluateMPCShadowGate",
        "RunMPCClosedLoopSmoke",
        "EvaluateMPCClosedLoopSmokeGate",
        "AuditAuthoritativeClosedLoopReadiness",
        "RunAuthoritativeClosedLoopDev",
        "EvaluateAuthoritativeClosedLoopDevGate",
        "RunPairedClosedLoopDev",
        "EvaluatePairedClosedLoopDevGate",
        "BuildEvaluationEventSplits",
        "AuditEvaluationEventSplits",
        "CalibrationA",
        "EvaluateCalibrationAGate",
        "LockedValidationB",
        "EvaluateLockedValidationBGate",
        "PolicyLock",
        "AuditPolicyLock",
        "FormalBlind",
        "BuildFormalPairedComparison",
        "EvaluateFormalPerformanceGate",
        "ExportFormalPaperTables",
        "EvaluatePrompt3Completion",
    ]:
        assert f'"{stage}"' in text
    assert "scripts\\201_prompt3_action_effect_mpc.py" in text
    assert "Invoke-Prompt3Stage" in text
    assert "Smoke" in text
    assert "EnsembleSize" in text
    assert "IncludeRounds" in text
