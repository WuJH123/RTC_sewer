param(
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml",
  [switch]$Status,
  [switch]$Audit,
  [switch]$InitCoverageSchema,
  [switch]$FatalAudit,
  [switch]$AuditNativeRules,
  [switch]$AuditFallbacks,
  [switch]$RegisterGAT,
  [switch]$RecoverGATMetadata,
  [switch]$InspectGATCheckpoints,
  [switch]$AuditGAT,
  [switch]$BuildStateFeatures,
  [switch]$PrepareStateFeatureContracts,
  [switch]$RunGATForwardSmoke,
  [switch]$RunGATReconstructionAudit,
  [switch]$SelectPrimaryGAT,
  [switch]$RunGATRobustnessAudit,
  [switch]$AuditGATValidationProvenance,
  [switch]$BuildGATIndependentValidationCatalog,
  [switch]$GenerateGATIndependentHoldoutTrajectories,
  [switch]$LockGATIndependentValidationManifest,
  [switch]$EvaluateGATRobustnessGate,
  [switch]$BuildStateInputManifest,
  [switch]$BuildEventCatalog,
  [switch]$BuildCheckpointCatalog,
  [switch]$StateCloneTest,
  [switch]$PrepareStateCloneCheckpoints,
  [switch]$EstimateStateCloneNumericalNoise,
  [switch]$RunStateCloneEquivalence,
  [switch]$EvaluateStateCloneGate,
  [switch]$RunContinuousReplayDeterminismAudit,
  [switch]$EvaluateContinuousReplayDeterminismGate,
  [switch]$RunStateCloneDiagnosticMatrix,
  [switch]$RunSameStateReplayEquivalence,
  [switch]$EvaluateHotstartCloneGate,
  [switch]$EvaluateSameStateBranchGate,
  [switch]$BuildCanonicalHotstartCache,
  [switch]$DiagnoseHotstartFirstDivergence,
  [switch]$AuditHotstartCompatibility,
  [switch]$RunHotstartSmoke,
  [switch]$EvaluateHotstartSmokeGate,
  [switch]$RunHotstartFullValidation,
  [switch]$EvaluateHotstartFullGate,
  [switch]$BenchmarkHotstartAcceleration,
  [switch]$CertifyHotstartCheckpoints,
  [switch]$EvaluateHotstartAccelerationReadiness,
  [switch]$AuditRunoffCacheEligibility,
  [switch]$BuildRainfallInterfaceCache,
  [switch]$BuildRunoffInterfaceCache,
  [switch]$AuditRunoffInterfaceEquivalence,
  [switch]$EvaluateRunoffCacheGate,
  [switch]$BuildReferenceBranchCache,
  [switch]$RunCandidatePrefilterAudit,
  [switch]$BenchmarkReplayAcceleration,
  [switch]$EvaluateReplayAccelerationGate,
  [switch]$AuditPrompt2Entry,
  [switch]$PlanPrompt2FitEventExpansion,
  [switch]$AuditPrompt2FitEventExpansion,
  [switch]$PlanPrompt2BaselineExpansion,
  [switch]$GeneratePrompt2BaselineExpansion,
  [switch]$AuditPrompt2BaselineExpansion,
  [switch]$BuildPrompt2ControlCheckpointCandidates,
  [switch]$SelectPrompt2ControlCheckpoints,
  [switch]$AuditPrompt2ControlCheckpointSupport,
  [switch]$BuildPrompt2StateInputManifest,
  [switch]$BuildPrompt2StateFeatures,
  [switch]$AuditPrompt2StateCoverage,
  [switch]$EvaluatePrompt2CheckpointSupportGate,
  [switch]$BuildControlAlignedCheckpointCatalog,
  [switch]$AuditControlAlignedCheckpointCatalog,
  [switch]$BuildRound0CoverageContract,
  [switch]$AuditRound0Manifest,
  [switch]$PlanRound0HydraulicDryRun,
  [switch]$RunRound0HydraulicDryRun,
  [switch]$EvaluateRound0HydraulicDryRunGate,
  [switch]$ApproveRound0Manifest,
  [switch]$GenerateRound0Pilot,
  [switch]$EvaluateRound0Pilot,
  [switch]$ReplanRound0Adaptive,
  [switch]$GenerateRound0Batch,
  [switch]$BuildRound0Dataset,
  [switch]$AuditRound0Dataset,
  [switch]$EvaluateRound0DataGate,
  [switch]$EvaluateActionEffectTrainingReadiness,
  [switch]$AuditPrompt3Entry,
  [switch]$EvaluatePrompt3EntryGate,
  [switch]$BuildActionEffectDataset,
  [switch]$AuditActionEffectDataset,
  [switch]$EvaluateActionEffectDatasetGate,
  [switch]$TrainActionEffectBaselineModels,
  [switch]$TrainActionEffectEnsemble,
  [switch]$EvaluateActionEffectModelGate,
  [switch]$CalibrateDevelopmentUncertainty,
  [switch]$EvaluateUncertaintyGate,
  [switch]$TrainOODModel,
  [switch]$EvaluateOODGate,
  [switch]$TrainSafetyClassifier,
  [switch]$EvaluateSafetyClassifierGate,
  [switch]$TrainFallbackSelector,
  [switch]$EvaluatePrompt3ModelGate,
  [switch]$BuildPFVFirstDualFallbackMPC,
  [switch]$AuditMPCContract,
  [switch]$RunMPCUnitSmoke,
  [switch]$EvaluateMPCUnitGate,
  [switch]$RunMPCShadowSmoke,
  [switch]$RunMPCShadowDevelopment,
  [switch]$EvaluateMPCShadowGate,
  [switch]$RunMPCClosedLoopSmoke,
  [switch]$EvaluateMPCClosedLoopSmokeGate,
  [switch]$AuditAuthoritativeClosedLoopReadiness,
  [switch]$RunAuthoritativeClosedLoopDev,
  [switch]$EvaluateAuthoritativeClosedLoopDevGate,
  [switch]$RunPairedClosedLoopDev,
  [switch]$EvaluatePairedClosedLoopDevGate,
  [switch]$BuildEvaluationEventSplits,
  [switch]$AuditEvaluationEventSplits,
  [switch]$EvaluateCalibrationAGate,
  [switch]$EvaluateLockedValidationBGate,
  [switch]$AuditPolicyLock,
  [switch]$BuildFormalPairedComparison,
  [switch]$EvaluateFormalPerformanceGate,
  [switch]$ExportFormalPaperTables,
  [switch]$DiagnoseFormalFailuresV31,
  [switch]$PlanRound3HardNegativesV31,
  [switch]$GenerateRound3HardNegativesV31,
  [switch]$BuildRound3DatasetV31,
  [switch]$AuditRound3DatasetV31,
  [switch]$TrainActionEffectV31,
  [switch]$CalibrateUncertaintyV31,
  [switch]$TrainOODSafetyFallbackV31,
  [switch]$EvaluateModelGateV31,
  [switch]$RunClosedLoopDevV31,
  [switch]$BuildEvaluationRainfallAssetsV31,
  [switch]$BuildEvaluationSplitsV31,
  [switch]$AuditEvaluationSplitsV31,
  [switch]$CalibrationAV31,
  [switch]$LockedValidationBV31,
  [switch]$PolicyLockV31,
  [switch]$AuditPolicyLockV31,
  [switch]$FormalBlindV31,
  [switch]$BuildFormalComparisonV31,
  [switch]$EvaluateFormalPerformanceV31,
  [switch]$ExportFormalTablesV31,
  [switch]$DiagnoseFormalFailuresV32,
  [switch]$PlanRound4HardNegativesV32,
  [switch]$GenerateRound4HardNegativesV32,
  [switch]$BuildRound4DatasetV32,
  [switch]$AuditRound4DatasetV32,
  [switch]$TrainActionEffectV32,
  [switch]$CalibrateUncertaintyV32,
  [switch]$TrainOODSafetyFallbackV32,
  [switch]$EvaluateModelGateV32,
  [switch]$RunClosedLoopDevV32,
  [switch]$BuildEvaluationRainfallAssetsV32,
  [switch]$BuildEvaluationSplitsV32,
  [switch]$AuditEvaluationSplitsV32,
  [switch]$CalibrationAV32,
  [switch]$LockedValidationBV32,
  [switch]$PolicyLockV32,
  [switch]$AuditPolicyLockV32,
  [switch]$FormalBlindV32,
  [switch]$RunFormalExtraBaselinesV32,
  [switch]$BuildFormalComparisonV32,
  [switch]$EvaluateFormalPerformanceV32,
  [switch]$ExportFormalTablesV32,
  [switch]$DiagnoseV32RegressionV33,
  [switch]$RunModuleAblationV33,
  [switch]$PlanRound5HardNegativesV33,
  [switch]$GenerateRound5HardNegativesV33,
  [switch]$BuildRound5DatasetV33,
  [switch]$AuditRound5DatasetV33,
  [switch]$TrainActionEffectV33,
  [switch]$CalibrateUncertaintyV33,
  [switch]$TrainOODSafetyFallbackV33,
  [switch]$EvaluateModelGateV33,
  [switch]$RunClosedLoopDevV33,
  [switch]$BuildEvaluationRainfallAssetsV33,
  [switch]$BuildEvaluationSplitsV33,
  [switch]$AuditEvaluationSplitsV33,
  [switch]$CalibrationAV33,
  [switch]$LockedValidationBV33,
  [switch]$PolicyLockV33,
  [switch]$AuditPolicyLockV33,
  [switch]$FormalBlindV33,
  [switch]$RunFormalExtraBaselinesV33,
  [switch]$BuildFormalComparisonV33,
  [switch]$EvaluateFormalPerformanceV33,
  [switch]$ExportFormalTablesV33,
  [switch]$EvaluatePrompt3Completion,
  [switch]$RunInternalPFVOpportunityScan,
  [switch]$PlanRound0,
  [switch]$DryRunRound0,
  [switch]$GenerateRound0,
  [switch]$BuildDataset,
  [switch]$TrainPilot,
  [switch]$RunPolicyShiftAudit,
  [switch]$PlanRound1,
  [switch]$AuditRound1Manifest,
  [switch]$ApproveRound1Manifest,
  [switch]$GenerateRound1,
  [switch]$GenerateRound1Pilot,
  [switch]$EvaluateRound1Pilot,
  [switch]$GenerateRound1Batch,
  [switch]$BuildRound1Dataset,
  [switch]$AuditRound1Dataset,
  [switch]$EvaluateRound1DataGate,
  [switch]$EvaluateRound1,
  [switch]$PlanRound2,
  [switch]$AuditRound2Manifest,
  [switch]$ApproveRound2Manifest,
  [switch]$GenerateRound2,
  [switch]$GenerateRound2Pilot,
  [switch]$EvaluateRound2Pilot,
  [switch]$GenerateRound2Batch,
  [switch]$BuildRound2Dataset,
  [switch]$AuditRound2Dataset,
  [switch]$EvaluateRound2DataGate,
  [switch]$EvaluateRound2,
  [switch]$TrainFinal,
  [switch]$MinimalGate,
  [switch]$OptimizerExploitationAudit,
  [switch]$DecisionShadowGate,
  [switch]$BuildMPC,
  [switch]$RunMPCDryRun,
  [switch]$RunSmoke,
  [switch]$CalibrationA,
  [switch]$LockedValidationB,
  [switch]$PolicyLock,
  [switch]$FormalBlind,
  [switch]$EvaluatePrompt2Completion,
  [switch]$EvaluatePrompt2GATReadiness,
  [switch]$ImportPrompt2Artifacts,
  [switch]$AuditReferencesFallbacks,
  [switch]$RebuildContract,
  [switch]$BuildFormalRainfallAssets,
  [switch]$BuildRainfallAssetIndex,
  [switch]$PlanBaselineTrajectories,
  [switch]$GenerateBaselineTrajectories,
  [switch]$BuildCoverageContract,
  [switch]$AuditCurrentTruth,
  [switch]$EvaluatePrompt3AEngineeringGate,
  [switch]$EvaluatePrompt3ARuntimeGate,
  [switch]$EvaluatePrompt3ACompletion,
  [switch]$Resume,
  [switch]$Smoke,
  [switch]$ContractDryRun,
  [switch]$SkipExisting,
  [switch]$RefreshExistingOnly,
  [switch]$ForceReinitializeEmptyCoverage,
  [switch]$AcknowledgeDataLoss,
  [string]$GATRegistryName = "",
  [string]$SelectionDecisionPath = "docs/contracts/gat_primary_selection_decision.json",
  [switch]$AcknowledgeSelection,
  [string]$ValidationManifest = "",
  [switch]$AcknowledgeIndependentHoldout,
  [string]$HoldoutPlan = "",
  [int]$MaxHoldoutEvents = 0,
  [string]$HoldoutPolicies = "no_control",
  [int]$MaxEvents = 0,
  [string]$PolicyFilter = "",
  [int]$Workers = 1,
  [int]$TailMin = 180,
  [int]$TimeStride = 1,
  [int]$TargetEffectiveCandidates = 1800,
  [int]$TargetFitEvents = 36,
  [int]$TargetCheckpoints = 144,
  [int]$MaxPerEvent = 6,
  [int]$ReserveCandidates = 400,
  [int]$PressureCandidates = 90,
  [int]$TargetRound3Samples = 600,
  [int]$Seed = 20260719,
  [int]$MaxCandidates = 20,
  [string]$Round0Manifest = "",
  [switch]$AcknowledgeRound0Manifest,
  [switch]$AcknowledgeRound1Manifest,
  [switch]$AcknowledgeRound2Manifest,
  [string]$Round = "round0",
  [string]$StateInputManifest = "",
  [string]$StateOutputDir = "",
  [int]$MaxSamples = 0,
  [int]$Epochs = 2,
  [int]$EnsembleSize = 2,
  [string]$Seeds = "",
  [int]$MaxCases = 20,
  [string]$IncludeRounds = "round0",
  [ValidateSet("smoke", "full")]
  [string]$StateCloneMode = "full",
  [int]$MaxCheckpoints = 0,
  [string]$SameStateMethod = "",
  [ValidateSet("smoke", "full")]
  [string]$Mode = "full",
  [string]$CandidateCounts = "1,5,10,20",
  [string]$WorkerCounts = "1,2,4",
  [int]$BatchSize = 8,
  [int]$FlushEvery = 32,
  [double]$MaxMemoryGB = 4.0,
  [string]$ScenarioFilter = "",
  [int]$RobustnessSeed = 150,
  [string]$SourceMode = "",
  [string]$TrajectoryRoot = "",
  [string]$StateValidationMode = "full_project6_augmented_state"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

class StageError : System.Exception {
  [string]$Stage
  StageError([string]$stage, [string]$message) : base($message) {
    $this.Stage = $stage
  }
}
class DisabledStageError : StageError {
  DisabledStageError([string]$stage, [string]$message) : base($stage, $message) {}
}
class BlockedStageError : StageError {
  BlockedStageError([string]$stage, [string]$message) : base($stage, $message) {}
}
class RuntimeStageError : StageError {
  RuntimeStageError([string]$stage, [string]$message) : base($stage, $message) {}
}
class GateFailedError : StageError {
  GateFailedError([string]$stage, [string]$message) : base($stage, $message) {}
}
class ContractMismatchError : StageError {
  ContractMismatchError([string]$stage, [string]$message) : base($stage, $message) {}
}
class CliContractError : StageError {
  CliContractError([string]$stage, [string]$message) : base($stage, $message) {}
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunTag = "project6_pfvfirst_dualfallback_10min_v3"
$OutRoot = Join-Path $Root "outputs\$RunTag"
$OutRootV31 = Join-Path $Root "outputs\project6_pfvfirst_dualfallback_10min_v3_1"
$OutRootV32 = Join-Path $Root "outputs\project6_pfvfirst_dualfallback_10min_v3_2"
$OutRootV33 = Join-Path $Root "outputs\project6_pfvfirst_dualfallback_10min_v3_3"
$StatusDir = Join-Path $OutRoot "execution_status"
$MarkerDir = Join-Path $OutRoot "completion_markers"

$ImplementedStages = @(
  "Status",
  "Audit",
  "InitCoverageSchema",
  "RegisterGAT",
  "RecoverGATMetadata",
  "InspectGATCheckpoints",
  "AuditGAT",
  "PrepareStateFeatureContracts",
  "BuildStateFeatures",
  "RunGATForwardSmoke",
  "RunGATReconstructionAudit",
  "SelectPrimaryGAT",
  "RunGATRobustnessAudit",
  "AuditGATValidationProvenance",
  "BuildGATIndependentValidationCatalog",
  "GenerateGATIndependentHoldoutTrajectories",
  "LockGATIndependentValidationManifest",
  "EvaluateGATRobustnessGate",
  "BuildStateInputManifest",
  "ImportPrompt2Artifacts",
  "FatalAudit",
  "AuditReferencesFallbacks",
  "RebuildContract",
  "AuditNativeRules",
  "AuditFallbacks",
  "BuildFormalRainfallAssets",
  "BuildRainfallAssetIndex",
  "BuildEventCatalog",
  "PlanBaselineTrajectories",
  "GenerateBaselineTrajectories",
  "BuildCheckpointCatalog",
  "PrepareStateCloneCheckpoints",
  "EstimateStateCloneNumericalNoise",
  "RunStateCloneEquivalence",
  "EvaluateStateCloneGate",
  "RunContinuousReplayDeterminismAudit",
  "EvaluateContinuousReplayDeterminismGate",
  "RunStateCloneDiagnosticMatrix",
  "RunSameStateReplayEquivalence",
  "EvaluateHotstartCloneGate",
  "EvaluateSameStateBranchGate",
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
  "AuditRunoffCacheEligibility",
  "BuildRainfallInterfaceCache",
  "BuildRunoffInterfaceCache",
  "AuditRunoffInterfaceEquivalence",
  "EvaluateRunoffCacheGate",
  "BuildReferenceBranchCache",
  "RunCandidatePrefilterAudit",
  "BenchmarkReplayAcceleration",
  "EvaluateReplayAccelerationGate",
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
  "RunMPCShadowDevelopment",
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
  "DiagnoseFormalFailuresV31",
  "PlanRound3HardNegativesV31",
  "GenerateRound3HardNegativesV31",
  "BuildRound3DatasetV31",
  "AuditRound3DatasetV31",
  "TrainActionEffectV31",
  "CalibrateUncertaintyV31",
  "TrainOODSafetyFallbackV31",
  "EvaluateModelGateV31",
  "RunClosedLoopDevV31",
  "BuildEvaluationRainfallAssetsV31",
  "BuildEvaluationSplitsV31",
  "AuditEvaluationSplitsV31",
  "CalibrationAV31",
  "LockedValidationBV31",
  "PolicyLockV31",
  "AuditPolicyLockV31",
  "FormalBlindV31",
  "BuildFormalComparisonV31",
  "EvaluateFormalPerformanceV31",
  "ExportFormalTablesV31",
  "DiagnoseFormalFailuresV32",
  "PlanRound4HardNegativesV32",
  "GenerateRound4HardNegativesV32",
  "BuildRound4DatasetV32",
  "AuditRound4DatasetV32",
  "TrainActionEffectV32",
  "CalibrateUncertaintyV32",
  "TrainOODSafetyFallbackV32",
  "EvaluateModelGateV32",
  "RunClosedLoopDevV32",
  "BuildEvaluationRainfallAssetsV32",
  "BuildEvaluationSplitsV32",
  "AuditEvaluationSplitsV32",
  "CalibrationAV32",
  "LockedValidationBV32",
  "PolicyLockV32",
  "AuditPolicyLockV32",
  "FormalBlindV32",
  "RunFormalExtraBaselinesV32",
  "BuildFormalComparisonV32",
  "EvaluateFormalPerformanceV32",
  "ExportFormalTablesV32",
  "DiagnoseV32RegressionV33",
  "RunModuleAblationV33",
  "PlanRound5HardNegativesV33",
  "GenerateRound5HardNegativesV33",
  "BuildRound5DatasetV33",
  "AuditRound5DatasetV33",
  "TrainActionEffectV33",
  "CalibrateUncertaintyV33",
  "TrainOODSafetyFallbackV33",
  "EvaluateModelGateV33",
  "RunClosedLoopDevV33",
  "BuildEvaluationRainfallAssetsV33",
  "BuildEvaluationSplitsV33",
  "AuditEvaluationSplitsV33",
  "CalibrationAV33",
  "LockedValidationBV33",
  "PolicyLockV33",
  "AuditPolicyLockV33",
  "FormalBlindV33",
  "RunFormalExtraBaselinesV33",
  "BuildFormalComparisonV33",
  "EvaluateFormalPerformanceV33",
  "ExportFormalTablesV33",
  "EvaluatePrompt3Completion",
  "PlanRound1",
  "AuditRound1Manifest",
  "ApproveRound1Manifest",
  "GenerateRound1",
  "GenerateRound1Pilot",
  "EvaluateRound1Pilot",
  "GenerateRound1Batch",
  "BuildRound1Dataset",
  "AuditRound1Dataset",
  "EvaluateRound1DataGate",
  "EvaluateRound1",
  "PlanRound2",
  "AuditRound2Manifest",
  "ApproveRound2Manifest",
  "GenerateRound2",
  "GenerateRound2Pilot",
  "EvaluateRound2Pilot",
  "GenerateRound2Batch",
  "BuildRound2Dataset",
  "AuditRound2Dataset",
  "EvaluateRound2DataGate",
  "EvaluateRound2",
  "BuildCoverageContract",
  "AuditCurrentTruth",
  "EvaluatePrompt3AEngineeringGate",
  "EvaluatePrompt3ARuntimeGate",
  "PlanRound0",
  "DryRunRound0",
  "EvaluatePrompt3ACompletion",
  "EvaluatePrompt2Completion",
  "EvaluatePrompt2GATReadiness",
  "StateCloneTest"
)

$AllStages = @(
  "Status",
  "Audit",
  "InitCoverageSchema",
  "FatalAudit",
  "AuditNativeRules",
  "AuditFallbacks",
  "RegisterGAT",
  "RecoverGATMetadata",
  "InspectGATCheckpoints",
  "AuditGAT",
  "BuildStateFeatures",
  "PrepareStateFeatureContracts",
  "RunGATForwardSmoke",
  "RunGATReconstructionAudit",
  "SelectPrimaryGAT",
  "RunGATRobustnessAudit",
  "AuditGATValidationProvenance",
  "BuildGATIndependentValidationCatalog",
  "GenerateGATIndependentHoldoutTrajectories",
  "LockGATIndependentValidationManifest",
  "EvaluateGATRobustnessGate",
  "BuildStateInputManifest",
  "BuildEventCatalog",
  "BuildCheckpointCatalog",
  "StateCloneTest",
  "PrepareStateCloneCheckpoints",
  "EstimateStateCloneNumericalNoise",
  "RunStateCloneEquivalence",
  "EvaluateStateCloneGate",
  "RunContinuousReplayDeterminismAudit",
  "EvaluateContinuousReplayDeterminismGate",
  "RunStateCloneDiagnosticMatrix",
  "RunSameStateReplayEquivalence",
  "EvaluateHotstartCloneGate",
  "EvaluateSameStateBranchGate",
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
  "AuditRunoffCacheEligibility",
  "BuildRainfallInterfaceCache",
  "BuildRunoffInterfaceCache",
  "AuditRunoffInterfaceEquivalence",
  "EvaluateRunoffCacheGate",
  "BuildReferenceBranchCache",
  "RunCandidatePrefilterAudit",
  "BenchmarkReplayAcceleration",
  "EvaluateReplayAccelerationGate",
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
  "RunMPCShadowDevelopment",
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
  "EvaluateCalibrationAGate",
  "EvaluateLockedValidationBGate",
  "AuditPolicyLock",
  "BuildFormalPairedComparison",
  "EvaluateFormalPerformanceGate",
  "ExportFormalPaperTables",
  "DiagnoseFormalFailuresV31",
  "PlanRound3HardNegativesV31",
  "GenerateRound3HardNegativesV31",
  "BuildRound3DatasetV31",
  "AuditRound3DatasetV31",
  "TrainActionEffectV31",
  "CalibrateUncertaintyV31",
  "TrainOODSafetyFallbackV31",
  "EvaluateModelGateV31",
  "RunClosedLoopDevV31",
  "BuildEvaluationRainfallAssetsV31",
  "BuildEvaluationSplitsV31",
  "AuditEvaluationSplitsV31",
  "CalibrationAV31",
  "LockedValidationBV31",
  "PolicyLockV31",
  "AuditPolicyLockV31",
  "FormalBlindV31",
  "BuildFormalComparisonV31",
  "EvaluateFormalPerformanceV31",
  "ExportFormalTablesV31",
  "DiagnoseFormalFailuresV32",
  "PlanRound4HardNegativesV32",
  "GenerateRound4HardNegativesV32",
  "BuildRound4DatasetV32",
  "AuditRound4DatasetV32",
  "TrainActionEffectV32",
  "CalibrateUncertaintyV32",
  "TrainOODSafetyFallbackV32",
  "EvaluateModelGateV32",
  "RunClosedLoopDevV32",
  "BuildEvaluationRainfallAssetsV32",
  "BuildEvaluationSplitsV32",
  "AuditEvaluationSplitsV32",
  "CalibrationAV32",
  "LockedValidationBV32",
  "PolicyLockV32",
  "AuditPolicyLockV32",
  "FormalBlindV32",
  "RunFormalExtraBaselinesV32",
  "BuildFormalComparisonV32",
  "EvaluateFormalPerformanceV32",
  "ExportFormalTablesV32",
  "DiagnoseV32RegressionV33",
  "RunModuleAblationV33",
  "PlanRound5HardNegativesV33",
  "GenerateRound5HardNegativesV33",
  "BuildRound5DatasetV33",
  "AuditRound5DatasetV33",
  "TrainActionEffectV33",
  "CalibrateUncertaintyV33",
  "TrainOODSafetyFallbackV33",
  "EvaluateModelGateV33",
  "RunClosedLoopDevV33",
  "BuildEvaluationRainfallAssetsV33",
  "BuildEvaluationSplitsV33",
  "AuditEvaluationSplitsV33",
  "CalibrationAV33",
  "LockedValidationBV33",
  "PolicyLockV33",
  "AuditPolicyLockV33",
  "FormalBlindV33",
  "RunFormalExtraBaselinesV33",
  "BuildFormalComparisonV33",
  "EvaluateFormalPerformanceV33",
  "ExportFormalTablesV33",
  "EvaluatePrompt3Completion",
  "RunInternalPFVOpportunityScan",
  "PlanRound0",
  "DryRunRound0",
  "GenerateRound0",
  "BuildDataset",
  "TrainPilot",
  "RunPolicyShiftAudit",
  "PlanRound1",
  "AuditRound1Manifest",
  "ApproveRound1Manifest",
  "GenerateRound1",
  "GenerateRound1Pilot",
  "EvaluateRound1Pilot",
  "GenerateRound1Batch",
  "BuildRound1Dataset",
  "AuditRound1Dataset",
  "EvaluateRound1DataGate",
  "EvaluateRound1",
  "PlanRound2",
  "AuditRound2Manifest",
  "ApproveRound2Manifest",
  "GenerateRound2",
  "GenerateRound2Pilot",
  "EvaluateRound2Pilot",
  "GenerateRound2Batch",
  "BuildRound2Dataset",
  "AuditRound2Dataset",
  "EvaluateRound2DataGate",
  "EvaluateRound2",
  "TrainFinal",
  "MinimalGate",
  "OptimizerExploitationAudit",
  "DecisionShadowGate",
  "BuildMPC",
  "RunMPCDryRun",
  "RunSmoke",
  "CalibrationA",
  "LockedValidationB",
  "PolicyLock",
  "FormalBlind",
  "EvaluatePrompt2Completion",
  "EvaluatePrompt2GATReadiness",
  "ImportPrompt2Artifacts",
  "AuditReferencesFallbacks",
  "RebuildContract",
  "BuildFormalRainfallAssets",
  "BuildRainfallAssetIndex",
  "PlanBaselineTrajectories",
  "GenerateBaselineTrajectories",
  "BuildCoverageContract",
  "AuditCurrentTruth",
  "EvaluatePrompt3AEngineeringGate",
  "EvaluatePrompt3ARuntimeGate",
  "EvaluatePrompt3ACompletion"
)

function Ensure-Directories {
  New-Item -ItemType Directory -Force -Path $OutRoot, $StatusDir, $MarkerDir | Out-Null
}

function Get-Sha256OrNull([string]$Path) {
  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
    return $null
  }
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-ConfigHash {
  $cfgPath = Resolve-Path -LiteralPath (Join-Path $Root $Config) -ErrorAction SilentlyContinue
  if ($null -eq $cfgPath) {
    $cfgPath = Resolve-Path -LiteralPath $Config -ErrorAction SilentlyContinue
  }
  if ($null -eq $cfgPath) {
    throw [ContractMismatchError]::new("preflight", "Config not found: $Config")
  }
  return Get-Sha256OrNull $cfgPath.Path
}

function Write-JsonAtomic([string]$Path, [object]$Data) {
  $dir = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $tmp = "$Path.$PID.$([guid]::NewGuid().ToString('N')).tmp"
  $Data | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $tmp -Encoding UTF8
  $lastError = $null
  for ($attempt = 0; $attempt -lt 12; $attempt++) {
    try {
      Move-Item -LiteralPath $tmp -Destination $Path -Force
      return
    } catch {
      $lastError = $_
      Start-Sleep -Milliseconds (50 * ($attempt + 1))
    }
  }
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  throw $lastError
}

function Write-StageStatus {
  param(
    [string]$Stage,
    [string]$Status,
    [int]$ExitCode,
    [AllowNull()][string]$FailureReason,
    [AllowNull()][object]$CompletionMarker,
    [AllowNull()][hashtable]$Extra
  )
  Ensure-Directories
  $statusPath = Join-Path $StatusDir "$Stage.json"
  $payload = [ordered]@{
    stage = $Stage
    enabled = ($ImplementedStages -contains $Stage)
    status = $Status
    substatus = $null
    config_path = $Config
    config_hash = $null
    upstream_dependencies = @()
    upstream_marker_paths = @()
    completion_marker = $(if ($null -eq $CompletionMarker -or [string]$CompletionMarker -eq "") { $null } else { [string]$CompletionMarker })
    last_attempted_time = (Get-Date).ToString("o")
    last_exit_code = $ExitCode
    failure_reason = $FailureReason
    stale_reason = $null
    allowed_to_run = ($ImplementedStages -contains $Stage)
  }
  try {
    $payload.config_hash = Get-ConfigHash
  } catch {
    $payload.config_hash = $null
  }
  if ($null -ne $Extra) {
    foreach ($k in $Extra.Keys) {
      $payload[$k] = $Extra[$k]
    }
  }
  Write-JsonAtomic -Path $statusPath -Data $payload
  Update-StatusIndex
}

function Update-StatusIndex {
  Ensure-Directories
  $items = @()
  foreach ($stage in $AllStages) {
    $path = Join-Path $StatusDir "$stage.json"
    if (Test-Path $path) {
      try {
        $items += (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json)
      } catch {
        $items += [pscustomobject]@{
          stage = $stage
          enabled = ($ImplementedStages -contains $stage)
          status = "status_file_unreadable"
          completion_marker = $null
          last_exit_code = 4
          failure_reason = $_.Exception.Message
        }
      }
    } else {
      $items += [pscustomobject]@{
        stage = $stage
        enabled = ($ImplementedStages -contains $stage)
        status = "not_attempted"
        completion_marker = $null
        last_exit_code = $null
        failure_reason = $null
      }
    }
  }
  $index = [ordered]@{
    created_at = (Get-Date).ToString("o")
    run_tag = $RunTag
    stages = $items
  }
  Write-JsonAtomic -Path (Join-Path $StatusDir "execution_status_index.json") -Data $index
}

function Write-CompletionMarker {
  param(
    [string]$Stage,
    [string[]]$OutputPaths
  )
  Ensure-Directories
  $markerPath = Join-Path $MarkerDir "$Stage`_COMPLETED.json"
  $outputs = @()
  foreach ($p in $OutputPaths) {
    $resolved = Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue
    if ($null -eq $resolved) {
      throw [RuntimeStageError]::new($Stage, "Expected output missing: $p")
    }
    $outputs += [ordered]@{
      path = $resolved.Path
      sha256 = Get-Sha256OrNull $resolved.Path
    }
  }
  $payload = [ordered]@{
    stage = $Stage
    stage_implementation_version = "project6_v3_marker_contract_v2"
    completed_at = (Get-Date).ToString("o")
    created_at = (Get-Date).ToString("o")
    config_path = $Config
    config_hash = Get-ConfigHash
    config_scope_hash = Get-ConfigHash
    input_hashes = @()
    outputs = $outputs
    output_paths = @($outputs | ForEach-Object { $_.path })
    output_hashes = @($outputs | ForEach-Object { $_.sha256 })
    runtime_actually_executed = $true
    scientific_gate_evaluated = ($Stage -like "*Gate*" -or $Stage -like "*Completion*")
  }
  Write-JsonAtomic -Path $markerPath -Data $payload
  return $markerPath
}

function Complete-Stage {
  param(
    [string]$Stage,
    [string[]]$OutputPaths
  )
  $marker = Write-CompletionMarker -Stage $Stage -OutputPaths $OutputPaths
  Write-StageStatus -Stage $Stage -Status "completed" -ExitCode 0 -FailureReason $null -CompletionMarker $marker
  Write-Host "[Project6 PFV-first dual-fallback V3] stage=$Stage status=completed"
  exit 0
}

function Assert-UpstreamCompletion {
  param(
    [string]$Stage,
    [string]$UpstreamStage
  )
  $marker = Join-Path $MarkerDir "$UpstreamStage`_COMPLETED.json"
  if (-not (Test-Path $marker)) {
    Write-StageStatus -Stage $Stage -Status "blocked" -ExitCode 3 -FailureReason "missing_upstream_completion:$UpstreamStage" -CompletionMarker $null -Extra @{
      upstream_dependencies = @($UpstreamStage)
      upstream_marker_paths = @($marker)
    }
    throw [BlockedStageError]::new($Stage, "Missing upstream completion marker: $UpstreamStage")
  }
}

function Validate-CompletionHashes {
  param(
    [string]$Stage,
    [string]$UpstreamStage
  )
  $marker = Join-Path $MarkerDir "$UpstreamStage`_COMPLETED.json"
  if (-not (Test-Path $marker)) {
    throw [BlockedStageError]::new($Stage, "Missing upstream marker for hash validation: $UpstreamStage")
  }
  $record = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
  $current = Get-ConfigHash
  if ($record.config_hash -ne $current) {
    Write-StageStatus -Stage $Stage -Status "contract_mismatch" -ExitCode 6 -FailureReason "stale_upstream_config_hash:$UpstreamStage" -CompletionMarker $null
    throw [ContractMismatchError]::new($Stage, "Upstream marker config hash is stale: $UpstreamStage")
  }
}

function Assert-UpstreamMarkerContainsOutput {
  param(
    [string]$Stage,
    [string]$UpstreamStage,
    [string]$ExpectedOutputSuffix
  )
  $marker = Join-Path $MarkerDir "$UpstreamStage`_COMPLETED.json"
  if (-not (Test-Path $marker)) {
    throw [BlockedStageError]::new($Stage, "Missing upstream marker for output validation: $UpstreamStage")
  }
  $record = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
  $paths = @()
  if ($null -ne $record.outputs) {
    foreach ($item in $record.outputs) {
      $paths += [string]$item.path
    }
  }
  $matched = $false
  foreach ($p in $paths) {
    if ($p.EndsWith($ExpectedOutputSuffix, [System.StringComparison]::OrdinalIgnoreCase)) {
      $matched = $true
    }
  }
  if (-not $matched) {
    Write-StageStatus -Stage $Stage -Status "contract_mismatch" -ExitCode 6 -FailureReason "stale_upstream_outputs:$UpstreamStage" -CompletionMarker $null
    throw [ContractMismatchError]::new($Stage, "Upstream marker output set is stale: $UpstreamStage")
  }
}

function Invoke-PythonStage {
  param(
    [string]$Stage,
    [string]$Script,
    [string[]]$Arguments
  )
  if (-not (Test-Path $Python)) {
    throw [ContractMismatchError]::new($Stage, "Python executable not found: $Python")
  }
  $scriptPath = Join-Path $Root $Script
  if (-not (Test-Path $scriptPath)) {
    throw [RuntimeStageError]::new($Stage, "Stage script not found: $Script")
  }
  & $Python $scriptPath @Arguments
  $code = $LASTEXITCODE
  if ($code -eq 0) {
    return
  }
  if ($code -eq 2) {
    throw [DisabledStageError]::new($Stage, "Python stage disabled: $Stage")
  }
  if ($code -eq 3) {
    throw [BlockedStageError]::new($Stage, "Python stage blocked: $Stage")
  }
  if ($code -eq 5) {
    throw [GateFailedError]::new($Stage, "Python stage gate failed: $Stage")
  }
  if ($code -eq 6) {
    throw [ContractMismatchError]::new($Stage, "Python stage contract mismatch: $Stage")
  }
  if ($code -eq 7) {
    throw [CliContractError]::new($Stage, "Python stage CLI contract error: $Stage")
  }
  throw [RuntimeStageError]::new($Stage, "Python stage failed with exit code $code`: $Stage")
}

function Disable-Stage {
  param([string]$Name)
  Write-StageStatus -Stage $Name -Status "disabled" -ExitCode 2 -FailureReason "not_implemented" -CompletionMarker $null
  throw [DisabledStageError]::new($Name, "Stage [$Name] is disabled: not_implemented")
}

function Get-SelectedStage {
  $selected = @()
  foreach ($stage in $AllStages) {
    $var = Get-Variable -Name $stage -ErrorAction SilentlyContinue
    if ($null -ne $var -and [bool]$var.Value) {
      $selected += $stage
    }
  }
  if ($selected.Count -ne 1) {
    Write-StageStatus -Stage "stage_selection" -Status "failed" -ExitCode 7 -FailureReason "select_exactly_one_stage" -CompletionMarker $null
    throw [CliContractError]::new("stage_selection", "Select exactly one stage; selected=$($selected -join ',')")
  }
  return $selected[0]
}

function Run-Status {
  Ensure-Directories
  Update-StatusIndex
  Write-StageStatus -Stage "Status" -Status "scaffold_only" -ExitCode 0 -FailureReason $null -CompletionMarker $null
  Write-Host "[Project6 PFV-first dual-fallback V3] status index: $(Join-Path $StatusDir 'execution_status_index.json')"
  exit 0
}

function Run-Audit {
  $auditDir = Join-Path $OutRoot "audit"
  New-Item -ItemType Directory -Force -Path $auditDir | Out-Null
  $out = Join-Path $auditDir "static_asset_audit.json"
  Invoke-PythonStage -Stage "Audit" -Script "scripts\125_audit_pfvfirst_dualfallback_assets.py" -Arguments @("--config", $Config, "--out", $out)
  Complete-Stage -Stage "Audit" -OutputPaths @($out)
}

function Run-InitCoverageSchema {
  $coverageDir = Join-Path $OutRoot "coverage"
  $args = @("--config", $Config, "--out-dir", $coverageDir)
  if ($ForceReinitializeEmptyCoverage) { $args += "--force-reinitialize-empty-coverage" }
  if ($AcknowledgeDataLoss) { $args += "--acknowledge-data-loss" }
  Invoke-PythonStage -Stage "InitCoverageSchema" -Script "scripts\126_plan_information_coverage_cases.py" -Arguments $args
  Write-StageStatus -Stage "InitCoverageSchema" -Status "scaffold_only" -ExitCode 0 -FailureReason $null -CompletionMarker $null
  Write-Host "[Project6 PFV-first dual-fallback V3] stage=InitCoverageSchema status=scaffold_only"
  exit 0
}

function Run-RegisterGAT {
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "RegisterGAT" -Script "scripts\134_register_project4_gat.py" -Arguments @("--config", $Config, "--out-dir", $gatDir)
  Complete-Stage -Stage "RegisterGAT" -OutputPaths @(
    (Join-Path $gatDir "gat_external_registry.csv"),
    (Join-Path $gatDir "gat_checkpoint_hashes.csv"),
    (Join-Path $gatDir "gat_registration_report.json")
  )
}

function Run-AuditGAT {
  Assert-UpstreamCompletion -Stage "AuditGAT" -UpstreamStage "InspectGATCheckpoints"
  Validate-CompletionHashes -Stage "AuditGAT" -UpstreamStage "InspectGATCheckpoints"
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "AuditGAT" -Script "scripts\135_audit_gat_compatibility.py" -Arguments @(
    "--config", $Config,
    "--registry", (Join-Path $gatDir "gat_external_registry.csv"),
    "--out-dir", $gatDir
  )
  Complete-Stage -Stage "AuditGAT" -OutputPaths @(
    (Join-Path $gatDir "gat_compatibility_report.json"),
    (Join-Path $gatDir "gat_node_mapping.csv"),
    (Join-Path $gatDir "gat_sensor_mapping.csv"),
    (Join-Path $gatDir "gat_normalization_audit.csv"),
    (Join-Path $gatDir "gat_graph_signature_audit.csv"),
    (Join-Path $gatDir "gat_checkpoint_load_audit.csv")
  )
}

function Run-RecoverGATMetadata {
  Assert-UpstreamCompletion -Stage "RecoverGATMetadata" -UpstreamStage "RegisterGAT"
  Validate-CompletionHashes -Stage "RecoverGATMetadata" -UpstreamStage "RegisterGAT"
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "RecoverGATMetadata" -Script "scripts\139_recover_gat_metadata.py" -Arguments @("--config", $Config, "--out-dir", $gatDir)
  Complete-Stage -Stage "RecoverGATMetadata" -OutputPaths @(
    (Join-Path $gatDir "gat_training_artifact_inventory.csv"),
    (Join-Path $gatDir "gat_metadata_recovery_report.csv"),
    (Join-Path $gatDir "gat_metadata_source_provenance.csv"),
    (Join-Path $gatDir "gat_metadata_conflicts.csv"),
    (Join-Path $gatDir "recovered_metadata\sr0p05.metadata.json"),
    (Join-Path $gatDir "recovered_metadata\sr0p10.metadata.json"),
    (Join-Path $gatDir "recovered_metadata\sr0p15.metadata.json"),
    (Join-Path $gatDir "recovered_metadata\sr0p20.metadata.json"),
    (Join-Path $gatDir "recovered_metadata\sr0p30.metadata.json")
  )
}

function Run-InspectGATCheckpoints {
  Assert-UpstreamCompletion -Stage "InspectGATCheckpoints" -UpstreamStage "RecoverGATMetadata"
  Validate-CompletionHashes -Stage "InspectGATCheckpoints" -UpstreamStage "RecoverGATMetadata"
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "InspectGATCheckpoints" -Script "scripts\140_inspect_gat_checkpoints.py" -Arguments @("--config", $Config, "--out-dir", $gatDir)
  Complete-Stage -Stage "InspectGATCheckpoints" -OutputPaths @(
    (Join-Path $gatDir "gat_checkpoint_load_audit.csv"),
    (Join-Path $gatDir "gat_checkpoint_tensor_audit.csv"),
    (Join-Path $gatDir "gat_strict_load_audit.csv")
  )
}

function Run-PrepareStateFeatureContracts {
  $stateDir = Join-Path $OutRoot "state"
  $gatCompatibility = Join-Path $OutRoot "gat\gat_compatibility_report.json"
  if (-not (Test-Path -LiteralPath $gatCompatibility)) {
    Write-StageStatus -Stage "PrepareStateFeatureContracts" -Status "blocked" -ExitCode 3 -FailureReason "gat_compatibility_report_missing" -CompletionMarker $null
    throw [BlockedStageError]::new("PrepareStateFeatureContracts", "PrepareStateFeatureContracts requires an existing gat_compatibility_report.json.")
  }
  Invoke-PythonStage -Stage "PrepareStateFeatureContracts" -Script "scripts\136_build_augmented_state.py" -Arguments @(
    "--config", $Config,
    "--gat-compatibility", $gatCompatibility,
    "--out-dir", $stateDir
  )
  Complete-Stage -Stage "PrepareStateFeatureContracts" -OutputPaths @(
    (Join-Path $stateDir "state_feature_contract.json"),
    (Join-Path $stateDir "state_feature_schema.json"),
    (Join-Path $stateDir "facility_state_schema.csv"),
    (Join-Path $stateDir "temporal_state_alignment_audit.csv"),
    (Join-Path $stateDir "state_quality_contract.json"),
    (Join-Path $stateDir "local_flow_feature_contract.json")
  )
}

function Run-BuildStateFeatures {
  if ($StateInputManifest -match "<|placeholder|TODO|实际") {
    Write-StageStatus -Stage "BuildStateFeatures" -Status "failed" -ExitCode 7 -FailureReason "StateInputManifest_placeholder_path" -CompletionMarker $null
    throw [CliContractError]::new("BuildStateFeatures", "BuildStateFeatures requires a real -StateInputManifest path, not a placeholder.")
  }
  if ([string]::IsNullOrWhiteSpace($StateInputManifest)) {
    Write-StageStatus -Stage "BuildStateFeatures" -Status "blocked" -ExitCode 3 -FailureReason "StateInputManifest_required" -CompletionMarker $null
    throw [BlockedStageError]::new("BuildStateFeatures", "BuildStateFeatures requires -StateInputManifest with explicit real trajectory/state inputs.")
  }
  Assert-UpstreamCompletion -Stage "BuildStateFeatures" -UpstreamStage "PrepareStateFeatureContracts"
  Validate-CompletionHashes -Stage "BuildStateFeatures" -UpstreamStage "PrepareStateFeatureContracts"
  Assert-UpstreamCompletion -Stage "BuildStateFeatures" -UpstreamStage "BuildStateInputManifest"
  Validate-CompletionHashes -Stage "BuildStateFeatures" -UpstreamStage "BuildStateInputManifest"
  $requiresGatGate = ($StateValidationMode -eq "gat_independent_node_only" -or $StateValidationMode -eq "project4_node_only")
  if ($requiresGatGate) {
    Assert-UpstreamCompletion -Stage "BuildStateFeatures" -UpstreamStage "EvaluateGATRobustnessGate"
    Validate-CompletionHashes -Stage "BuildStateFeatures" -UpstreamStage "EvaluateGATRobustnessGate"
    Assert-UpstreamCompletion -Stage "BuildStateFeatures" -UpstreamStage "SelectPrimaryGAT"
    Validate-CompletionHashes -Stage "BuildStateFeatures" -UpstreamStage "SelectPrimaryGAT"
    $robustnessGatePath = if ($StateValidationMode -eq "gat_independent_node_only") {
      Join-Path $OutRoot "gat\independent_holdout\sr0p15\gat_sr0p15_independent_robustness_gate.json"
    } else {
      Join-Path $OutRoot "gat\gat_sr0p15_robustness_gate.json"
    }
    if (-not (Test-Path -LiteralPath $robustnessGatePath)) {
      Write-StageStatus -Stage "BuildStateFeatures" -Status "blocked" -ExitCode 3 -FailureReason "sr0p15_robustness_gate_missing" -CompletionMarker $null
      throw [BlockedStageError]::new("BuildStateFeatures", "BuildStateFeatures requires the matching sr0p15 robustness gate output.")
    }
    $robustnessGate = Get-Content -LiteralPath $robustnessGatePath -Raw | ConvertFrom-Json
    if ($robustnessGate.status -ne "pass") {
      Write-StageStatus -Stage "BuildStateFeatures" -Status "blocked" -ExitCode 3 -FailureReason "sr0p15_robustness_gate_not_pass" -CompletionMarker $null
      throw [BlockedStageError]::new("BuildStateFeatures", "BuildStateFeatures is blocked until the matching sr0p15 robustness gate status is pass.")
    }
  }
  $stateDir = if ([string]::IsNullOrWhiteSpace($StateOutputDir)) { Join-Path $OutRoot "state" } else { $StateOutputDir }
  $gatLock = Join-Path $OutRoot "gat\gat_primary_selection_lock.json"
  $args = @(
    "--config", $Config,
    "--gat-lock", $gatLock,
    "--state-input-manifest", $StateInputManifest,
    "--out-dir", $stateDir,
    "--state-validation-mode", $StateValidationMode
  )
  if ($MaxSamples -ge 0) { $args += @("--max-samples", [string]$MaxSamples) }
  Invoke-PythonStage -Stage "BuildStateFeatures" -Script "scripts\144_build_runtime_state_features.py" -Arguments $args
  Complete-Stage -Stage "BuildStateFeatures" -OutputPaths @(
    (Join-Path $stateDir "augmented_state_sample_manifest.csv"),
    (Join-Path $stateDir "augmented_state_shape_audit.json"),
    (Join-Path $stateDir "augmented_state_causality_audit.csv"),
    (Join-Path $stateDir "augmented_state_missingness_audit.csv"),
    (Join-Path $stateDir "augmented_state_facility_audit.csv"),
    (Join-Path $stateDir "node_feature_index.json"),
    (Join-Path $stateDir "facility_feature_index.json"),
    (Join-Path $stateDir "storage_feature_index.json"),
    (Join-Path $stateDir "feature_materialization_audit.csv"),
    (Join-Path $stateDir "state_input_gap_report.json")
  )
}

function Run-BuildStateInputManifest {
  Assert-UpstreamCompletion -Stage "BuildStateInputManifest" -UpstreamStage "PrepareStateFeatureContracts"
  Validate-CompletionHashes -Stage "BuildStateInputManifest" -UpstreamStage "PrepareStateFeatureContracts"
  if ([string]::IsNullOrWhiteSpace($SourceMode)) {
    Write-StageStatus -Stage "BuildStateInputManifest" -Status "failed" -ExitCode 7 -FailureReason "SourceMode_required" -CompletionMarker $null
    throw [CliContractError]::new("BuildStateInputManifest", "BuildStateInputManifest requires -SourceMode project4_gat_validation or project6_retrofit_baseline.")
  }
  if ($SourceMode -eq "project6_retrofit_baseline") {
    Assert-UpstreamCompletion -Stage "BuildStateInputManifest" -UpstreamStage "GenerateBaselineTrajectories"
    Validate-CompletionHashes -Stage "BuildStateInputManifest" -UpstreamStage "GenerateBaselineTrajectories"
  }
  $stateInputDir = Join-Path $OutRoot "state_inputs"
  $args = @(
    "--config", $Config,
    "--out-dir", $stateInputDir,
    "--source-mode", $SourceMode
  )
  if (-not [string]::IsNullOrWhiteSpace($TrajectoryRoot)) { $args += @("--trajectory-root", $TrajectoryRoot) }
  if (-not [string]::IsNullOrWhiteSpace($ValidationManifest)) { $args += @("--validation-manifest", $ValidationManifest) }
  if ($MaxSamples -ge 0) { $args += @("--max-samples", [string]$MaxSamples) }
  Invoke-PythonStage -Stage "BuildStateInputManifest" -Script "scripts\146_build_state_input_manifest.py" -Arguments $args
  Complete-Stage -Stage "BuildStateInputManifest" -OutputPaths @(
    (Join-Path $stateInputDir "state_input_manifest_v1.csv"),
    (Join-Path $stateInputDir "state_trajectory_gap_report.json")
  )
}

function Run-GATForwardSmoke {
  Assert-UpstreamCompletion -Stage "RunGATForwardSmoke" -UpstreamStage "AuditGAT"
  Validate-CompletionHashes -Stage "RunGATForwardSmoke" -UpstreamStage "AuditGAT"
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "RunGATForwardSmoke" -Script "scripts\141_run_gat_forward_smoke.py" -Arguments @("--config", $Config, "--out-dir", $gatDir)
  Complete-Stage -Stage "RunGATForwardSmoke" -OutputPaths @(
    (Join-Path $gatDir "gat_forward_smoke_audit.csv"),
    (Join-Path $gatDir "gat_forward_smoke_report.json")
  )
}

function Run-GATReconstructionAudit {
  Assert-UpstreamCompletion -Stage "RunGATReconstructionAudit" -UpstreamStage "RunGATForwardSmoke"
  Validate-CompletionHashes -Stage "RunGATReconstructionAudit" -UpstreamStage "RunGATForwardSmoke"
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "RunGATReconstructionAudit" -Script "scripts\137_run_gat_reconstruction_audit.py" -Arguments @("--config", $Config, "--out-dir", $gatDir)
  Complete-Stage -Stage "RunGATReconstructionAudit" -OutputPaths @(
    (Join-Path $gatDir "gat_reconstruction_audit.csv"),
    (Join-Path $gatDir "gat_unsensed_node_audit.csv"),
    (Join-Path $gatDir "gat_priority_leaveout_audit.csv"),
    (Join-Path $gatDir "gat_sentinel_leaveout_audit.csv"),
    (Join-Path $gatDir "gat_highwater_audit.csv"),
    (Join-Path $gatDir "gat_sensor_failure_audit.csv"),
    (Join-Path $gatDir "gat_candidate_comparison.csv")
  )
}

function Run-SelectPrimaryGAT {
  Assert-UpstreamCompletion -Stage "SelectPrimaryGAT" -UpstreamStage "AuditGAT"
  Validate-CompletionHashes -Stage "SelectPrimaryGAT" -UpstreamStage "AuditGAT"
  if ([string]::IsNullOrWhiteSpace($GATRegistryName)) {
    Write-StageStatus -Stage "SelectPrimaryGAT" -Status "failed" -ExitCode 7 -FailureReason "GATRegistryName_required" -CompletionMarker $null
    throw [CliContractError]::new("SelectPrimaryGAT", "SelectPrimaryGAT requires -GATRegistryName `"sr0p15`".")
  }
  $gatDir = Join-Path $OutRoot "gat"
  $args = @(
    "--config", $Config,
    "--gat-dir", $gatDir,
    "--registry-name", $GATRegistryName,
    "--selection-decision-path", $SelectionDecisionPath
  )
  if ($AcknowledgeSelection) { $args += "--acknowledge-selection" }
  Invoke-PythonStage -Stage "SelectPrimaryGAT" -Script "scripts\142_select_primary_gat.py" -Arguments $args
  Complete-Stage -Stage "SelectPrimaryGAT" -OutputPaths @(
    (Join-Path $gatDir "gat_primary_selection_lock.json")
  )
}

function Run-GATRobustnessAudit {
  Assert-UpstreamCompletion -Stage "RunGATRobustnessAudit" -UpstreamStage "SelectPrimaryGAT"
  Validate-CompletionHashes -Stage "RunGATRobustnessAudit" -UpstreamStage "SelectPrimaryGAT"
  if ([string]::IsNullOrWhiteSpace($ValidationManifest)) {
    Write-StageStatus -Stage "RunGATRobustnessAudit" -Status "blocked" -ExitCode 3 -FailureReason "independent_validation_manifest_required; current default Project4 transition_cache is diagnostic_contaminated" -CompletionMarker $null
    throw [BlockedStageError]::new("RunGATRobustnessAudit", "RunGATRobustnessAudit requires -ValidationManifest from a locked independent holdout; the old default cache is contaminated.")
  }
  $gatDir = Join-Path $OutRoot "gat\independent_holdout\sr0p15"
  $lockPath = Join-Path $OutRoot "gat\gat_independent_validation_lock.json"
  if (-not (Test-Path -LiteralPath $lockPath)) {
    Write-StageStatus -Stage "RunGATRobustnessAudit" -Status "blocked" -ExitCode 3 -FailureReason "gat_independent_validation_lock_missing" -CompletionMarker $null
    throw [BlockedStageError]::new("RunGATRobustnessAudit", "Independent holdout lock is required before formal sr0p15 robustness audit.")
  }
  $args = @("--config", $Config, "--gat-dir", (Join-Path $OutRoot "gat"), "--out-dir", $gatDir, "--validation-manifest", $ValidationManifest)
  if ($MaxSamples -ge 0) { $args += @("--max-samples", [string]$MaxSamples) }
  if ($BatchSize -gt 0) { $args += @("--batch-size", [string]$BatchSize) }
  if ($FlushEvery -gt 0) { $args += @("--flush-every", [string]$FlushEvery) }
  if ($MaxMemoryGB -gt 0) { $args += @("--max-memory-gb", [string]$MaxMemoryGB) }
  if (-not [string]::IsNullOrWhiteSpace($ScenarioFilter)) { $args += @("--scenario-filter", $ScenarioFilter) }
  if ($RobustnessSeed -gt 0) { $args += @("--seed", [string]$RobustnessSeed) }
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "RunGATRobustnessAudit" -Script "scripts\143_run_sr0p15_robustness_audit.py" -Arguments $args
  Complete-Stage -Stage "RunGATRobustnessAudit" -OutputPaths @(
    (Join-Path $gatDir "gat_sr0p15_validation_dataset_manifest.json"),
    (Join-Path $gatDir "gat_sr0p15_validation_event_support.csv"),
    (Join-Path $gatDir "gat_sr0p15_validation_leakage_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_temporal_dependence_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_node_group_metrics.csv"),
    (Join-Path $gatDir "gat_sr0p15_priority_leaveout_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_sentinel_leaveout_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_highwater_phase_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_sensor_failure_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_latency_repeatability_audit.csv"),
    (Join-Path $gatDir "gat_robustness_memory_plan.json"),
    (Join-Path $gatDir "gat_sr0p15_robustness_gate.json")
  )
}

function Run-BuildGATIndependentValidationCatalog {
  Assert-UpstreamCompletion -Stage "BuildGATIndependentValidationCatalog" -UpstreamStage "SelectPrimaryGAT"
  Validate-CompletionHashes -Stage "BuildGATIndependentValidationCatalog" -UpstreamStage "SelectPrimaryGAT"
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "BuildGATIndependentValidationCatalog" -Script "scripts\150_build_gat_independent_validation_catalog.py" -Arguments @(
    "--config", $Config,
    "--gat-dir", $gatDir
  )
  Complete-Stage -Stage "BuildGATIndependentValidationCatalog" -OutputPaths @(
    (Join-Path $gatDir "gat_validation_asset_inventory.csv"),
    (Join-Path $gatDir "gat_contaminated_event_manifest.csv"),
    (Join-Path $gatDir "gat_contaminated_storm_family_manifest.csv"),
    (Join-Path $gatDir "gat_contaminated_rainfall_hashes.csv"),
    (Join-Path $gatDir "gat_model_selection_event_manifest.csv"),
    (Join-Path $gatDir "gat_independent_validation_candidates.csv"),
    (Join-Path $gatDir "gat_independent_validation_exclusion_audit.csv"),
    (Join-Path $gatDir "gat_independent_validation_manifest.csv"),
    (Join-Path $gatDir "gat_independent_trajectory_plan.csv"),
    (Join-Path $gatDir "gat_independent_validation_catalog_report.json")
  )
}

function Run-LockGATIndependentValidationManifest {
  Assert-UpstreamCompletion -Stage "LockGATIndependentValidationManifest" -UpstreamStage "BuildGATIndependentValidationCatalog"
  Validate-CompletionHashes -Stage "LockGATIndependentValidationManifest" -UpstreamStage "BuildGATIndependentValidationCatalog"
  if ([string]::IsNullOrWhiteSpace($ValidationManifest)) {
    Write-StageStatus -Stage "LockGATIndependentValidationManifest" -Status "failed" -ExitCode 7 -FailureReason "ValidationManifest_required" -CompletionMarker $null
    throw [CliContractError]::new("LockGATIndependentValidationManifest", "LockGATIndependentValidationManifest requires -ValidationManifest.")
  }
  $gatDir = Join-Path $OutRoot "gat"
  $args = @("--config", $Config, "--gat-dir", $gatDir, "--validation-manifest", $ValidationManifest)
  if ($AcknowledgeIndependentHoldout) { $args += "--acknowledge-independent-holdout" }
  Invoke-PythonStage -Stage "LockGATIndependentValidationManifest" -Script "scripts\151_lock_gat_independent_validation_manifest.py" -Arguments $args
  Complete-Stage -Stage "LockGATIndependentValidationManifest" -OutputPaths @(
    (Join-Path $gatDir "gat_independent_validation_lock.json")
  )
}

function Run-GenerateGATIndependentHoldoutTrajectories {
  Assert-UpstreamCompletion -Stage "GenerateGATIndependentHoldoutTrajectories" -UpstreamStage "BuildGATIndependentValidationCatalog"
  Validate-CompletionHashes -Stage "GenerateGATIndependentHoldoutTrajectories" -UpstreamStage "BuildGATIndependentValidationCatalog"
  $gatDir = Join-Path $OutRoot "gat"
  $planPath = if ([string]::IsNullOrWhiteSpace($HoldoutPlan)) {
    Join-Path $gatDir "gat_independent_trajectory_plan.csv"
  } else {
    $HoldoutPlan
  }
  if (-not (Test-Path -LiteralPath $planPath)) {
    Write-StageStatus -Stage "GenerateGATIndependentHoldoutTrajectories" -Status "blocked" -ExitCode 3 -FailureReason "gat_independent_trajectory_plan_missing" -CompletionMarker $null
    throw [BlockedStageError]::new("GenerateGATIndependentHoldoutTrajectories", "Missing independent trajectory plan: $planPath")
  }
  $args = @(
    "--config", $Config,
    "--gat-dir", $gatDir,
    "--plan", $planPath,
    "--max-events", [string]$MaxHoldoutEvents,
    "--policies", $HoldoutPolicies,
    "--workers", [string]$Workers,
    "--tail-min", [string]$TailMin,
    "--control-step-sec", "600",
    "--time-stride", [string]$TimeStride
  )
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "GenerateGATIndependentHoldoutTrajectories" -Script "scripts\152_generate_gat_independent_holdout_trajectories.py" -Arguments $args
  Complete-Stage -Stage "GenerateGATIndependentHoldoutTrajectories" -OutputPaths @(
    (Join-Path $gatDir "independent_holdout\generated_trajectories\gat_holdout_generation_report.json"),
    (Join-Path $gatDir "independent_holdout\generated_trajectories\gat_independent_holdout_summary.csv"),
    (Join-Path $gatDir "independent_holdout\generated_trajectories\gat_independent_holdout_trajectory_schedule.csv"),
    (Join-Path $gatDir "independent_holdout\generated_trajectories\gat_independent_holdout_cache_report.json"),
    (Join-Path $gatDir "independent_holdout\generated_trajectories\sr0p15_cache\gat_independent_holdout_sr0p15_cache.npz"),
    (Join-Path $gatDir "gat_independent_validation_manifest.csv")
  )
}

function Run-EvaluateGATRobustnessGate {
  Assert-UpstreamCompletion -Stage "EvaluateGATRobustnessGate" -UpstreamStage "SelectPrimaryGAT"
  Validate-CompletionHashes -Stage "EvaluateGATRobustnessGate" -UpstreamStage "SelectPrimaryGAT"
  $independent = -not [string]::IsNullOrWhiteSpace($ValidationManifest)
  if ($independent) {
    Assert-UpstreamCompletion -Stage "EvaluateGATRobustnessGate" -UpstreamStage "RunGATRobustnessAudit"
    Validate-CompletionHashes -Stage "EvaluateGATRobustnessGate" -UpstreamStage "RunGATRobustnessAudit"
    $gatDir = Join-Path $OutRoot "gat\independent_holdout\sr0p15"
    $args = @(
      "--config", $Config,
      "--gat-dir", $gatDir,
      "--independent-holdout"
    )
    $outputGate = Join-Path $gatDir "gat_sr0p15_independent_robustness_gate.json"
  } else {
    Assert-UpstreamCompletion -Stage "EvaluateGATRobustnessGate" -UpstreamStage "AuditGATValidationProvenance"
    Validate-CompletionHashes -Stage "EvaluateGATRobustnessGate" -UpstreamStage "AuditGATValidationProvenance"
    $gatDir = Join-Path $OutRoot "gat"
    $args = @(
      "--config", $Config,
      "--gat-dir", $gatDir
    )
    $outputGate = Join-Path $gatDir "gat_sr0p15_robustness_gate.json"
  }
  Invoke-PythonStage -Stage "EvaluateGATRobustnessGate" -Script "scripts\147_evaluate_gat_robustness_gate.py" -Arguments $args
  Complete-Stage -Stage "EvaluateGATRobustnessGate" -OutputPaths @(
    $outputGate
  )
}

function Run-AuditGATValidationProvenance {
  Assert-UpstreamCompletion -Stage "AuditGATValidationProvenance" -UpstreamStage "RunGATRobustnessAudit"
  Validate-CompletionHashes -Stage "AuditGATValidationProvenance" -UpstreamStage "RunGATRobustnessAudit"
  $gatDir = Join-Path $OutRoot "gat"
  Invoke-PythonStage -Stage "AuditGATValidationProvenance" -Script "scripts\149_audit_gat_validation_provenance.py" -Arguments @(
    "--config", $Config,
    "--gat-dir", $gatDir
  )
  Complete-Stage -Stage "AuditGATValidationProvenance" -OutputPaths @(
    (Join-Path $gatDir "gat_training_validation_artifact_inventory.csv"),
    (Join-Path $gatDir "gat_sr0p15_validation_sample_inventory.csv"),
    (Join-Path $gatDir "gat_sr0p15_training_event_manifest.csv"),
    (Join-Path $gatDir "gat_sr0p15_validation_event_manifest.csv"),
    (Join-Path $gatDir "gat_sr0p15_validation_provenance_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_validation_leakage_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_rainfall_near_duplicate_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_split_membership_audit.csv"),
    (Join-Path $gatDir "gat_sr0p15_validation_event_support.csv")
  )
}

function Run-StateCloneTest {
  Assert-UpstreamCompletion -Stage "StateCloneTest" -UpstreamStage "BuildStateFeatures"
  Validate-CompletionHashes -Stage "StateCloneTest" -UpstreamStage "BuildStateFeatures"
  $stateDir = Join-Path $OutRoot "state"
  Invoke-PythonStage -Stage "StateCloneTest" -Script "scripts\138_prepare_state_clone_test.py" -Arguments @("--config", $Config, "--out-dir", $stateDir)
  Complete-Stage -Stage "StateCloneTest" -OutputPaths @(
    (Join-Path $stateDir "state_clone_contract.json"),
    (Join-Path $stateDir "controller_state_manifest.schema.json"),
    (Join-Path $stateDir "state_clone_equivalence.schema.csv")
  )
}

function Run-PrepareStateCloneCheckpoints {
  Assert-UpstreamCompletion -Stage "PrepareStateCloneCheckpoints" -UpstreamStage "BuildCheckpointCatalog"
  Validate-CompletionHashes -Stage "PrepareStateCloneCheckpoints" -UpstreamStage "BuildCheckpointCatalog"
  $cloneDir = Join-Path $OutRoot "state_clone"
  Invoke-PythonStage -Stage "PrepareStateCloneCheckpoints" -Script "scripts\171_prepare_state_clone_checkpoints.py" -Arguments @("--config", $Config, "--out-dir", $cloneDir)
  Complete-Stage -Stage "PrepareStateCloneCheckpoints" -OutputPaths @(
    (Join-Path $cloneDir "state_clone_checkpoint_readiness.csv"),
    (Join-Path $cloneDir "state_clone_checkpoint_readiness_report.json")
  )
}

function Run-EstimateStateCloneNumericalNoise {
  Assert-UpstreamCompletion -Stage "EstimateStateCloneNumericalNoise" -UpstreamStage "PrepareStateCloneCheckpoints"
  Validate-CompletionHashes -Stage "EstimateStateCloneNumericalNoise" -UpstreamStage "PrepareStateCloneCheckpoints"
  $cloneDir = Join-Path $OutRoot "state_clone"
  Invoke-PythonStage -Stage "EstimateStateCloneNumericalNoise" -Script "scripts\172_estimate_state_clone_numerical_noise.py" -Arguments @(
    "--config", $Config,
    "--out-dir", $cloneDir,
    "--max-checkpoints", ([string]$MaxCheckpoints),
    "--workers", ([string]$Workers)
  )
  Complete-Stage -Stage "EstimateStateCloneNumericalNoise" -OutputPaths @(
    (Join-Path $cloneDir "state_clone_numerical_noise.json")
  )
}

function Run-RunStateCloneEquivalence {
  Assert-UpstreamCompletion -Stage "RunStateCloneEquivalence" -UpstreamStage "PrepareStateCloneCheckpoints"
  Validate-CompletionHashes -Stage "RunStateCloneEquivalence" -UpstreamStage "PrepareStateCloneCheckpoints"
  if ($StateCloneMode -eq "full") {
    Assert-UpstreamCompletion -Stage "RunStateCloneEquivalence" -UpstreamStage "EstimateStateCloneNumericalNoise"
    Validate-CompletionHashes -Stage "RunStateCloneEquivalence" -UpstreamStage "EstimateStateCloneNumericalNoise"
  }
  $cloneDir = Join-Path $OutRoot "state_clone"
  $args = @(
    "--config", $Config,
    "--out-dir", $cloneDir,
    "--mode", $StateCloneMode,
    "--max-checkpoints", ([string]$MaxCheckpoints),
    "--workers", ([string]$Workers)
  )
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "RunStateCloneEquivalence" -Script "scripts\173_run_state_clone_equivalence.py" -Arguments $args
  if ($StateCloneMode -eq "smoke") {
    Write-StageStatus -Stage "RunStateCloneEquivalence" -Status "runtime_partial" -ExitCode 0 -FailureReason $null -CompletionMarker $null
    return
  }
  Complete-Stage -Stage "RunStateCloneEquivalence" -OutputPaths @(
    (Join-Path $cloneDir "state_clone_equivalence.csv"),
    (Join-Path $cloneDir "state_clone_controller_memory_audit.csv"),
    (Join-Path $cloneDir "state_clone_timeline_audit.csv"),
    (Join-Path $cloneDir "state_clone_report.json")
  )
}

function Run-EvaluateStateCloneGate {
  Assert-UpstreamCompletion -Stage "EvaluateStateCloneGate" -UpstreamStage "PrepareStateCloneCheckpoints"
  Validate-CompletionHashes -Stage "EvaluateStateCloneGate" -UpstreamStage "PrepareStateCloneCheckpoints"
  $cloneDir = Join-Path $OutRoot "state_clone"
  Invoke-PythonStage -Stage "EvaluateStateCloneGate" -Script "scripts\174_evaluate_state_clone_gate.py" -Arguments @("--config", $Config, "--out-dir", $cloneDir)
  Complete-Stage -Stage "EvaluateStateCloneGate" -OutputPaths @(
    (Join-Path $cloneDir "state_clone_gate.json")
  )
}

function Run-RunContinuousReplayDeterminismAudit {
  Assert-UpstreamCompletion -Stage "RunContinuousReplayDeterminismAudit" -UpstreamStage "PrepareStateCloneCheckpoints"
  Validate-CompletionHashes -Stage "RunContinuousReplayDeterminismAudit" -UpstreamStage "PrepareStateCloneCheckpoints"
  $cloneDir = Join-Path $OutRoot "state_clone"
  Invoke-PythonStage -Stage "RunContinuousReplayDeterminismAudit" -Script "scripts\175_run_continuous_replay_determinism_audit.py" -Arguments @(
    "--config", $Config,
    "--out-dir", $cloneDir,
    "--max-checkpoints", ([string]$MaxCheckpoints),
    "--workers", ([string]$Workers)
  )
  Complete-Stage -Stage "RunContinuousReplayDeterminismAudit" -OutputPaths @(
    (Join-Path $cloneDir "continuous_replay_determinism.csv"),
    (Join-Path $cloneDir "continuous_replay_determinism_report.json")
  )
}

function Run-EvaluateContinuousReplayDeterminismGate {
  $cloneDir = Join-Path $OutRoot "state_clone"
  Invoke-PythonStage -Stage "EvaluateContinuousReplayDeterminismGate" -Script "scripts\176_evaluate_continuous_replay_determinism_gate.py" -Arguments @("--config", $Config, "--out-dir", $cloneDir)
  Complete-Stage -Stage "EvaluateContinuousReplayDeterminismGate" -OutputPaths @(
    (Join-Path $cloneDir "continuous_replay_determinism_report.json")
  )
}

function Run-RunStateCloneDiagnosticMatrix {
  Assert-UpstreamCompletion -Stage "RunStateCloneDiagnosticMatrix" -UpstreamStage "PrepareStateCloneCheckpoints"
  Validate-CompletionHashes -Stage "RunStateCloneDiagnosticMatrix" -UpstreamStage "PrepareStateCloneCheckpoints"
  $cloneDir = Join-Path $OutRoot "state_clone"
  $args = @(
    "--config", $Config,
    "--out-dir", $cloneDir,
    "--max-checkpoints", ([string]$MaxCheckpoints),
    "--workers", ([string]$Workers)
  )
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "RunStateCloneDiagnosticMatrix" -Script "scripts\177_run_state_clone_diagnostic_matrix.py" -Arguments $args
  Complete-Stage -Stage "RunStateCloneDiagnosticMatrix" -OutputPaths @(
    (Join-Path $cloneDir "state_clone_diagnostic_matrix.csv"),
    (Join-Path $cloneDir "state_clone_diagnostic_report.json"),
    (Join-Path $cloneDir "state_clone_object_order_audit.csv"),
    (Join-Path $cloneDir "state_clone_object_order_report.json"),
    (Join-Path $cloneDir "round0_control_aligned_checkpoint_audit.csv")
  )
}

function Run-RunSameStateReplayEquivalence {
  Assert-UpstreamCompletion -Stage "RunSameStateReplayEquivalence" -UpstreamStage "RunContinuousReplayDeterminismAudit"
  Validate-CompletionHashes -Stage "RunSameStateReplayEquivalence" -UpstreamStage "RunContinuousReplayDeterminismAudit"
  $cloneDir = Join-Path $OutRoot "state_clone"
  $args = @(
    "--config", $Config,
    "--out-dir", $cloneDir,
    "--mode", $Mode,
    "--max-checkpoints", ([string]$MaxCheckpoints),
    "--workers", ([string]$Workers)
  )
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "RunSameStateReplayEquivalence" -Script "scripts\178_run_same_state_replay_equivalence.py" -Arguments $args
  if ($Mode -eq "smoke") {
    Write-StageStatus -Stage "RunSameStateReplayEquivalence" -Status "runtime_partial" -ExitCode 0 -FailureReason $null -CompletionMarker $null
    return
  }
  Complete-Stage -Stage "RunSameStateReplayEquivalence" -OutputPaths @(
    (Join-Path $cloneDir "same_state_replay_equivalence.csv"),
    (Join-Path $cloneDir "same_state_replay_report.json"),
    (Join-Path $cloneDir "same_state_method_selection.json")
  )
}

function Run-EvaluateHotstartCloneGate {
  $cloneDir = Join-Path $OutRoot "state_clone"
  Invoke-PythonStage -Stage "EvaluateHotstartCloneGate" -Script "scripts\179_evaluate_hotstart_clone_gate.py" -Arguments @("--config", $Config, "--out-dir", $cloneDir)
  Complete-Stage -Stage "EvaluateHotstartCloneGate" -OutputPaths @(
    (Join-Path $cloneDir "hotstart_clone_gate.json")
  )
}

function Run-EvaluateSameStateBranchGate {
  $cloneDir = Join-Path $OutRoot "state_clone"
  Invoke-PythonStage -Stage "EvaluateSameStateBranchGate" -Script "scripts\180_evaluate_same_state_branch_gate.py" -Arguments @("--config", $Config, "--out-dir", $cloneDir)
  Complete-Stage -Stage "EvaluateSameStateBranchGate" -OutputPaths @(
    (Join-Path $cloneDir "same_state_branch_gate.json"),
    (Join-Path $cloneDir "same_state_method_selection.json")
  )
}

function Run-DiagnoseHotstartFirstDivergence {
  $hotDir = Join-Path $OutRoot "hotstart"
  Invoke-PythonStage -Stage "DiagnoseHotstartFirstDivergence" -Script "scripts\181_diagnose_hotstart_first_divergence.py" -Arguments @(
    "--config", $Config,
    "--out-dir", $hotDir,
    "--max-checkpoints", [string]$MaxCheckpoints,
    "--workers", [string]$Workers
  )
  Complete-Stage -Stage "DiagnoseHotstartFirstDivergence" -OutputPaths @(
    (Join-Path $hotDir "hotstart_first_divergence.csv"),
    (Join-Path $hotDir "hotstart_first_divergence_report.json")
  )
}

function Run-AuditHotstartCompatibility {
  $hotDir = Join-Path $OutRoot "hotstart"
  Invoke-PythonStage -Stage "AuditHotstartCompatibility" -Script "scripts\182_audit_hotstart_compatibility.py" -Arguments @(
    "--config", $Config,
    "--out-dir", $hotDir,
    "--max-checkpoints", [string]$MaxCheckpoints
  )
  Complete-Stage -Stage "AuditHotstartCompatibility" -OutputPaths @(
    (Join-Path $hotDir "hotstart_compatibility_signature.json"),
    (Join-Path $hotDir "hotstart_object_order_audit.csv"),
    (Join-Path $hotDir "hotstart_engine_compatibility_audit.json")
  )
}

function Run-BuildCanonicalHotstartCache {
  $hotDir = Join-Path $OutRoot "hotstart"
  $args = @(
    "--config", $Config,
    "--out-dir", $hotDir,
    "--max-checkpoints", [string]$MaxCheckpoints,
    "--workers", [string]$Workers
  )
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "BuildCanonicalHotstartCache" -Script "scripts\183_build_canonical_hotstart_cache.py" -Arguments $args
  Complete-Stage -Stage "BuildCanonicalHotstartCache" -OutputPaths @(
    (Join-Path $hotDir "replay_oracle_lock.json"),
    (Join-Path $hotDir "replay_oracle_checkpoint_index.csv"),
    (Join-Path $hotDir "hotstart_cache_index.csv"),
    (Join-Path $hotDir "hotstart_cache_report.json")
  )
}

function Run-RunHotstartSmoke {
  $hotDir = Join-Path $OutRoot "hotstart"
  $args = @(
    "--config", $Config,
    "--out-dir", $hotDir,
    "--max-checkpoints", [string]$MaxCheckpoints,
    "--workers", [string]$Workers
  )
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "RunHotstartSmoke" -Script "scripts\184_run_hotstart_smoke.py" -Arguments $args
  Complete-Stage -Stage "RunHotstartSmoke" -OutputPaths @(
    (Join-Path $hotDir "hotstart_smoke_certification.csv"),
    (Join-Path $hotDir "hotstart_smoke_gate.json")
  )
}

function Run-EvaluateHotstartSmokeGate {
  $hotDir = Join-Path $OutRoot "hotstart"
  Invoke-PythonStage -Stage "EvaluateHotstartSmokeGate" -Script "scripts\185_evaluate_hotstart_smoke_gate.py" -Arguments @("--config", $Config, "--out-dir", $hotDir)
  Complete-Stage -Stage "EvaluateHotstartSmokeGate" -OutputPaths @(
    (Join-Path $hotDir "hotstart_smoke_gate.json")
  )
}

function Run-RunHotstartFullValidation {
  $hotDir = Join-Path $OutRoot "hotstart"
  $args = @(
    "--config", $Config,
    "--out-dir", $hotDir,
    "--max-checkpoints", [string]$MaxCheckpoints,
    "--workers", [string]$Workers
  )
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "RunHotstartFullValidation" -Script "scripts\186_run_hotstart_full_validation.py" -Arguments $args
  Complete-Stage -Stage "RunHotstartFullValidation" -OutputPaths @(
    (Join-Path $hotDir "hotstart_full_certification.csv"),
    (Join-Path $hotDir "hotstart_full_gate.json")
  )
}

function Run-EvaluateHotstartFullGate {
  $hotDir = Join-Path $OutRoot "hotstart"
  Invoke-PythonStage -Stage "EvaluateHotstartFullGate" -Script "scripts\187_evaluate_hotstart_full_gate.py" -Arguments @("--config", $Config, "--out-dir", $hotDir)
  Complete-Stage -Stage "EvaluateHotstartFullGate" -OutputPaths @(
    (Join-Path $hotDir "hotstart_full_gate.json")
  )
}

function Run-CertifyHotstartCheckpoints {
  $hotDir = Join-Path $OutRoot "hotstart"
  Invoke-PythonStage -Stage "CertifyHotstartCheckpoints" -Script "scripts\188_certify_hotstart_checkpoints.py" -Arguments @("--config", $Config, "--out-dir", $hotDir)
  Complete-Stage -Stage "CertifyHotstartCheckpoints" -OutputPaths @(
    (Join-Path $hotDir "hotstart_checkpoint_certification.csv"),
    (Join-Path $hotDir "hotstart_checkpoint_certification_report.json")
  )
}

function Run-BenchmarkHotstartAcceleration {
  $hotDir = Join-Path $OutRoot "hotstart"
  Invoke-PythonStage -Stage "BenchmarkHotstartAcceleration" -Script "scripts\189_benchmark_hotstart_acceleration.py" -Arguments @(
    "--config", $Config,
    "--out-dir", $hotDir,
    "--candidate-counts", $CandidateCounts,
    "--worker-counts", $WorkerCounts
  )
  Complete-Stage -Stage "BenchmarkHotstartAcceleration" -OutputPaths @(
    (Join-Path $hotDir "hotstart_performance_benchmark.csv"),
    (Join-Path $hotDir "hotstart_amortized_speedup.json"),
    (Join-Path $hotDir "hotstart_worker_scaling.csv")
  )
}

function Run-EvaluateHotstartAccelerationReadiness {
  $hotDir = Join-Path $OutRoot "hotstart"
  Invoke-PythonStage -Stage "EvaluateHotstartAccelerationReadiness" -Script "scripts\190_evaluate_hotstart_acceleration_readiness.py" -Arguments @("--config", $Config, "--out-dir", $hotDir)
  Complete-Stage -Stage "EvaluateHotstartAccelerationReadiness" -OutputPaths @(
    (Join-Path $hotDir "hotstart_acceleration_readiness_gate.json")
  )
}

function Run-AuditRunoffCacheEligibility {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  Invoke-PythonStage -Stage "AuditRunoffCacheEligibility" -Script "scripts\191_audit_runoff_cache_eligibility.py" -Arguments @("--config", $Config, "--out-dir", $cacheDir)
  Complete-Stage -Stage "AuditRunoffCacheEligibility" -OutputPaths @(
    (Join-Path $cacheDir "runoff_cache_eligibility.csv"),
    (Join-Path $cacheDir "runoff_cache_eligibility_report.json")
  )
}

function Run-BuildRainfallInterfaceCache {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  $args = @("--config", $Config, "--out-dir", $cacheDir, "--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "BuildRainfallInterfaceCache" -Script "scripts\192_build_rainfall_interface_cache.py" -Arguments $args
  Complete-Stage -Stage "BuildRainfallInterfaceCache" -OutputPaths @(
    (Join-Path $cacheDir "rainfall_interface_cache_index.csv"),
    (Join-Path $cacheDir "rainfall_interface_cache_report.json")
  )
}

function Run-BuildRunoffInterfaceCache {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  $args = @("--config", $Config, "--out-dir", $cacheDir, "--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-PythonStage -Stage "BuildRunoffInterfaceCache" -Script "scripts\193_build_runoff_interface_cache.py" -Arguments $args
  Complete-Stage -Stage "BuildRunoffInterfaceCache" -OutputPaths @(
    (Join-Path $cacheDir "runoff_interface_cache_index.csv"),
    (Join-Path $cacheDir "runoff_interface_cache_report.json")
  )
}

function Run-AuditRunoffInterfaceEquivalence {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  Invoke-PythonStage -Stage "AuditRunoffInterfaceEquivalence" -Script "scripts\194_audit_runoff_interface_equivalence.py" -Arguments @(
    "--config", $Config,
    "--out-dir", $cacheDir,
    "--max-events", [string]$MaxEvents,
    "--workers", [string]$Workers
  )
  Complete-Stage -Stage "AuditRunoffInterfaceEquivalence" -OutputPaths @(
    (Join-Path $cacheDir "runoff_interface_equivalence_audit.csv"),
    (Join-Path $cacheDir "runoff_interface_equivalence_report.json")
  )
}

function Run-EvaluateRunoffCacheGate {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  Invoke-PythonStage -Stage "EvaluateRunoffCacheGate" -Script "scripts\195_evaluate_runoff_cache_gate.py" -Arguments @("--config", $Config, "--out-dir", $cacheDir)
  Complete-Stage -Stage "EvaluateRunoffCacheGate" -OutputPaths @(
    (Join-Path $cacheDir "runoff_cache_gate.json")
  )
}

function Run-BuildReferenceBranchCache {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  Invoke-PythonStage -Stage "BuildReferenceBranchCache" -Script "scripts\196_build_reference_branch_cache.py" -Arguments @("--config", $Config, "--out-dir", $cacheDir)
  Complete-Stage -Stage "BuildReferenceBranchCache" -OutputPaths @(
    (Join-Path $cacheDir "reference_branch_cache_index.csv"),
    (Join-Path $cacheDir "reference_branch_cache_audit.csv"),
    (Join-Path $cacheDir "reference_branch_cache_report.json")
  )
}

function Run-RunCandidatePrefilterAudit {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  Invoke-PythonStage -Stage "RunCandidatePrefilterAudit" -Script "scripts\197_candidate_prefilter_audit.py" -Arguments @("--config", $Config, "--out-dir", $cacheDir)
  Complete-Stage -Stage "RunCandidatePrefilterAudit" -OutputPaths @(
    (Join-Path $cacheDir "candidate_prefilter_audit.csv"),
    (Join-Path $cacheDir "candidate_prefilter_summary.json"),
    (Join-Path $cacheDir "binary_pump_direction_support.csv")
  )
}

function Run-BenchmarkReplayAcceleration {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  Invoke-PythonStage -Stage "BenchmarkReplayAcceleration" -Script "scripts\198_benchmark_replay_acceleration.py" -Arguments @(
    "--config", $Config,
    "--out-dir", $cacheDir,
    "--candidate-counts", $CandidateCounts,
    "--worker-counts", $WorkerCounts
  )
  Complete-Stage -Stage "BenchmarkReplayAcceleration" -OutputPaths @(
    (Join-Path $cacheDir "replay_acceleration_benchmark.csv"),
    (Join-Path $cacheDir "replay_acceleration_benchmark_report.json")
  )
}

function Run-EvaluateReplayAccelerationGate {
  $cacheDir = Join-Path $OutRoot "interface_cache"
  Invoke-PythonStage -Stage "EvaluateReplayAccelerationGate" -Script "scripts\199_evaluate_replay_acceleration_gate.py" -Arguments @("--config", $Config, "--out-dir", $cacheDir)
  Complete-Stage -Stage "EvaluateReplayAccelerationGate" -OutputPaths @(
    (Join-Path $cacheDir "replay_acceleration_gate.json")
  )
}

function Invoke-Prompt2Round0Stage {
  param(
    [string]$StageName,
    [string[]]$ExtraArgs
  )
  $args = @("--stage", $StageName, "--config", $Config) + $ExtraArgs
  Invoke-PythonStage -Stage $StageName -Script "scripts\200_prompt2_round0.py" -Arguments $args
}

function Invoke-Prompt3Stage {
  param(
    [string]$StageName,
    [string[]]$ExtraArgs
  )
  $args = @("--stage", $StageName, "--config", $Config) + $ExtraArgs
  if ($Smoke) { $args += "--smoke" }
  Invoke-PythonStage -Stage $StageName -Script "scripts\201_prompt3_action_effect_mpc.py" -Arguments $args
}

function Run-AuditPrompt2Entry {
  Invoke-Prompt2Round0Stage -StageName "AuditPrompt2Entry" -ExtraArgs @()
  Complete-Stage -Stage "AuditPrompt2Entry" -OutputPaths @(
    (Join-Path $OutRoot "prompt2\prompt2_entry_audit.json"),
    (Join-Path $OutRoot "prompt2\prompt2_entry_inputs.csv"),
    (Join-Path $OutRoot "gates\prompt2_entry_gate.json")
  )
}

function Run-PlanPrompt2FitEventExpansion {
  Assert-UpstreamCompletion -Stage "PlanPrompt2FitEventExpansion" -UpstreamStage "AuditPrompt2Entry"
  Invoke-Prompt2Round0Stage -StageName "PlanPrompt2FitEventExpansion" -ExtraArgs @("--target-fit-events", [string]$TargetFitEvents, "--seed", [string]$Seed)
  Complete-Stage -Stage "PlanPrompt2FitEventExpansion" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_fit_expansion\prompt2_fit_event_expansion_plan.csv"),
    (Join-Path $OutRoot "prompt2_fit_expansion\prompt2_fit_event_support.csv"),
    (Join-Path $OutRoot "prompt2_fit_expansion\prompt2_storm_family_support.csv"),
    (Join-Path $OutRoot "prompt2_fit_expansion\prompt2_rainfall_asset_audit.csv"),
    (Join-Path $OutRoot "prompt2_fit_expansion\prompt2_event_expansion_exclusions.csv"),
    (Join-Path $OutRoot "prompt2_fit_expansion\prompt2_event_expansion_report.json")
  )
}

function Run-AuditPrompt2FitEventExpansion {
  Invoke-Prompt2Round0Stage -StageName "AuditPrompt2FitEventExpansion" -ExtraArgs @()
  Complete-Stage -Stage "AuditPrompt2FitEventExpansion" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_fit_expansion\prompt2_fit_event_expansion_audit.json")
  )
}

function Run-PlanPrompt2BaselineExpansion {
  Assert-UpstreamCompletion -Stage "PlanPrompt2BaselineExpansion" -UpstreamStage "AuditPrompt2FitEventExpansion"
  Invoke-Prompt2Round0Stage -StageName "PlanPrompt2BaselineExpansion" -ExtraArgs @("--max-candidates", [string]$MaxEvents)
  Complete-Stage -Stage "PlanPrompt2BaselineExpansion" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_baseline_expansion\baseline_trajectory_plan.csv"),
    (Join-Path $OutRoot "prompt2_baseline_expansion\prompt2_baseline_expansion_plan_report.json")
  )
}

function Run-GeneratePrompt2BaselineExpansion {
  Assert-UpstreamCompletion -Stage "GeneratePrompt2BaselineExpansion" -UpstreamStage "PlanPrompt2BaselineExpansion"
  $args = @(
    "--config", $Config,
    "--plan", (Join-Path $OutRoot "prompt2_baseline_expansion\baseline_trajectory_plan.csv"),
    "--out-dir", (Join-Path $OutRoot "prompt2_baseline_expansion"),
    "--max-events", [string]$MaxEvents,
    "--workers", [string]$Workers,
    "--tail-min", [string]$TailMin
  )
  if ($Resume) { $args += "--resume" }
  if ($SkipExisting) { $args += "--skip-existing" }
  if ($RefreshExistingOnly) { $args += "--refresh-existing-only" }
  Invoke-PythonStage -Stage "GeneratePrompt2BaselineExpansion" -Script "scripts\160_generate_baseline_trajectories.py" -Arguments $args
  Complete-Stage -Stage "GeneratePrompt2BaselineExpansion" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_baseline_expansion\baseline_trajectory_manifest.csv"),
    (Join-Path $OutRoot "prompt2_baseline_expansion\baseline_recovery_audit.csv"),
    (Join-Path $OutRoot "prompt2_baseline_expansion\baseline_checkpoint_audit.csv"),
    (Join-Path $OutRoot "prompt2_baseline_expansion\baseline_trajectory_failures.csv"),
    (Join-Path $OutRoot "prompt2_baseline_expansion\baseline_trajectory_status.csv"),
    (Join-Path $OutRoot "prompt2_baseline_expansion\trajectory_quality_report.json")
  )
}

function Run-AuditPrompt2BaselineExpansion {
  Invoke-Prompt2Round0Stage -StageName "AuditPrompt2BaselineExpansion" -ExtraArgs @()
  Complete-Stage -Stage "AuditPrompt2BaselineExpansion" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_baseline_expansion\prompt2_baseline_expansion_audit_report.json")
  )
}

function Run-BuildPrompt2ControlCheckpointCandidates {
  Assert-UpstreamCompletion -Stage "BuildPrompt2ControlCheckpointCandidates" -UpstreamStage "AuditPrompt2BaselineExpansion"
  Invoke-Prompt2Round0Stage -StageName "BuildPrompt2ControlCheckpointCandidates" -ExtraArgs @()
  Complete-Stage -Stage "BuildPrompt2ControlCheckpointCandidates" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_control_checkpoints\prompt2_control_checkpoint_candidates.csv"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\control_checkpoint_duplicate_audit.csv"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\control_checkpoint_near_duplicate_audit.csv"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\control_checkpoint_temporal_cluster_audit.csv"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\prompt2_control_checkpoint_candidate_report.json")
  )
}

function Run-SelectPrompt2ControlCheckpoints {
  Invoke-Prompt2Round0Stage -StageName "SelectPrompt2ControlCheckpoints" -ExtraArgs @("--target-checkpoints", [string]$TargetCheckpoints, "--max-per-event", [string]$MaxPerEvent, "--seed", [string]$Seed)
  Complete-Stage -Stage "SelectPrompt2ControlCheckpoints" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_control_checkpoints\prompt2_selected_control_checkpoints.csv"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\control_checkpoint_catalog.csv"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\control_checkpoint_catalog_report.json"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\control_checkpoint_phase_support.csv"),
    (Join-Path $OutRoot "prompt2_control_checkpoints\control_checkpoint_event_support.csv")
  )
}

function Run-AuditPrompt2ControlCheckpointSupport {
  Invoke-Prompt2Round0Stage -StageName "AuditPrompt2ControlCheckpointSupport" -ExtraArgs @()
  Complete-Stage -Stage "AuditPrompt2ControlCheckpointSupport" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_control_checkpoints\prompt2_control_checkpoint_support_audit.json")
  )
}

function Run-BuildPrompt2StateInputManifest {
  Assert-UpstreamCompletion -Stage "BuildPrompt2StateInputManifest" -UpstreamStage "AuditPrompt2ControlCheckpointSupport"
  Invoke-Prompt2Round0Stage -StageName "BuildPrompt2StateInputManifest" -ExtraArgs @("--max-candidates", [string]$MaxSamples)
  Complete-Stage -Stage "BuildPrompt2StateInputManifest" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_state\state_inputs\state_input_manifest_v1.csv"),
    (Join-Path $OutRoot "prompt2_state\prompt2_state_input_manifest_report.json")
  )
}

function Run-BuildPrompt2StateFeatures {
  Assert-UpstreamCompletion -Stage "BuildPrompt2StateFeatures" -UpstreamStage "BuildPrompt2StateInputManifest"
  Invoke-Prompt2Round0Stage -StageName "BuildPrompt2StateFeatures" -ExtraArgs @("--max-candidates", [string]$MaxSamples)
  Complete-Stage -Stage "BuildPrompt2StateFeatures" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_state\state\augmented_state_sample_manifest.csv"),
    (Join-Path $OutRoot "prompt2_state\state\augmented_state_shape_audit.json"),
    (Join-Path $OutRoot "prompt2_state\state\node_feature_index.json"),
    (Join-Path $OutRoot "prompt2_state\state\facility_feature_index.json"),
    (Join-Path $OutRoot "prompt2_state\state\storage_feature_index.json")
  )
}

function Run-AuditPrompt2StateCoverage {
  Invoke-Prompt2Round0Stage -StageName "AuditPrompt2StateCoverage" -ExtraArgs @()
  Complete-Stage -Stage "AuditPrompt2StateCoverage" -OutputPaths @(
    (Join-Path $OutRoot "prompt2_state\prompt2_state_coverage_audit.json")
  )
}

function Run-EvaluatePrompt2CheckpointSupportGate {
  Invoke-Prompt2Round0Stage -StageName "EvaluatePrompt2CheckpointSupportGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluatePrompt2CheckpointSupportGate" -OutputPaths @(
    (Join-Path $OutRoot "gates\prompt2_checkpoint_support_gate.json")
  )
}

function Run-BuildControlAlignedCheckpointCatalog {
  Assert-UpstreamCompletion -Stage "BuildControlAlignedCheckpointCatalog" -UpstreamStage "AuditPrompt2Entry"
  Invoke-Prompt2Round0Stage -StageName "BuildControlAlignedCheckpointCatalog" -ExtraArgs @()
  Complete-Stage -Stage "BuildControlAlignedCheckpointCatalog" -OutputPaths @(
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_catalog.csv"),
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_catalog_report.json"),
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_split_audit.csv"),
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_history_audit.csv"),
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_future_audit.csv"),
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_phase_support.csv"),
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_event_support.csv")
  )
}

function Run-AuditControlAlignedCheckpointCatalog {
  Invoke-Prompt2Round0Stage -StageName "AuditControlAlignedCheckpointCatalog" -ExtraArgs @()
  Complete-Stage -Stage "AuditControlAlignedCheckpointCatalog" -OutputPaths @(
    (Join-Path $OutRoot "control_checkpoints\control_checkpoint_catalog_audit_report.json")
  )
}

function Run-BuildRound0CoverageContract {
  Assert-UpstreamCompletion -Stage "BuildRound0CoverageContract" -UpstreamStage "AuditControlAlignedCheckpointCatalog"
  Invoke-Prompt2Round0Stage -StageName "BuildRound0CoverageContract" -ExtraArgs @()
  Complete-Stage -Stage "BuildRound0CoverageContract" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_coverage_contract.json"),
    (Join-Path $OutRoot "round0\round0_candidate_manifest_schema.csv")
  )
}

function Run-AuditRound0Manifest {
  Invoke-Prompt2Round0Stage -StageName "AuditRound0Manifest" -ExtraArgs @()
  Complete-Stage -Stage "AuditRound0Manifest" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_manifest_audit.csv"),
    (Join-Path $OutRoot "round0\round0_manifest_audit_report.json"),
    (Join-Path $OutRoot "round0\round0_manifest_schema_runtime_audit.csv"),
    (Join-Path $OutRoot "round0\round0_manifest_field_population_audit.csv"),
    (Join-Path $OutRoot "round0\round0_manifest_semantic_mismatch_report.json"),
    (Join-Path $OutRoot "round0\round0_event_leakage_audit.csv"),
    (Join-Path $OutRoot "round0\round0_rainfall_near_duplicate_audit.csv"),
    (Join-Path $OutRoot "round0\round0_split_membership_audit.csv")
  )
}

function Run-PlanRound0HydraulicDryRun {
  Assert-UpstreamCompletion -Stage "PlanRound0HydraulicDryRun" -UpstreamStage "AuditRound0Manifest"
  Invoke-Prompt2Round0Stage -StageName "PlanRound0HydraulicDryRun" -ExtraArgs @("--max-candidates", [string]$MaxCandidates)
  Complete-Stage -Stage "PlanRound0HydraulicDryRun" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_plan.csv"),
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_plan_report.json")
  )
}

function Run-RunRound0HydraulicDryRun {
  $args = @("--max-candidates", [string]$MaxCandidates, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "RunRound0HydraulicDryRun" -ExtraArgs $args
  Complete-Stage -Stage "RunRound0HydraulicDryRun" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_manifest.csv"),
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_branch_audit.csv"),
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_action_audit.csv"),
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_kpi_audit.csv"),
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_fallback_audit.csv"),
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_failures.csv"),
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_report.json")
  )
}

function Run-EvaluateRound0HydraulicDryRunGate {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound0HydraulicDryRunGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound0HydraulicDryRunGate" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_hydraulic_dryrun_gate.json")
  )
}

function Run-ApproveRound0Manifest {
  if ([string]::IsNullOrWhiteSpace($Round0Manifest)) {
    $Round0Manifest = Join-Path $OutRoot "round0\paired_manifest_round0.csv"
  }
  $ackArgs = @("--round0-manifest", $Round0Manifest)
  if ($AcknowledgeRound0Manifest) { $ackArgs += "--acknowledge-round0-manifest" }
  Invoke-Prompt2Round0Stage -StageName "ApproveRound0Manifest" -ExtraArgs $ackArgs
  Complete-Stage -Stage "ApproveRound0Manifest" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_manifest_approval_lock.json")
  )
}

function Run-GenerateRound0Pilot {
  $args = @("--max-candidates", [string]$MaxCandidates, "--batch-size", [string]$BatchSize, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "GenerateRound0Pilot" -ExtraArgs $args
  Complete-Stage -Stage "GenerateRound0Pilot" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_pilot_generation_manifest.csv"),
    (Join-Path $OutRoot "round0\round0_pilot_branch_audit.csv"),
    (Join-Path $OutRoot "round0\round0_pilot_action_audit.csv"),
    (Join-Path $OutRoot "round0\round0_pilot_kpi_audit.csv"),
    (Join-Path $OutRoot "round0\round0_pilot_fallback_audit.csv"),
    (Join-Path $OutRoot "round0\round0_pilot_failures.csv"),
    (Join-Path $OutRoot "round0\round0_pilot_report.json")
  )
}

function Run-EvaluateRound0Pilot {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound0Pilot" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound0Pilot" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_pilot_gate.json")
  )
}

function Run-ReplanRound0Adaptive {
  Invoke-Prompt2Round0Stage -StageName "ReplanRound0Adaptive" -ExtraArgs @("--target-effective-candidates", [string]$TargetEffectiveCandidates)
  Complete-Stage -Stage "ReplanRound0Adaptive" -OutputPaths @(
    (Join-Path $OutRoot "round0\paired_manifest_round0_adaptive.csv"),
    (Join-Path $OutRoot "round0\round0_adaptive_replan_report.json")
  )
}

function Run-GenerateRound0Batch {
  $args = @("--batch-size", [string]$BatchSize, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  if ($RefreshExistingOnly) { $args += "--refresh-existing-only" }
  Invoke-Prompt2Round0Stage -StageName "GenerateRound0Batch" -ExtraArgs $args
  Complete-Stage -Stage "GenerateRound0Batch" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_generation_manifest.csv"),
    (Join-Path $OutRoot "round0\round0_branch_audit.csv"),
    (Join-Path $OutRoot "round0\round0_action_audit.csv"),
    (Join-Path $OutRoot "round0\round0_kpi_audit.csv"),
    (Join-Path $OutRoot "round0\round0_fallback_audit.csv"),
    (Join-Path $OutRoot "round0\round0_failures.csv"),
    (Join-Path $OutRoot "round0\round0_batch_report.json")
  )
}

function Run-BuildRound0Dataset {
  $args = @("--round", $Round)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "BuildRound0Dataset" -ExtraArgs $args
  Complete-Stage -Stage "BuildRound0Dataset" -OutputPaths @(
    (Join-Path $OutRoot "round0_dataset\$Round`_dataset_manifest.csv"),
    (Join-Path $OutRoot "round0_dataset\$Round`_dataset_report.json")
  )
}

function Run-AuditRound0Dataset {
  Invoke-Prompt2Round0Stage -StageName "AuditRound0Dataset" -ExtraArgs @("--round", $Round)
  Complete-Stage -Stage "AuditRound0Dataset" -OutputPaths @(
    (Join-Path $OutRoot "round0_dataset\$Round`_dataset_audit_report.json")
  )
}

function Run-EvaluateRound0DataGate {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound0DataGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound0DataGate" -OutputPaths @(
    (Join-Path $OutRoot "round0_dataset\round0_data_gate.json")
  )
}

function Run-EvaluateActionEffectTrainingReadiness {
  Invoke-Prompt2Round0Stage -StageName "EvaluateActionEffectTrainingReadiness" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateActionEffectTrainingReadiness" -OutputPaths @(
    (Join-Path $OutRoot "round0_dataset\action_effect_training_readiness_gate.json"),
    (Join-Path $OutRoot "round0_dataset\action_effect_training_inventory.csv"),
    (Join-Path $OutRoot "round0_dataset\action_effect_label_support.csv")
  )
}

function Run-PlanRound1 {
  Invoke-Prompt2Round0Stage -StageName "PlanRound1" -ExtraArgs @("--target-effective-candidates", [string]$TargetEffectiveCandidates, "--seed", [string]$Seed)
  Complete-Stage -Stage "PlanRound1" -OutputPaths @((Join-Path $OutRoot "round0\round1_candidate_pool.csv"), (Join-Path $OutRoot "round0\paired_manifest_round1.csv"), (Join-Path $OutRoot "round0\round1_plan_report.json"))
}

function Run-AuditRound1Manifest {
  Invoke-Prompt2Round0Stage -StageName "AuditRound1Manifest" -ExtraArgs @()
  Complete-Stage -Stage "AuditRound1Manifest" -OutputPaths @((Join-Path $OutRoot "round0\round1_manifest_audit_report.json"))
}

function Run-ApproveRound1Manifest {
  $args = @()
  if ($AcknowledgeRound1Manifest -or $AcknowledgeRound0Manifest) { $args += "--acknowledge-round0-manifest" }
  Invoke-Prompt2Round0Stage -StageName "ApproveRound1Manifest" -ExtraArgs $args
  Complete-Stage -Stage "ApproveRound1Manifest" -OutputPaths @((Join-Path $OutRoot "round0\round1_manifest_approval_lock.json"))
}

function Run-GenerateRound1Pilot {
  $args = @("--max-candidates", [string]$MaxCandidates, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "GenerateRound1Pilot" -ExtraArgs $args
  Complete-Stage -Stage "GenerateRound1Pilot" -OutputPaths @((Join-Path $OutRoot "round0\round1_pilot_generation_manifest.csv"), (Join-Path $OutRoot "round0\round1_pilot_branch_audit.csv"), (Join-Path $OutRoot "round0\round1_pilot_action_audit.csv"), (Join-Path $OutRoot "round0\round1_pilot_kpi_audit.csv"), (Join-Path $OutRoot "round0\round1_pilot_fallback_audit.csv"), (Join-Path $OutRoot "round0\round1_pilot_failures.csv"), (Join-Path $OutRoot "round0\round1_pilot_report.json"))
}

function Run-EvaluateRound1Pilot {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound1Pilot" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound1Pilot" -OutputPaths @((Join-Path $OutRoot "round0\round1_pilot_gate.json"))
}

function Run-GenerateRound1Batch {
  $args = @("--batch-size", [string]$BatchSize, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "GenerateRound1Batch" -ExtraArgs $args
  Complete-Stage -Stage "GenerateRound1Batch" -OutputPaths @((Join-Path $OutRoot "round0\round1_generation_manifest.csv"), (Join-Path $OutRoot "round0\round1_branch_audit.csv"), (Join-Path $OutRoot "round0\round1_action_audit.csv"), (Join-Path $OutRoot "round0\round1_kpi_audit.csv"), (Join-Path $OutRoot "round0\round1_fallback_audit.csv"), (Join-Path $OutRoot "round0\round1_failures.csv"), (Join-Path $OutRoot "round0\round1_report.json"))
}

function Run-BuildRound1Dataset {
  $args = @("--round", "round1")
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "BuildRound1Dataset" -ExtraArgs $args
  Complete-Stage -Stage "BuildRound1Dataset" -OutputPaths @((Join-Path $OutRoot "round0_dataset\round1_dataset_manifest.csv"), (Join-Path $OutRoot "round0_dataset\round1_dataset_report.json"))
}

function Run-AuditRound1Dataset {
  Invoke-Prompt2Round0Stage -StageName "AuditRound1Dataset" -ExtraArgs @()
  Complete-Stage -Stage "AuditRound1Dataset" -OutputPaths @((Join-Path $OutRoot "round0_dataset\round1_dataset_audit_report.json"))
}

function Run-EvaluateRound1DataGate {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound1DataGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound1DataGate" -OutputPaths @((Join-Path $OutRoot "round0_dataset\round1_data_gate.json"))
}

function Run-EvaluateRound1 {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound1" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound1" -OutputPaths @((Join-Path $OutRoot "round0\round1_active_learning_report.json"))
}

function Run-GenerateRound1 { Run-GenerateRound1Pilot }

function Run-PlanRound2 {
  Invoke-Prompt2Round0Stage -StageName "PlanRound2" -ExtraArgs @("--target-effective-candidates", [string]$TargetEffectiveCandidates, "--seed", [string]$Seed)
  Complete-Stage -Stage "PlanRound2" -OutputPaths @((Join-Path $OutRoot "round0\round2_hard_negative_pool.csv"), (Join-Path $OutRoot "round0\paired_manifest_round2.csv"), (Join-Path $OutRoot "round0\round2_plan_report.json"))
}

function Run-AuditRound2Manifest {
  Invoke-Prompt2Round0Stage -StageName "AuditRound2Manifest" -ExtraArgs @()
  Complete-Stage -Stage "AuditRound2Manifest" -OutputPaths @((Join-Path $OutRoot "round0\round2_manifest_audit_report.json"))
}

function Run-ApproveRound2Manifest {
  $args = @()
  if ($AcknowledgeRound2Manifest -or $AcknowledgeRound0Manifest) { $args += "--acknowledge-round0-manifest" }
  Invoke-Prompt2Round0Stage -StageName "ApproveRound2Manifest" -ExtraArgs $args
  Complete-Stage -Stage "ApproveRound2Manifest" -OutputPaths @((Join-Path $OutRoot "round0\round2_manifest_approval_lock.json"))
}

function Run-GenerateRound2Pilot {
  $args = @("--max-candidates", [string]$MaxCandidates, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "GenerateRound2Pilot" -ExtraArgs $args
  Complete-Stage -Stage "GenerateRound2Pilot" -OutputPaths @((Join-Path $OutRoot "round0\round2_pilot_generation_manifest.csv"), (Join-Path $OutRoot "round0\round2_pilot_branch_audit.csv"), (Join-Path $OutRoot "round0\round2_pilot_action_audit.csv"), (Join-Path $OutRoot "round0\round2_pilot_kpi_audit.csv"), (Join-Path $OutRoot "round0\round2_pilot_fallback_audit.csv"), (Join-Path $OutRoot "round0\round2_pilot_failures.csv"), (Join-Path $OutRoot "round0\round2_pilot_report.json"))
}

function Run-EvaluateRound2Pilot {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound2Pilot" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound2Pilot" -OutputPaths @((Join-Path $OutRoot "round0\round2_pilot_gate.json"))
}

function Run-GenerateRound2Batch {
  $args = @("--batch-size", [string]$BatchSize, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "GenerateRound2Batch" -ExtraArgs $args
  Complete-Stage -Stage "GenerateRound2Batch" -OutputPaths @((Join-Path $OutRoot "round0\round2_generation_manifest.csv"), (Join-Path $OutRoot "round0\round2_branch_audit.csv"), (Join-Path $OutRoot "round0\round2_action_audit.csv"), (Join-Path $OutRoot "round0\round2_kpi_audit.csv"), (Join-Path $OutRoot "round0\round2_fallback_audit.csv"), (Join-Path $OutRoot "round0\round2_failures.csv"), (Join-Path $OutRoot "round0\round2_report.json"))
}

function Run-BuildRound2Dataset {
  $args = @("--round", "round2")
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt2Round0Stage -StageName "BuildRound2Dataset" -ExtraArgs $args
  Complete-Stage -Stage "BuildRound2Dataset" -OutputPaths @((Join-Path $OutRoot "round0_dataset\round2_dataset_manifest.csv"), (Join-Path $OutRoot "round0_dataset\round2_dataset_report.json"))
}

function Run-AuditRound2Dataset {
  Invoke-Prompt2Round0Stage -StageName "AuditRound2Dataset" -ExtraArgs @()
  Complete-Stage -Stage "AuditRound2Dataset" -OutputPaths @((Join-Path $OutRoot "round0_dataset\round2_dataset_audit_report.json"))
}

function Run-EvaluateRound2DataGate {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound2DataGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound2DataGate" -OutputPaths @((Join-Path $OutRoot "round0_dataset\round2_data_gate.json"))
}

function Run-EvaluateRound2 {
  Invoke-Prompt2Round0Stage -StageName "EvaluateRound2" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateRound2" -OutputPaths @((Join-Path $OutRoot "round0\round2_hard_negative_report.json"))
}

function Run-GenerateRound2 { Run-GenerateRound2Pilot }

function Run-AuditPrompt3Entry {
  Invoke-Prompt3Stage -StageName "AuditPrompt3Entry" -ExtraArgs @()
  Complete-Stage -Stage "AuditPrompt3Entry" -OutputPaths @(
    (Join-Path $OutRoot "prompt3\prompt3_current_truth.json"),
    (Join-Path $OutRoot "prompt3\prompt3_dependency_matrix.csv")
  )
}

function Run-EvaluatePrompt3EntryGate {
  Invoke-Prompt3Stage -StageName "EvaluatePrompt3EntryGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluatePrompt3EntryGate" -OutputPaths @(
    (Join-Path $OutRoot "prompt3\prompt3_current_truth.json"),
    (Join-Path $OutRoot "prompt3\prompt3_dependency_matrix.csv"),
    (Join-Path $OutRoot "prompt3\prompt3_entry_gate.json")
  )
}

function Run-BuildActionEffectDataset {
  $args = @("--include-rounds", $IncludeRounds)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "BuildActionEffectDataset" -ExtraArgs $args
  Complete-Stage -Stage "BuildActionEffectDataset" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_dataset\action_effect_dataset_manifest.csv"),
    (Join-Path $OutRoot "action_effect_dataset\action_effect_dataset_report.json")
  )
}

function Run-AuditActionEffectDataset {
  Invoke-Prompt3Stage -StageName "AuditActionEffectDataset" -ExtraArgs @()
  Complete-Stage -Stage "AuditActionEffectDataset" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_dataset\action_effect_dataset_audit_report.json")
  )
}

function Run-EvaluateActionEffectDatasetGate {
  Invoke-Prompt3Stage -StageName "EvaluateActionEffectDatasetGate" -ExtraArgs @()
  $gateName = $(if ($Smoke) { "action_effect_dataset_smoke_gate.json" } else { "action_effect_dataset_gate.json" })
  Complete-Stage -Stage "EvaluateActionEffectDatasetGate" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_dataset\$gateName")
  )
}

function Run-TrainActionEffectBaselineModels {
  Invoke-Prompt3Stage -StageName "TrainActionEffectBaselineModels" -ExtraArgs @("--max-samples", [string]$MaxSamples)
  $report = $(if ($Smoke) { "baseline_smoke_model_report.json" } else { "baseline_model_report.json" })
  $model = $(if ($Smoke) { "baseline_smoke_model.npz" } else { "baseline_model.npz" })
  Complete-Stage -Stage "TrainActionEffectBaselineModels" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$model"),
    (Join-Path $OutRoot "action_effect_models\$report")
  )
}

function Run-TrainActionEffectEnsemble {
  $args = @(
    "--max-samples", [string]$MaxSamples,
    "--epochs", [string]$Epochs,
    "--ensemble-size", [string]$EnsembleSize
  )
  if (-not [string]::IsNullOrWhiteSpace($Seeds)) { $args += @("--seeds", $Seeds) }
  Invoke-Prompt3Stage -StageName "TrainActionEffectEnsemble" -ExtraArgs $args
  $report = $(if ($Smoke) { "action_effect_ensemble_smoke_report.json" } else { "action_effect_ensemble_report.json" })
  $model = $(if ($Smoke) { "action_effect_ensemble_smoke.npz" } else { "action_effect_ensemble.npz" })
  $metrics = $(if ($Smoke) { "action_effect_ensemble_smoke_metrics.csv" } else { "action_effect_ensemble_metrics.csv" })
  Complete-Stage -Stage "TrainActionEffectEnsemble" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$model"),
    (Join-Path $OutRoot "action_effect_models\$metrics"),
    (Join-Path $OutRoot "action_effect_models\$report")
  )
}

function Run-EvaluateActionEffectModelGate {
  Invoke-Prompt3Stage -StageName "EvaluateActionEffectModelGate" -ExtraArgs @()
  $gate = $(if ($Smoke) { "action_effect_model_smoke_gate.json" } else { "action_effect_model_gate.json" })
  Complete-Stage -Stage "EvaluateActionEffectModelGate" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$gate")
  )
}

function Run-CalibrateDevelopmentUncertainty {
  Invoke-Prompt3Stage -StageName "CalibrateDevelopmentUncertainty" -ExtraArgs @()
  $report = $(if ($Smoke) { "uncertainty_smoke_calibration_report.json" } else { "uncertainty_calibration_report.json" })
  Complete-Stage -Stage "CalibrateDevelopmentUncertainty" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$report")
  )
}

function Run-EvaluateUncertaintyGate {
  Invoke-Prompt3Stage -StageName "EvaluateUncertaintyGate" -ExtraArgs @()
  $gate = $(if ($Smoke) { "uncertainty_smoke_gate.json" } else { "uncertainty_gate.json" })
  Complete-Stage -Stage "EvaluateUncertaintyGate" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$gate")
  )
}

function Run-TrainOODModel {
  Invoke-Prompt3Stage -StageName "TrainOODModel" -ExtraArgs @()
  $report = $(if ($Smoke) { "ood_smoke_model_report.json" } else { "ood_model_report.json" })
  Complete-Stage -Stage "TrainOODModel" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$report")
  )
}

function Run-EvaluateOODGate {
  Invoke-Prompt3Stage -StageName "EvaluateOODGate" -ExtraArgs @()
  $gate = $(if ($Smoke) { "ood_smoke_gate.json" } else { "ood_gate.json" })
  Complete-Stage -Stage "EvaluateOODGate" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$gate")
  )
}

function Run-TrainSafetyClassifier {
  Invoke-Prompt3Stage -StageName "TrainSafetyClassifier" -ExtraArgs @()
  $report = $(if ($Smoke) { "safety_classifier_smoke_report.json" } else { "safety_classifier_report.json" })
  Complete-Stage -Stage "TrainSafetyClassifier" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$report")
  )
}

function Run-EvaluateSafetyClassifierGate {
  Invoke-Prompt3Stage -StageName "EvaluateSafetyClassifierGate" -ExtraArgs @()
  $gate = $(if ($Smoke) { "safety_classifier_smoke_gate.json" } else { "safety_classifier_gate.json" })
  Complete-Stage -Stage "EvaluateSafetyClassifierGate" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$gate")
  )
}

function Run-TrainFallbackSelector {
  Invoke-Prompt3Stage -StageName "TrainFallbackSelector" -ExtraArgs @()
  $report = $(if ($Smoke) { "fallback_selector_smoke_report.json" } else { "fallback_selector_report.json" })
  Complete-Stage -Stage "TrainFallbackSelector" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$report")
  )
}

function Run-EvaluatePrompt3ModelGate {
  Invoke-Prompt3Stage -StageName "EvaluatePrompt3ModelGate" -ExtraArgs @()
  $gate = $(if ($Smoke) { "prompt3_model_smoke_gate.json" } else { "prompt3_model_gate.json" })
  Complete-Stage -Stage "EvaluatePrompt3ModelGate" -OutputPaths @(
    (Join-Path $OutRoot "action_effect_models\$gate")
  )
}

function Run-BuildPFVFirstDualFallbackMPC {
  Invoke-Prompt3Stage -StageName "BuildPFVFirstDualFallbackMPC" -ExtraArgs @()
  $contract = $(if ($Smoke) { "mpc_smoke_contract.json" } else { "mpc_contract_lock.json" })
  Complete-Stage -Stage "BuildPFVFirstDualFallbackMPC" -OutputPaths @(
    (Join-Path $OutRoot "mpc\$contract")
  )
}

function Run-AuditMPCContract {
  Invoke-Prompt3Stage -StageName "AuditMPCContract" -ExtraArgs @()
  $audit = $(if ($Smoke) { "mpc_smoke_contract_audit.json" } else { "mpc_contract_audit.json" })
  Complete-Stage -Stage "AuditMPCContract" -OutputPaths @(
    (Join-Path $OutRoot "mpc\$audit")
  )
}

function Run-RunMPCUnitSmoke {
  Invoke-Prompt3Stage -StageName "RunMPCUnitSmoke" -ExtraArgs @("--max-cases", [string]$MaxCases)
  Complete-Stage -Stage "RunMPCUnitSmoke" -OutputPaths @(
    (Join-Path $OutRoot "mpc\mpc_unit_smoke_audit.csv"),
    (Join-Path $OutRoot "mpc\mpc_unit_smoke_report.json")
  )
}

function Run-EvaluateMPCUnitGate {
  Invoke-Prompt3Stage -StageName "EvaluateMPCUnitGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateMPCUnitGate" -OutputPaths @(
    (Join-Path $OutRoot "mpc\mpc_unit_smoke_gate.json")
  )
}

function Run-RunMPCShadowSmoke {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "RunMPCShadowSmoke" -ExtraArgs $args
  Complete-Stage -Stage "RunMPCShadowSmoke" -OutputPaths @(
    (Join-Path $OutRoot "mpc_shadow\mpc_shadow_smoke_audit.csv"),
    (Join-Path $OutRoot "mpc_shadow\mpc_shadow_smoke_report.json")
  )
}

function Run-RunMPCShadowDevelopment {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "RunMPCShadowDevelopment" -ExtraArgs $args
  Complete-Stage -Stage "RunMPCShadowDevelopment" -OutputPaths @(
    (Join-Path $OutRoot "mpc_shadow\mpc_shadow_smoke_audit.csv"),
    (Join-Path $OutRoot "mpc_shadow\mpc_shadow_smoke_report.json")
  )
}

function Run-EvaluateMPCShadowGate {
  Invoke-Prompt3Stage -StageName "EvaluateMPCShadowGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateMPCShadowGate" -OutputPaths @(
    (Join-Path $OutRoot "mpc_shadow\mpc_shadow_smoke_gate.json")
  )
}

function Run-RunMPCClosedLoopSmoke {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "RunMPCClosedLoopSmoke" -ExtraArgs $args
  Complete-Stage -Stage "RunMPCClosedLoopSmoke" -OutputPaths @(
    (Join-Path $OutRoot "mpc\mpc_closed_loop_smoke_report.json")
  )
}

function Run-EvaluateMPCClosedLoopSmokeGate {
  Invoke-Prompt3Stage -StageName "EvaluateMPCClosedLoopSmokeGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateMPCClosedLoopSmokeGate" -OutputPaths @(
    (Join-Path $OutRoot "mpc\mpc_closed_loop_smoke_gate.json")
  )
}

function Run-AuditAuthoritativeClosedLoopReadiness {
  Invoke-Prompt3Stage -StageName "AuditAuthoritativeClosedLoopReadiness" -ExtraArgs @()
  Complete-Stage -Stage "AuditAuthoritativeClosedLoopReadiness" -OutputPaths @(
    (Join-Path $OutRoot "authoritative_closed_loop\authoritative_closed_loop_readiness.json")
  )
}

function Run-RunAuthoritativeClosedLoopDev {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "RunAuthoritativeClosedLoopDev" -ExtraArgs $args
  Complete-Stage -Stage "RunAuthoritativeClosedLoopDev" -OutputPaths @(
    (Join-Path $OutRoot "authoritative_closed_loop\authoritative_closed_loop_dev_report.json"),
    (Join-Path $OutRoot "authoritative_closed_loop\authoritative_closed_loop_dev_decisions.csv"),
    (Join-Path $OutRoot "authoritative_closed_loop\authoritative_closed_loop_dev_action_audit.csv"),
    (Join-Path $OutRoot "authoritative_closed_loop\authoritative_closed_loop_dev_event_policy_results.csv")
  )
}

function Run-EvaluateAuthoritativeClosedLoopDevGate {
  Invoke-Prompt3Stage -StageName "EvaluateAuthoritativeClosedLoopDevGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateAuthoritativeClosedLoopDevGate" -OutputPaths @(
    (Join-Path $OutRoot "authoritative_closed_loop\authoritative_closed_loop_dev_gate.json")
  )
}

function Run-RunPairedClosedLoopDev {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "RunPairedClosedLoopDev" -ExtraArgs $args
  Complete-Stage -Stage "RunPairedClosedLoopDev" -OutputPaths @(
    (Join-Path $OutRoot "authoritative_closed_loop\paired_closed_loop_dev_report.json"),
    (Join-Path $OutRoot "authoritative_closed_loop\paired_closed_loop_dev_manifest.csv"),
    (Join-Path $OutRoot "authoritative_closed_loop\paired_closed_loop_dev_event_policy_results.csv")
  )
}

function Run-EvaluatePairedClosedLoopDevGate {
  Invoke-Prompt3Stage -StageName "EvaluatePairedClosedLoopDevGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluatePairedClosedLoopDevGate" -OutputPaths @(
    (Join-Path $OutRoot "authoritative_closed_loop\paired_closed_loop_dev_gate.json")
  )
}

function Run-BuildEvaluationEventSplits {
  Invoke-Prompt3Stage -StageName "BuildEvaluationEventSplits" -ExtraArgs @()
  Complete-Stage -Stage "BuildEvaluationEventSplits" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\evaluation_event_splits.csv"),
    (Join-Path $OutRoot "formal_evaluation\formal_blind_core_matrix.csv"),
    (Join-Path $OutRoot "formal_evaluation\evaluation_event_split_report.json")
  )
}

function Run-AuditEvaluationEventSplits {
  Invoke-Prompt3Stage -StageName "AuditEvaluationEventSplits" -ExtraArgs @()
  Complete-Stage -Stage "AuditEvaluationEventSplits" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\evaluation_event_split_audit.json")
  )
}

function Run-CalibrationA {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "CalibrationA" -ExtraArgs $args
  Complete-Stage -Stage "CalibrationA" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\calibration_a_run_manifest.json"),
    (Join-Path $OutRoot "formal_evaluation\calibration_a_event_policy_results.csv"),
    (Join-Path $OutRoot "formal_evaluation\calibration_a_action_audit.csv"),
    (Join-Path $OutRoot "formal_evaluation\calibration_a_timeseries.parquet")
  )
}

function Run-EvaluateCalibrationAGate {
  Invoke-Prompt3Stage -StageName "EvaluateCalibrationAGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateCalibrationAGate" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\calibration_a_gate.json")
  )
}

function Run-LockedValidationB {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "LockedValidationB" -ExtraArgs $args
  Complete-Stage -Stage "LockedValidationB" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\locked_validation_b_run_manifest.json"),
    (Join-Path $OutRoot "formal_evaluation\locked_validation_b_event_policy_results.csv"),
    (Join-Path $OutRoot "formal_evaluation\locked_validation_b_action_audit.csv"),
    (Join-Path $OutRoot "formal_evaluation\locked_validation_b_timeseries.parquet")
  )
}

function Run-EvaluateLockedValidationBGate {
  Invoke-Prompt3Stage -StageName "EvaluateLockedValidationBGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateLockedValidationBGate" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\locked_validation_b_gate.json")
  )
}

function Run-PolicyLock {
  Invoke-Prompt3Stage -StageName "PolicyLock" -ExtraArgs @()
  Complete-Stage -Stage "PolicyLock" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\policy_lock.json")
  )
}

function Run-AuditPolicyLock {
  Invoke-Prompt3Stage -StageName "AuditPolicyLock" -ExtraArgs @()
  Complete-Stage -Stage "AuditPolicyLock" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\policy_lock_audit.json")
  )
}

function Run-FormalBlind {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3Stage -StageName "FormalBlind" -ExtraArgs $args
  Complete-Stage -Stage "FormalBlind" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\formal_run_manifest.json"),
    (Join-Path $OutRoot "formal_evaluation\formal_event_policy_results.csv"),
    (Join-Path $OutRoot "formal_evaluation\formal_action_audit.csv"),
    (Join-Path $OutRoot "formal_evaluation\formal_timeseries.parquet")
  )
}

function Run-BuildFormalPairedComparison {
  Invoke-Prompt3Stage -StageName "BuildFormalPairedComparison" -ExtraArgs @()
  Complete-Stage -Stage "BuildFormalPairedComparison" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\formal_paired_comparison.csv"),
    (Join-Path $OutRoot "formal_evaluation\formal_paired_comparison_report.json"),
    (Join-Path $OutRoot "formal_evaluation\formal_statistical_tests.json")
  )
}

function Run-EvaluateFormalPerformanceGate {
  Invoke-Prompt3Stage -StageName "EvaluateFormalPerformanceGate" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateFormalPerformanceGate" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\formal_performance_gate.json")
  )
}

function Run-ExportFormalPaperTables {
  Invoke-Prompt3Stage -StageName "ExportFormalPaperTables" -ExtraArgs @()
  Complete-Stage -Stage "ExportFormalPaperTables" -OutputPaths @(
    (Join-Path $OutRoot "formal_evaluation\formal_summary_table_mean.csv"),
    (Join-Path $OutRoot "formal_evaluation\formal_summary_table_mean.md"),
    (Join-Path $OutRoot "formal_evaluation\formal_summary_table_median.csv"),
    (Join-Path $OutRoot "formal_evaluation\formal_summary_table_median.md"),
    (Join-Path $OutRoot "formal_evaluation\formal_action_audit.csv"),
    (Join-Path $OutRoot "formal_evaluation\formal_run_manifest.json"),
    (Join-Path $OutRoot "formal_evaluation\formal_table_export_report.json")
  )
}

function Invoke-Prompt3V31Stage {
  param(
    [string]$StageName,
    [string[]]$ExtraArgs
  )
  $args = @("--stage", $StageName, "--config", $Config) + $ExtraArgs
  if ($Smoke) { $args += "--smoke" }
  if ($ContractDryRun) { $args += "--contract-dry-run" }
  Invoke-PythonStage -Stage $StageName -Script "scripts\202_prompt3_v31.py" -Arguments $args
}

function Run-DiagnoseFormalFailuresV31 {
  Invoke-Prompt3V31Stage -StageName "DiagnoseFormalFailuresV31" -ExtraArgs @("--max-events", [string]$MaxEvents)
  Complete-Stage -Stage "DiagnoseFormalFailuresV31" -OutputPaths @(
    (Join-Path $OutRootV31 "diagnostics\v3_formal_failure_events.csv"),
    (Join-Path $OutRootV31 "diagnostics\v3_formal_failure_decisions.csv"),
    (Join-Path $OutRootV31 "diagnostics\v3_failure_type_summary.csv"),
    (Join-Path $OutRootV31 "diagnostics\v3_formal_failure_report.json")
  )
}

function Run-PlanRound3HardNegativesV31 {
  Invoke-Prompt3V31Stage -StageName "PlanRound3HardNegativesV31" -ExtraArgs @("--target-samples", [string]$TargetRound3Samples, "--seed", [string]$Seed)
  Complete-Stage -Stage "PlanRound3HardNegativesV31" -OutputPaths @(
    (Join-Path $OutRootV31 "round3\round3_hard_negative_plan.csv"),
    (Join-Path $OutRootV31 "round3\round3_hard_negative_support.csv"),
    (Join-Path $OutRootV31 "round3\round3_hard_negative_plan_report.json")
  )
}

function Run-GenerateRound3HardNegativesV31 {
  $args = @("--max-samples", [string]$MaxSamples)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V31Stage -StageName "GenerateRound3HardNegativesV31" -ExtraArgs $args
  $manifest = $(if ($Smoke) { "round3_generation_smoke_manifest.csv" } else { "round3_generation_manifest.csv" })
  $pending = $(if ($Smoke) { "round3_generation_smoke_pending.csv" } else { "round3_generation_pending.csv" })
  $report = $(if ($Smoke) { "round3_generation_smoke_report.json" } else { "round3_generation_report.json" })
  Complete-Stage -Stage "GenerateRound3HardNegativesV31" -OutputPaths @(
    (Join-Path $OutRootV31 "round3\$manifest"),
    (Join-Path $OutRootV31 "round3\$pending"),
    (Join-Path $OutRootV31 "round3\$report")
  )
}

function Run-BuildRound3DatasetV31 {
  Invoke-Prompt3V31Stage -StageName "BuildRound3DatasetV31" -ExtraArgs @()
  $manifest = $(if ($Smoke) { "round3_dataset_smoke_manifest.csv" } else { "round3_dataset_manifest.csv" })
  $report = $(if ($Smoke) { "round3_dataset_smoke_report.json" } else { "round3_dataset_report.json" })
  Complete-Stage -Stage "BuildRound3DatasetV31" -OutputPaths @(
    (Join-Path $OutRootV31 "round3_dataset\$manifest"),
    (Join-Path $OutRootV31 "round3_dataset\$report")
  )
}

function Run-AuditRound3DatasetV31 {
  Invoke-Prompt3V31Stage -StageName "AuditRound3DatasetV31" -ExtraArgs @()
  $audit = $(if ($Smoke) { "round3_dataset_smoke_audit.json" } else { "round3_dataset_audit.json" })
  Complete-Stage -Stage "AuditRound3DatasetV31" -OutputPaths @(
    (Join-Path $OutRootV31 "round3_dataset\$audit")
  )
}

function Run-TrainActionEffectV31 {
  Invoke-Prompt3V31Stage -StageName "TrainActionEffectV31" -ExtraArgs @("--max-samples", [string]$MaxSamples, "--epochs", [string]$Epochs, "--ensemble-size", [string]$EnsembleSize)
  $model = $(if ($Smoke) { "action_effect_v31_smoke_model.json" } else { "action_effect_v31_model.json" })
  $metrics = $(if ($Smoke) { "action_effect_v31_smoke_metrics.csv" } else { "action_effect_v31_metrics.csv" })
  $report = $(if ($Smoke) { "action_effect_v31_smoke_report.json" } else { "action_effect_v31_report.json" })
  Complete-Stage -Stage "TrainActionEffectV31" -OutputPaths @(
    (Join-Path $OutRootV31 "action_effect_models\$model"),
    (Join-Path $OutRootV31 "action_effect_models\$metrics"),
    (Join-Path $OutRootV31 "action_effect_models\$report")
  )
}

function Run-CalibrateUncertaintyV31 {
  Invoke-Prompt3V31Stage -StageName "CalibrateUncertaintyV31" -ExtraArgs @()
  $report = $(if ($Smoke) { "uncertainty_v31_smoke_report.json" } else { "uncertainty_v31_report.json" })
  Complete-Stage -Stage "CalibrateUncertaintyV31" -OutputPaths @(
    (Join-Path $OutRootV31 "action_effect_models\$report")
  )
}

function Run-TrainOODSafetyFallbackV31 {
  Invoke-Prompt3V31Stage -StageName "TrainOODSafetyFallbackV31" -ExtraArgs @()
  $report = $(if ($Smoke) { "ood_safety_fallback_v31_smoke_report.json" } else { "ood_safety_fallback_v31_report.json" })
  Complete-Stage -Stage "TrainOODSafetyFallbackV31" -OutputPaths @(
    (Join-Path $OutRootV31 "action_effect_models\$report")
  )
}

function Run-EvaluateModelGateV31 {
  Invoke-Prompt3V31Stage -StageName "EvaluateModelGateV31" -ExtraArgs @()
  $gate = $(if ($Smoke) { "model_gate_v31_smoke.json" } else { "model_gate_v31.json" })
  Complete-Stage -Stage "EvaluateModelGateV31" -OutputPaths @(
    (Join-Path $OutRootV31 "action_effect_models\$gate")
  )
}

function Run-RunClosedLoopDevV31 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V31Stage -StageName "RunClosedLoopDevV31" -ExtraArgs $args
  Complete-Stage -Stage "RunClosedLoopDevV31" -OutputPaths @(
    (Join-Path $OutRootV31 "authoritative_closed_loop\closed_loop_dev_v31_report.json"),
    (Join-Path $OutRootV31 "authoritative_closed_loop\closed_loop_dev_v31_decisions.csv"),
    (Join-Path $OutRootV31 "authoritative_closed_loop\closed_loop_dev_v31_action_audit.csv")
  )
}

function Run-BuildEvaluationRainfallAssetsV31 {
  Invoke-Prompt3V31Stage -StageName "BuildEvaluationRainfallAssetsV31" -ExtraArgs @()
  Complete-Stage -Stage "BuildEvaluationRainfallAssetsV31" -OutputPaths @(
    (Join-Path $OutRootV31 "rainfall_assets\rainfall_asset_inventory_v31.csv"),
    (Join-Path $OutRootV31 "rainfall_assets\rainfall_asset_generation_report_v31.json")
  )
}

function Run-BuildEvaluationSplitsV31 {
  Invoke-Prompt3V31Stage -StageName "BuildEvaluationSplitsV31" -ExtraArgs @()
  Complete-Stage -Stage "BuildEvaluationSplitsV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\evaluation_event_splits_v31.csv"),
    (Join-Path $OutRootV31 "formal_evaluation\evaluation_event_exclusions_v31.csv"),
    (Join-Path $OutRootV31 "formal_evaluation\evaluation_event_split_report_v31.json")
  )
}

function Run-AuditEvaluationSplitsV31 {
  Invoke-Prompt3V31Stage -StageName "AuditEvaluationSplitsV31" -ExtraArgs @()
  Complete-Stage -Stage "AuditEvaluationSplitsV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\evaluation_event_split_audit_v31.json")
  )
}

function Run-CalibrationAV31 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V31Stage -StageName "CalibrationAV31" -ExtraArgs $args
  $manifest = $(if ($ContractDryRun) { "calibration_a_v31_contract_dry_run_manifest.json" } else { "calibration_a_v31_run_manifest.json" })
  Complete-Stage -Stage "CalibrationAV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\$manifest")
  )
}

function Run-LockedValidationBV31 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V31Stage -StageName "LockedValidationBV31" -ExtraArgs $args
  $manifest = $(if ($ContractDryRun) { "locked_validation_b_v31_contract_dry_run_manifest.json" } else { "locked_validation_b_v31_run_manifest.json" })
  Complete-Stage -Stage "LockedValidationBV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\$manifest")
  )
}

function Run-PolicyLockV31 {
  Invoke-Prompt3V31Stage -StageName "PolicyLockV31" -ExtraArgs @()
  Complete-Stage -Stage "PolicyLockV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\policy_lock_v31.json")
  )
}

function Run-AuditPolicyLockV31 {
  Invoke-Prompt3V31Stage -StageName "AuditPolicyLockV31" -ExtraArgs @()
  Complete-Stage -Stage "AuditPolicyLockV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\policy_lock_audit_v31.json")
  )
}

function Run-FormalBlindV31 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V31Stage -StageName "FormalBlindV31" -ExtraArgs $args
  $manifest = $(if ($ContractDryRun) { "formal_blind_v31_contract_dry_run_manifest.json" } else { "formal_blind_v31_run_manifest.json" })
  Complete-Stage -Stage "FormalBlindV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\$manifest")
  )
}

function Run-BuildFormalComparisonV31 {
  Invoke-Prompt3V31Stage -StageName "BuildFormalComparisonV31" -ExtraArgs @()
  Complete-Stage -Stage "BuildFormalComparisonV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\formal_paired_comparison.csv"),
    (Join-Path $OutRootV31 "formal_evaluation\formal_paired_comparison_report.json"),
    (Join-Path $OutRootV31 "formal_evaluation\formal_statistical_tests.json")
  )
}

function Run-EvaluateFormalPerformanceV31 {
  Invoke-Prompt3V31Stage -StageName "EvaluateFormalPerformanceV31" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateFormalPerformanceV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\formal_performance_gate_v31.json")
  )
}

function Run-ExportFormalTablesV31 {
  Invoke-Prompt3V31Stage -StageName "ExportFormalTablesV31" -ExtraArgs @()
  Complete-Stage -Stage "ExportFormalTablesV31" -OutputPaths @(
    (Join-Path $OutRootV31 "formal_evaluation\formal_summary_table_mean.csv"),
    (Join-Path $OutRootV31 "formal_evaluation\formal_summary_table_median.csv"),
    (Join-Path $OutRootV31 "formal_evaluation\formal_table_export_report.json")
  )
}

function Invoke-Prompt3V32Stage {
  param(
    [string]$StageName,
    [string[]]$ExtraArgs = @()
  )
  $args = @("--stage", $StageName, "--config", $Config) + $ExtraArgs
  if ($Smoke) { $args += "--smoke" }
  Invoke-PythonStage -Stage $StageName -Script "scripts\203_prompt3_v32.py" -Arguments $args
}

function Run-DiagnoseFormalFailuresV32 {
  Invoke-Prompt3V32Stage -StageName "DiagnoseFormalFailuresV32" -ExtraArgs @("--max-events", [string]$MaxEvents)
  Complete-Stage -Stage "DiagnoseFormalFailuresV32" -OutputPaths @(
    (Join-Path $OutRootV32 "diagnostics\v32_v31_formal_failure_events.csv"),
    (Join-Path $OutRootV32 "diagnostics\v32_v31_formal_failure_decisions.csv"),
    (Join-Path $OutRootV32 "diagnostics\v32_failure_type_summary.csv"),
    (Join-Path $OutRootV32 "diagnostics\v32_formal_failure_report.json")
  )
}

function Run-PlanRound4HardNegativesV32 {
  Invoke-Prompt3V32Stage -StageName "PlanRound4HardNegativesV32" -ExtraArgs @("--target-samples", [string]$TargetRound3Samples, "--seed", [string]$Seed)
  Complete-Stage -Stage "PlanRound4HardNegativesV32" -OutputPaths @(
    (Join-Path $OutRootV32 "round4\round4_hard_negative_plan.csv"),
    (Join-Path $OutRootV32 "round4\round4_hard_negative_support.csv"),
    (Join-Path $OutRootV32 "round4\round4_hard_negative_plan_report.json")
  )
}

function Run-GenerateRound4HardNegativesV32 {
  $args = @("--max-samples", [string]$MaxSamples)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V32Stage -StageName "GenerateRound4HardNegativesV32" -ExtraArgs $args
  $manifest = $(if ($Smoke) { "round4_generation_smoke_manifest.csv" } else { "round4_generation_manifest.csv" })
  $pending = $(if ($Smoke) { "round4_generation_smoke_pending.csv" } else { "round4_generation_pending.csv" })
  $report = $(if ($Smoke) { "round4_generation_smoke_report.json" } else { "round4_generation_report.json" })
  Complete-Stage -Stage "GenerateRound4HardNegativesV32" -OutputPaths @(
    (Join-Path $OutRootV32 "round4\$manifest"),
    (Join-Path $OutRootV32 "round4\$pending"),
    (Join-Path $OutRootV32 "round4\$report")
  )
}

function Run-BuildRound4DatasetV32 {
  Invoke-Prompt3V32Stage -StageName "BuildRound4DatasetV32" -ExtraArgs @()
  $manifest = $(if ($Smoke) { "round4_dataset_smoke_manifest.csv" } else { "round4_dataset_manifest.csv" })
  $report = $(if ($Smoke) { "round4_dataset_smoke_report.json" } else { "round4_dataset_report.json" })
  Complete-Stage -Stage "BuildRound4DatasetV32" -OutputPaths @(
    (Join-Path $OutRootV32 "round4_dataset\$manifest"),
    (Join-Path $OutRootV32 "round4_dataset\$report")
  )
}

function Run-AuditRound4DatasetV32 {
  Invoke-Prompt3V32Stage -StageName "AuditRound4DatasetV32" -ExtraArgs @()
  $audit = $(if ($Smoke) { "round4_dataset_smoke_audit.json" } else { "round4_dataset_audit.json" })
  Complete-Stage -Stage "AuditRound4DatasetV32" -OutputPaths @((Join-Path $OutRootV32 "round4_dataset\$audit"))
}

function Run-TrainActionEffectV32 {
  Invoke-Prompt3V32Stage -StageName "TrainActionEffectV32" -ExtraArgs @("--max-samples", [string]$MaxSamples, "--epochs", [string]$Epochs, "--ensemble-size", [string]$EnsembleSize)
  $report = $(if ($Smoke) { "action_effect_v32_smoke_report.json" } else { "action_effect_v32_report.json" })
  Complete-Stage -Stage "TrainActionEffectV32" -OutputPaths @((Join-Path $OutRootV32 "action_effect_models\$report"))
}

function Run-CalibrateUncertaintyV32 {
  Invoke-Prompt3V32Stage -StageName "CalibrateUncertaintyV32" -ExtraArgs @()
  $report = $(if ($Smoke) { "uncertainty_v32_smoke_report.json" } else { "uncertainty_v32_report.json" })
  Complete-Stage -Stage "CalibrateUncertaintyV32" -OutputPaths @((Join-Path $OutRootV32 "action_effect_models\$report"))
}

function Run-TrainOODSafetyFallbackV32 {
  Invoke-Prompt3V32Stage -StageName "TrainOODSafetyFallbackV32" -ExtraArgs @()
  $report = $(if ($Smoke) { "ood_safety_fallback_v32_smoke_report.json" } else { "ood_safety_fallback_v32_report.json" })
  Complete-Stage -Stage "TrainOODSafetyFallbackV32" -OutputPaths @((Join-Path $OutRootV32 "action_effect_models\$report"))
}

function Run-EvaluateModelGateV32 {
  Invoke-Prompt3V32Stage -StageName "EvaluateModelGateV32" -ExtraArgs @()
  $gate = $(if ($Smoke) { "model_gate_v32_smoke.json" } else { "model_gate_v32.json" })
  Complete-Stage -Stage "EvaluateModelGateV32" -OutputPaths @((Join-Path $OutRootV32 "action_effect_models\$gate"))
}

function Run-RunClosedLoopDevV32 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V32Stage -StageName "RunClosedLoopDevV32" -ExtraArgs $args
  Complete-Stage -Stage "RunClosedLoopDevV32" -OutputPaths @(
    (Join-Path $OutRootV32 "authoritative_closed_loop\closed_loop_dev_v32_report.json"),
    (Join-Path $OutRootV32 "authoritative_closed_loop\closed_loop_dev_v32_decisions.csv"),
    (Join-Path $OutRootV32 "authoritative_closed_loop\closed_loop_dev_v32_pfv_budget_audit.csv")
  )
}

function Run-BuildEvaluationRainfallAssetsV32 {
  Invoke-Prompt3V32Stage -StageName "BuildEvaluationRainfallAssetsV32" -ExtraArgs @()
  Complete-Stage -Stage "BuildEvaluationRainfallAssetsV32" -OutputPaths @(
    (Join-Path $OutRootV32 "rainfall_assets\rainfall_asset_inventory_v32.csv"),
    (Join-Path $OutRootV32 "rainfall_assets\rainfall_asset_generation_report_v32.json")
  )
}

function Run-BuildEvaluationSplitsV32 {
  Invoke-Prompt3V32Stage -StageName "BuildEvaluationSplitsV32" -ExtraArgs @()
  Complete-Stage -Stage "BuildEvaluationSplitsV32" -OutputPaths @(
    (Join-Path $OutRootV32 "formal_evaluation\evaluation_event_splits_v32.csv"),
    (Join-Path $OutRootV32 "formal_evaluation\evaluation_event_exclusions_v32.csv"),
    (Join-Path $OutRootV32 "formal_evaluation\evaluation_event_split_report_v32.json")
  )
}

function Run-AuditEvaluationSplitsV32 {
  Invoke-Prompt3V32Stage -StageName "AuditEvaluationSplitsV32" -ExtraArgs @()
  Complete-Stage -Stage "AuditEvaluationSplitsV32" -OutputPaths @((Join-Path $OutRootV32 "formal_evaluation\evaluation_event_split_audit_v32.json"))
}

function Run-CalibrationAV32 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  if ($ContractDryRun) { $args += "--contract-dry-run" }
  Invoke-Prompt3V32Stage -StageName "CalibrationAV32" -ExtraArgs $args
  $manifest = $(if ($ContractDryRun) { "calibration_a_v32_contract_dry_run_manifest.json" } else { "calibration_a_v32_run_manifest.json" })
  Complete-Stage -Stage "CalibrationAV32" -OutputPaths @((Join-Path $OutRootV32 "formal_evaluation\$manifest"))
}

function Run-LockedValidationBV32 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  if ($ContractDryRun) { $args += "--contract-dry-run" }
  Invoke-Prompt3V32Stage -StageName "LockedValidationBV32" -ExtraArgs $args
  $manifest = $(if ($ContractDryRun) { "locked_validation_b_v32_contract_dry_run_manifest.json" } else { "locked_validation_b_v32_run_manifest.json" })
  Complete-Stage -Stage "LockedValidationBV32" -OutputPaths @((Join-Path $OutRootV32 "formal_evaluation\$manifest"))
}

function Run-PolicyLockV32 {
  Invoke-Prompt3V32Stage -StageName "PolicyLockV32" -ExtraArgs @()
  Complete-Stage -Stage "PolicyLockV32" -OutputPaths @((Join-Path $OutRootV32 "formal_evaluation\policy_lock_v32.json"))
}

function Run-AuditPolicyLockV32 {
  Invoke-Prompt3V32Stage -StageName "AuditPolicyLockV32" -ExtraArgs @()
  Complete-Stage -Stage "AuditPolicyLockV32" -OutputPaths @((Join-Path $OutRootV32 "formal_evaluation\policy_lock_audit_v32.json"))
}

function Run-FormalBlindV32 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  if ($ContractDryRun) { $args += "--contract-dry-run" }
  Invoke-Prompt3V32Stage -StageName "FormalBlindV32" -ExtraArgs $args
  $manifest = $(if ($ContractDryRun) { "formal_blind_v32_contract_dry_run_manifest.json" } else { "formal_blind_v32_run_manifest.json" })
  Complete-Stage -Stage "FormalBlindV32" -OutputPaths @((Join-Path $OutRootV32 "formal_evaluation\$manifest"))
}

function Run-RunFormalExtraBaselinesV32 {
  $args = @("--max-events", [string]$MaxEvents, "--workers", [string]$Workers)
  if ($Resume) { $args += "--resume" }
  Invoke-Prompt3V32Stage -StageName "RunFormalExtraBaselinesV32" -ExtraArgs $args
  Complete-Stage -Stage "RunFormalExtraBaselinesV32" -OutputPaths @(
    (Join-Path $OutRootV32 "formal_evaluation\formal_blind_v32_extra_baseline_run_manifest.json"),
    (Join-Path $OutRootV32 "formal_evaluation\formal_blind_v32_extra_baseline_event_policy_results.csv")
  )
}

function Run-BuildFormalComparisonV32 {
  Invoke-Prompt3V32Stage -StageName "BuildFormalComparisonV32" -ExtraArgs @()
  Complete-Stage -Stage "BuildFormalComparisonV32" -OutputPaths @(
    (Join-Path $OutRootV32 "formal_evaluation\formal_paired_comparison.csv"),
    (Join-Path $OutRootV32 "formal_evaluation\formal_paired_comparison_report.json"),
    (Join-Path $OutRootV32 "formal_evaluation\formal_statistical_tests.json"),
    (Join-Path $OutRootV32 "formal_evaluation\formal_auto_rbc_efd_comparison_status_v32.json")
  )
}

function Run-EvaluateFormalPerformanceV32 {
  Invoke-Prompt3V32Stage -StageName "EvaluateFormalPerformanceV32" -ExtraArgs @()
  Complete-Stage -Stage "EvaluateFormalPerformanceV32" -OutputPaths @((Join-Path $OutRootV32 "formal_evaluation\formal_performance_gate.json"))
}

function Run-ExportFormalTablesV32 {
  Invoke-Prompt3V32Stage -StageName "ExportFormalTablesV32" -ExtraArgs @()
  Complete-Stage -Stage "ExportFormalTablesV32" -OutputPaths @(
    (Join-Path $OutRootV32 "formal_evaluation\formal_summary_table_mean.csv"),
    (Join-Path $OutRootV32 "formal_evaluation\formal_summary_table_median.csv"),
    (Join-Path $OutRootV32 "formal_evaluation\formal_table_export_report.json")
  )
}

function Invoke-Prompt3V33Stage {
  param(
    [string]$StageName,
    [string[]]$ExtraArgs = @()
  )
  $args = @("--stage", $StageName, "--config", $Config) + $ExtraArgs
  Invoke-PythonStage -Stage $StageName -Script "scripts\204_prompt3_v33.py" -Arguments $args
}

function Run-DiagnoseV32RegressionV33 { Invoke-Prompt3V33Stage -StageName "DiagnoseV32RegressionV33" -ExtraArgs @(); Complete-Stage -Stage "DiagnoseV32RegressionV33" -OutputPaths @((Join-Path $OutRootV33 "diagnostics\v33_module_root_cause.json")) }
function Run-RunModuleAblationV33 { Invoke-Prompt3V33Stage -StageName "RunModuleAblationV33" -ExtraArgs @("--max-events", [string]$(if ($MaxEvents) { $MaxEvents } else { 12 }), "--workers", [string]$Workers); Complete-Stage -Stage "RunModuleAblationV33" -OutputPaths @((Join-Path $OutRootV33 "ablation\v33_module_ablation.csv"), (Join-Path $OutRootV33 "ablation\v33_module_ablation_report.json")) }
function Run-PlanRound5HardNegativesV33 { Invoke-Prompt3V33Stage -StageName "PlanRound5HardNegativesV33" -ExtraArgs @("--target-samples", [string]$(if ($TargetRound3Samples) { $TargetRound3Samples } else { 400 }), "--seed", [string]$Seed); Complete-Stage -Stage "PlanRound5HardNegativesV33" -OutputPaths @((Join-Path $OutRootV33 "round5\round5_hard_negative_plan.csv")) }
function Run-GenerateRound5HardNegativesV33 { $args=@("--max-samples",[string]$MaxSamples); if($Smoke){$args+="--smoke"}; if($Resume){$args+="--resume"}; Invoke-Prompt3V33Stage -StageName "GenerateRound5HardNegativesV33" -ExtraArgs $args; Complete-Stage -Stage "GenerateRound5HardNegativesV33" -OutputPaths @((Join-Path $OutRootV33 "round5\round5_generation_report.json")) }
function Run-BuildRound5DatasetV33 { $args=@(); if($Smoke){$args+="--smoke"}; Invoke-Prompt3V33Stage -StageName "BuildRound5DatasetV33" -ExtraArgs $args; Complete-Stage -Stage "BuildRound5DatasetV33" -OutputPaths @((Join-Path $OutRootV33 "round5_dataset\round5_dataset_report.json")) }
function Run-AuditRound5DatasetV33 { $args=@(); if($Smoke){$args+="--smoke"}; Invoke-Prompt3V33Stage -StageName "AuditRound5DatasetV33" -ExtraArgs $args; Complete-Stage -Stage "AuditRound5DatasetV33" -OutputPaths @((Join-Path $OutRootV33 "round5_dataset\round5_dataset_audit.json")) }
function Run-TrainActionEffectV33 { $args=@("--epochs",[string]$Epochs,"--ensemble-size",[string]$EnsembleSize,"--max-samples",[string]$MaxSamples); if($Smoke){$args+="--smoke"}; Invoke-Prompt3V33Stage -StageName "TrainActionEffectV33" -ExtraArgs $args; Complete-Stage -Stage "TrainActionEffectV33" -OutputPaths @((Join-Path $OutRootV33 "action_effect_models\action_effect_v33_report.json")) }
function Run-CalibrateUncertaintyV33 { $args=@(); if($Smoke){$args+="--smoke"}; Invoke-Prompt3V33Stage -StageName "CalibrateUncertaintyV33" -ExtraArgs $args; Complete-Stage -Stage "CalibrateUncertaintyV33" -OutputPaths @((Join-Path $OutRootV33 "action_effect_models\uncertainty_v33_report.json")) }
function Run-TrainOODSafetyFallbackV33 { $args=@(); if($Smoke){$args+="--smoke"}; Invoke-Prompt3V33Stage -StageName "TrainOODSafetyFallbackV33" -ExtraArgs $args; Complete-Stage -Stage "TrainOODSafetyFallbackV33" -OutputPaths @((Join-Path $OutRootV33 "action_effect_models\ood_safety_fallback_v33_report.json")) }
function Run-EvaluateModelGateV33 { $args=@(); if($Smoke){$args+="--smoke"}; Invoke-Prompt3V33Stage -StageName "EvaluateModelGateV33" -ExtraArgs $args; Complete-Stage -Stage "EvaluateModelGateV33" -OutputPaths @((Join-Path $OutRootV33 "action_effect_models\model_gate_v33.json")) }
function Run-RunClosedLoopDevV33 { Invoke-Prompt3V33Stage -StageName "RunClosedLoopDevV33" -ExtraArgs @("--max-events",[string]$(if($MaxEvents){$MaxEvents}else{3}),"--workers",[string]$Workers); Complete-Stage -Stage "RunClosedLoopDevV33" -OutputPaths @((Join-Path $OutRootV33 "authoritative_closed_loop\closed_loop_dev_v33_report.json")) }
function Run-BuildEvaluationRainfallAssetsV33 { Invoke-Prompt3V33Stage -StageName "BuildEvaluationRainfallAssetsV33" -ExtraArgs @(); Complete-Stage -Stage "BuildEvaluationRainfallAssetsV33" -OutputPaths @((Join-Path $OutRootV33 "rainfall_assets\rainfall_asset_inventory_v33.csv")) }
function Run-BuildEvaluationSplitsV33 { Invoke-Prompt3V33Stage -StageName "BuildEvaluationSplitsV33" -ExtraArgs @(); Complete-Stage -Stage "BuildEvaluationSplitsV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\evaluation_event_splits_v33.csv")) }
function Run-AuditEvaluationSplitsV33 { Invoke-Prompt3V33Stage -StageName "AuditEvaluationSplitsV33" -ExtraArgs @(); Complete-Stage -Stage "AuditEvaluationSplitsV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\evaluation_event_split_audit_v33.json")) }
function Run-CalibrationAV33 { Invoke-Prompt3V33Stage -StageName "CalibrationAV33" -ExtraArgs @("--max-events",[string]$MaxEvents,"--workers",[string]$Workers); Complete-Stage -Stage "CalibrationAV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\calibration_a_v33_run_manifest.json")) }
function Run-LockedValidationBV33 { Invoke-Prompt3V33Stage -StageName "LockedValidationBV33" -ExtraArgs @("--max-events",[string]$MaxEvents,"--workers",[string]$Workers); Complete-Stage -Stage "LockedValidationBV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\locked_validation_b_v33_run_manifest.json")) }
function Run-PolicyLockV33 { Invoke-Prompt3V33Stage -StageName "PolicyLockV33" -ExtraArgs @(); Complete-Stage -Stage "PolicyLockV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\policy_lock_v33.json")) }
function Run-AuditPolicyLockV33 { Invoke-Prompt3V33Stage -StageName "AuditPolicyLockV33" -ExtraArgs @(); Complete-Stage -Stage "AuditPolicyLockV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\policy_lock_audit_v33.json")) }
function Run-FormalBlindV33 { Invoke-Prompt3V33Stage -StageName "FormalBlindV33" -ExtraArgs @("--max-events",[string]$MaxEvents,"--workers",[string]$Workers); Complete-Stage -Stage "FormalBlindV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\formal_blind_v33_run_manifest.json")) }
function Run-RunFormalExtraBaselinesV33 { Invoke-Prompt3V33Stage -StageName "RunFormalExtraBaselinesV33" -ExtraArgs @("--max-events",[string]$MaxEvents,"--workers",[string]$Workers); Complete-Stage -Stage "RunFormalExtraBaselinesV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\formal_blind_v33_extra_baseline_run_manifest.json")) }
function Run-BuildFormalComparisonV33 { Invoke-Prompt3V33Stage -StageName "BuildFormalComparisonV33" -ExtraArgs @(); Complete-Stage -Stage "BuildFormalComparisonV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\formal_paired_comparison_report.json")) }
function Run-EvaluateFormalPerformanceV33 { Invoke-Prompt3V33Stage -StageName "EvaluateFormalPerformanceV33" -ExtraArgs @(); Complete-Stage -Stage "EvaluateFormalPerformanceV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\formal_performance_gate.json")) }
function Run-ExportFormalTablesV33 { Invoke-Prompt3V33Stage -StageName "ExportFormalTablesV33" -ExtraArgs @(); Complete-Stage -Stage "ExportFormalTablesV33" -OutputPaths @((Join-Path $OutRootV33 "formal_evaluation\formal_table_export_report.json")) }

function Run-EvaluatePrompt3Completion {
  Invoke-Prompt3Stage -StageName "EvaluatePrompt3Completion" -ExtraArgs @()
  Complete-Stage -Stage "EvaluatePrompt3Completion" -OutputPaths @(
    (Join-Path $OutRoot "prompt3\prompt3_completion_gate.json")
  )
}

function Run-EvaluatePrompt2Completion {
  $gateDir = Join-Path $OutRoot "gates"
  Invoke-PythonStage -Stage "EvaluatePrompt2Completion" -Script "scripts\145_evaluate_prompt2_completion.py" -Arguments @(
    "--config", $Config,
    "--out-root", $OutRoot
  )
  Complete-Stage -Stage "EvaluatePrompt2Completion" -OutputPaths @(
    (Join-Path $gateDir "project6_prompt2_completion_gate.json")
  )
}

function Run-EvaluatePrompt2GATReadiness {
  $gateDir = Join-Path $OutRoot "gates"
  Invoke-PythonStage -Stage "EvaluatePrompt2GATReadiness" -Script "scripts\148_evaluate_prompt2_gat_readiness.py" -Arguments @(
    "--config", $Config,
    "--out-root", $OutRoot
  )
  Complete-Stage -Stage "EvaluatePrompt2GATReadiness" -OutputPaths @(
    (Join-Path $gateDir "project6_prompt2_gat_readiness_gate.json")
  )
}

function Run-ImportPrompt2Artifacts {
  Invoke-PythonStage -Stage "ImportPrompt2Artifacts" -Script "scripts\153_import_prompt2_artifacts.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "ImportPrompt2Artifacts" -OutputPaths @(
    (Join-Path $Root "docs\contracts\project6_prompt2_import_contract.json"),
    (Join-Path $OutRoot "contracts\prompt2_import_manifest.json"),
    (Join-Path $OutRoot "gates\prompt3a_entry_gate.json")
  )
}

function Run-FatalAudit {
  Assert-UpstreamCompletion -Stage "FatalAudit" -UpstreamStage "ImportPrompt2Artifacts"
  Validate-CompletionHashes -Stage "FatalAudit" -UpstreamStage "ImportPrompt2Artifacts"
  Invoke-PythonStage -Stage "FatalAudit" -Script "scripts\154_prompt3a_fatal_audit.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "FatalAudit" -OutputPaths @(
    (Join-Path $OutRoot "fatal_audit\fatal_audit_report.json"),
    (Join-Path $OutRoot "fatal_audit\engineering_development_gate.json"),
    (Join-Path $OutRoot "fatal_audit\formal_safety_readiness_gate.json"),
    (Join-Path $OutRoot "contracts\network_contract.json"),
    (Join-Path $OutRoot "contracts\contract_manifest.json")
  )
}

function Run-AuditReferencesFallbacks {
  Assert-UpstreamCompletion -Stage "AuditReferencesFallbacks" -UpstreamStage "FatalAudit"
  Validate-CompletionHashes -Stage "AuditReferencesFallbacks" -UpstreamStage "FatalAudit"
  Invoke-PythonStage -Stage "AuditReferencesFallbacks" -Script "scripts\166_audit_reference_roles.py" -Arguments @("--config", $Config, "--out-dir", (Join-Path $OutRoot "reference_roles"))
  Complete-Stage -Stage "AuditReferencesFallbacks" -OutputPaths @(
    (Join-Path $OutRoot "reference_roles\reference_roles_contract.json"),
    (Join-Path $OutRoot "reference_roles\reference_role_audit_report.json"),
    (Join-Path $OutRoot "reference_roles\no_control_reference_contract.json"),
    (Join-Path $OutRoot "reference_roles\online_benchmark_contract.json")
  )
}

function Run-RebuildContract {
  Assert-UpstreamCompletion -Stage "RebuildContract" -UpstreamStage "AuditReferencesFallbacks"
  Validate-CompletionHashes -Stage "RebuildContract" -UpstreamStage "AuditReferencesFallbacks"
  Assert-UpstreamMarkerContainsOutput -Stage "RebuildContract" -UpstreamStage "AuditReferencesFallbacks" -ExpectedOutputSuffix "reference_roles\reference_roles_contract.json"
  Invoke-PythonStage -Stage "RebuildContract" -Script "scripts\157_rebuild_project6_contract.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "RebuildContract" -OutputPaths @(
    (Join-Path $OutRoot "contracts\project6_prompt3a_contract_manifest.json")
  )
}

function Run-AuditNativeRules {
  Assert-UpstreamCompletion -Stage "AuditNativeRules" -UpstreamStage "RebuildContract"
  Validate-CompletionHashes -Stage "AuditNativeRules" -UpstreamStage "RebuildContract"
  Invoke-PythonStage -Stage "AuditNativeRules" -Script "scripts\155_audit_native_rules.py" -Arguments @("--config", $Config, "--out-dir", (Join-Path $OutRoot "native_rules"))
  Complete-Stage -Stage "AuditNativeRules" -OutputPaths @(
    (Join-Path $OutRoot "native_rules\native_rules_parsed.json"),
    (Join-Path $OutRoot "native_rules\native_rule_actions.csv"),
    (Join-Path $OutRoot "native_rules\native_rule_conditions.csv"),
    (Join-Path $OutRoot "native_rules\native_rule_conflicts.csv"),
    (Join-Path $OutRoot "native_rules\native_controlled_facilities.csv"),
    (Join-Path $OutRoot "native_rules\native_rule_audit_report.json")
  )
}

function Run-AuditFallbacks {
  Assert-UpstreamCompletion -Stage "AuditFallbacks" -UpstreamStage "AuditNativeRules"
  Validate-CompletionHashes -Stage "AuditFallbacks" -UpstreamStage "AuditNativeRules"
  Invoke-PythonStage -Stage "AuditFallbacks" -Script "scripts\156_audit_fallbacks.py" -Arguments @("--config", $Config, "--out-dir", (Join-Path $OutRoot "fallbacks"))
  Complete-Stage -Stage "AuditFallbacks" -OutputPaths @(
    (Join-Path $OutRoot "fallbacks\passive_fallback_contract.json"),
    (Join-Path $OutRoot "fallbacks\internal_fallback_contract.json"),
    (Join-Path $OutRoot "fallbacks\fallback_selection_contract.json"),
    (Join-Path $OutRoot "fallbacks\fallback_transition_tests.schema.csv"),
    (Join-Path $OutRoot "fallbacks\fallback_execution_audit_report.json")
  )
}

function Run-BuildEventCatalog {
  Assert-UpstreamCompletion -Stage "BuildEventCatalog" -UpstreamStage "BuildRainfallAssetIndex"
  Validate-CompletionHashes -Stage "BuildEventCatalog" -UpstreamStage "BuildRainfallAssetIndex"
  Invoke-PythonStage -Stage "BuildEventCatalog" -Script "scripts\158_build_event_catalog_v3.py" -Arguments @("--config", $Config, "--out-dir", (Join-Path $OutRoot "event_catalog"))
  Complete-Stage -Stage "BuildEventCatalog" -OutputPaths @(
    (Join-Path $OutRoot "event_catalog\event_catalog.csv"),
    (Join-Path $OutRoot "event_catalog\event_provenance_audit.json"),
    (Join-Path $OutRoot "event_catalog\event_near_duplicate_groups.csv"),
    (Join-Path $OutRoot "event_catalog\event_split_manifest.csv"),
    (Join-Path $OutRoot "event_catalog\event_split_leakage_audit.csv"),
    (Join-Path $OutRoot "event_catalog\gat_seen_event_manifest.csv"),
    (Join-Path $OutRoot "event_catalog\gat_independent_holdout_event_manifest.csv"),
    (Join-Path $OutRoot "event_catalog\unresolved_rainfall_events.csv")
  )
}

function Run-BuildRainfallAssetIndex {
  Assert-UpstreamCompletion -Stage "BuildRainfallAssetIndex" -UpstreamStage "AuditFallbacks"
  Validate-CompletionHashes -Stage "BuildRainfallAssetIndex" -UpstreamStage "AuditFallbacks"
  Invoke-PythonStage -Stage "BuildRainfallAssetIndex" -Script "scripts\167_build_rainfall_asset_index.py" -Arguments @("--config", $Config, "--out-dir", (Join-Path $OutRoot "rainfall_assets"))
  Complete-Stage -Stage "BuildRainfallAssetIndex" -OutputPaths @(
    (Join-Path $OutRoot "rainfall_assets\rainfall_asset_inventory.csv"),
    (Join-Path $OutRoot "rainfall_assets\rainfall_asset_duplicate_audit.csv"),
    (Join-Path $OutRoot "rainfall_assets\rainfall_asset_conflict_audit.csv"),
    (Join-Path $OutRoot "rainfall_assets\rainfall_asset_resolution_audit.csv"),
    (Join-Path $OutRoot "rainfall_assets\rainfall_asset_index_report.json")
  )
}

function Run-BuildFormalRainfallAssets {
  Invoke-PythonStage -Stage "BuildFormalRainfallAssets" -Script "scripts\203_build_formal_rainfall_assets.py" -Arguments @(
    "--config", $Config,
    "--out-dir", (Join-Path $Root "data\rainfall_library")
  )
  Complete-Stage -Stage "BuildFormalRainfallAssets" -OutputPaths @(
    (Join-Path $Root "data\rainfall_library\formal_rainfall_asset_manifest.csv"),
    (Join-Path $Root "data\rainfall_library\formal_rainfall_asset_report.json")
  )
}

function Run-PlanBaselineTrajectories {
  Assert-UpstreamCompletion -Stage "PlanBaselineTrajectories" -UpstreamStage "BuildEventCatalog"
  Validate-CompletionHashes -Stage "PlanBaselineTrajectories" -UpstreamStage "BuildEventCatalog"
  Invoke-PythonStage -Stage "PlanBaselineTrajectories" -Script "scripts\159_plan_baseline_trajectories.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "PlanBaselineTrajectories" -OutputPaths @(
    (Join-Path $OutRoot "baseline_trajectories\baseline_trajectory_plan.csv"),
    (Join-Path $OutRoot "baseline_trajectories\baseline_trajectory_plan_report.json"),
    (Join-Path $OutRoot "baseline_trajectories\trajectory_schema.json"),
    (Join-Path $OutRoot "baseline_trajectories\baseline_trajectory_exclusion_audit.csv")
  )
}

function Run-GenerateBaselineTrajectories {
  Assert-UpstreamCompletion -Stage "GenerateBaselineTrajectories" -UpstreamStage "PlanBaselineTrajectories"
  Validate-CompletionHashes -Stage "GenerateBaselineTrajectories" -UpstreamStage "PlanBaselineTrajectories"
  $args = @(
    "--config", $Config,
    "--max-events", [string]$MaxEvents,
    "--workers", [string]$Workers,
    "--tail-min", [string]$TailMin
  )
  if (-not [string]::IsNullOrWhiteSpace($PolicyFilter)) {
    $args += @("--policy-filter", $PolicyFilter)
  }
  if ($Resume) { $args += "--resume" }
  if ($SkipExisting) { $args += "--skip-existing" }
  if ($RefreshExistingOnly) { $args += "--refresh-existing-only" }
  Invoke-PythonStage -Stage "GenerateBaselineTrajectories" -Script "scripts\160_generate_baseline_trajectories.py" -Arguments $args
  Complete-Stage -Stage "GenerateBaselineTrajectories" -OutputPaths @(
    (Join-Path $OutRoot "baseline_trajectories\baseline_trajectory_manifest.csv"),
    (Join-Path $OutRoot "baseline_trajectories\baseline_recovery_audit.csv"),
    (Join-Path $OutRoot "baseline_trajectories\baseline_checkpoint_audit.csv"),
    (Join-Path $OutRoot "baseline_trajectories\baseline_trajectory_failures.csv"),
    (Join-Path $OutRoot "baseline_trajectories\baseline_trajectory_status.csv"),
    (Join-Path $OutRoot "baseline_trajectories\trajectory_quality_report.json")
  )
}

function Run-BuildCheckpointCatalog {
  Assert-UpstreamCompletion -Stage "BuildCheckpointCatalog" -UpstreamStage "GenerateBaselineTrajectories"
  Validate-CompletionHashes -Stage "BuildCheckpointCatalog" -UpstreamStage "GenerateBaselineTrajectories"
  Invoke-PythonStage -Stage "BuildCheckpointCatalog" -Script "scripts\161_build_checkpoint_catalog_v3.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "BuildCheckpointCatalog" -OutputPaths @(
    (Join-Path $OutRoot "checkpoint_catalog\checkpoint_catalog.csv"),
    (Join-Path $OutRoot "checkpoint_catalog\checkpoint_state_hash_audit.csv"),
    (Join-Path $OutRoot "checkpoint_catalog\checkpoint_near_duplicate_audit.csv"),
    (Join-Path $OutRoot "checkpoint_catalog\checkpoint_split_audit.csv"),
    (Join-Path $OutRoot "checkpoint_catalog\checkpoint_catalog_report.json")
  )
}

function Run-BuildCoverageContract {
  Assert-UpstreamCompletion -Stage "BuildCoverageContract" -UpstreamStage "StateCloneTest"
  Validate-CompletionHashes -Stage "BuildCoverageContract" -UpstreamStage "StateCloneTest"
  Invoke-PythonStage -Stage "BuildCoverageContract" -Script "scripts\162_build_coverage_contract_v3.py" -Arguments @("--config", $Config, "--out-dir", (Join-Path $OutRoot "coverage"))
  Complete-Stage -Stage "BuildCoverageContract" -OutputPaths @(
    (Join-Path $OutRoot "coverage\coverage_cells_schema.csv"),
    (Join-Path $OutRoot "coverage\coverage_contract.json"),
    (Join-Path $OutRoot "coverage\prompt3a_coverage_gate.json")
  )
}

function Run-PlanRound0 {
  Assert-UpstreamCompletion -Stage "PlanRound0" -UpstreamStage "EvaluatePrompt2CheckpointSupportGate"
  Invoke-Prompt2Round0Stage -StageName "PlanRound0" -ExtraArgs @(
    "--target-effective-candidates", [string]$TargetEffectiveCandidates,
    "--reserve-candidates", [string]$ReserveCandidates,
    "--pressure-candidates", [string]$PressureCandidates,
    "--seed", [string]$Seed
  )
  Complete-Stage -Stage "PlanRound0" -OutputPaths @(
    (Join-Path $OutRoot "round0\paired_manifest_round0.csv"),
    (Join-Path $OutRoot "round0\checkpoint_coverage_round0.csv"),
    (Join-Path $OutRoot "round0\noop_candidates.csv"),
    (Join-Path $OutRoot "round0\duplicate_candidates.csv"),
    (Join-Path $OutRoot "round0\planned_facility_support_round0.csv"),
    (Join-Path $OutRoot "round0\planned_phase_support_round0.csv"),
    (Join-Path $OutRoot "round0\planned_concurrency_support_round0.csv"),
    (Join-Path $OutRoot "round0\planned_interaction_support_round0.csv"),
    (Join-Path $OutRoot "round0\structural_infeasible_candidates.csv"),
    (Join-Path $OutRoot "round0\round0_plan_report.json")
  )
}

function Run-DryRunRound0 {
  Assert-UpstreamCompletion -Stage "DryRunRound0" -UpstreamStage "PlanRound0"
  Validate-CompletionHashes -Stage "DryRunRound0" -UpstreamStage "PlanRound0"
  Invoke-PythonStage -Stage "DryRunRound0" -Script "scripts\164_dryrun_round0_v3.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "DryRunRound0" -OutputPaths @(
    (Join-Path $OutRoot "round0\round0_dryrun_manifest.csv"),
    (Join-Path $OutRoot "round0\round0_dryrun_branch_audit.csv"),
    (Join-Path $OutRoot "round0\round0_dryrun_action_audit.csv"),
    (Join-Path $OutRoot "round0\round0_dryrun_kpi_audit.csv"),
    (Join-Path $OutRoot "round0\round0_dryrun_fallback_audit.csv"),
    (Join-Path $OutRoot "round0\round0_dryrun_report.json")
  )
}

function Run-AuditCurrentTruth {
  Invoke-PythonStage -Stage "AuditCurrentTruth" -Script "scripts\168_audit_current_truth.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "AuditCurrentTruth" -OutputPaths @(
    (Join-Path $OutRoot "status\project6_current_truth_matrix.csv"),
    (Join-Path $OutRoot "status\project6_current_truth_report.json"),
    (Join-Path $OutRoot "gates\project6_recovery_gate.json")
  )
}

function Run-EvaluatePrompt3AEngineeringGate {
  Invoke-PythonStage -Stage "EvaluatePrompt3AEngineeringGate" -Script "scripts\169_evaluate_prompt3a_engineering_gate.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "EvaluatePrompt3AEngineeringGate" -OutputPaths @(
    (Join-Path $OutRoot "gates\project6_prompt3a_engineering_gate.json"),
    (Join-Path $OutRoot "status\project6_current_truth_matrix.csv"),
    (Join-Path $OutRoot "status\project6_current_truth_report.json")
  )
}

function Run-EvaluatePrompt3ARuntimeGate {
  Invoke-PythonStage -Stage "EvaluatePrompt3ARuntimeGate" -Script "scripts\170_evaluate_prompt3a_runtime_gate.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "EvaluatePrompt3ARuntimeGate" -OutputPaths @(
    (Join-Path $OutRoot "gates\project6_prompt3a_runtime_gate.json"),
    (Join-Path $OutRoot "status\project6_current_truth_matrix.csv"),
    (Join-Path $OutRoot "status\project6_current_truth_report.json")
  )
}

function Run-EvaluatePrompt3ACompletion {
  Invoke-PythonStage -Stage "EvaluatePrompt3ACompletion" -Script "scripts\165_evaluate_prompt3a_completion.py" -Arguments @("--config", $Config)
  Complete-Stage -Stage "EvaluatePrompt3ACompletion" -OutputPaths @(
    (Join-Path $OutRoot "gates\project6_prompt3a_completion_gate.json")
  )
}

try {
  Ensure-Directories
  $stage = Get-SelectedStage
  Write-Host "[Project6 PFV-first dual-fallback V3] step=$stage"
  if (-not ($ImplementedStages -contains $stage)) {
    Disable-Stage -Name $stage
  }
  switch ($stage) {
    "Status" { Run-Status }
    "Audit" { Run-Audit }
    "InitCoverageSchema" { Run-InitCoverageSchema }
    "RegisterGAT" { Run-RegisterGAT }
    "RecoverGATMetadata" { Run-RecoverGATMetadata }
    "InspectGATCheckpoints" { Run-InspectGATCheckpoints }
    "AuditGAT" { Run-AuditGAT }
    "PrepareStateFeatureContracts" { Run-PrepareStateFeatureContracts }
    "BuildStateFeatures" { Run-BuildStateFeatures }
    "RunGATForwardSmoke" { Run-GATForwardSmoke }
    "RunGATReconstructionAudit" { Run-GATReconstructionAudit }
    "SelectPrimaryGAT" { Run-SelectPrimaryGAT }
    "RunGATRobustnessAudit" { Run-GATRobustnessAudit }
    "AuditGATValidationProvenance" { Run-AuditGATValidationProvenance }
    "BuildGATIndependentValidationCatalog" { Run-BuildGATIndependentValidationCatalog }
    "GenerateGATIndependentHoldoutTrajectories" { Run-GenerateGATIndependentHoldoutTrajectories }
    "LockGATIndependentValidationManifest" { Run-LockGATIndependentValidationManifest }
    "EvaluateGATRobustnessGate" { Run-EvaluateGATRobustnessGate }
    "BuildStateInputManifest" { Run-BuildStateInputManifest }
    "StateCloneTest" { Run-StateCloneTest }
    "PrepareStateCloneCheckpoints" { Run-PrepareStateCloneCheckpoints }
    "EstimateStateCloneNumericalNoise" { Run-EstimateStateCloneNumericalNoise }
    "RunStateCloneEquivalence" { Run-RunStateCloneEquivalence }
    "EvaluateStateCloneGate" { Run-EvaluateStateCloneGate }
    "RunContinuousReplayDeterminismAudit" { Run-RunContinuousReplayDeterminismAudit }
    "EvaluateContinuousReplayDeterminismGate" { Run-EvaluateContinuousReplayDeterminismGate }
    "RunStateCloneDiagnosticMatrix" { Run-RunStateCloneDiagnosticMatrix }
    "RunSameStateReplayEquivalence" { Run-RunSameStateReplayEquivalence }
    "EvaluateHotstartCloneGate" { Run-EvaluateHotstartCloneGate }
    "EvaluateSameStateBranchGate" { Run-EvaluateSameStateBranchGate }
    "DiagnoseHotstartFirstDivergence" { Run-DiagnoseHotstartFirstDivergence }
    "AuditHotstartCompatibility" { Run-AuditHotstartCompatibility }
    "BuildCanonicalHotstartCache" { Run-BuildCanonicalHotstartCache }
    "RunHotstartSmoke" { Run-RunHotstartSmoke }
    "EvaluateHotstartSmokeGate" { Run-EvaluateHotstartSmokeGate }
    "RunHotstartFullValidation" { Run-RunHotstartFullValidation }
    "EvaluateHotstartFullGate" { Run-EvaluateHotstartFullGate }
    "CertifyHotstartCheckpoints" { Run-CertifyHotstartCheckpoints }
    "BenchmarkHotstartAcceleration" { Run-BenchmarkHotstartAcceleration }
    "EvaluateHotstartAccelerationReadiness" { Run-EvaluateHotstartAccelerationReadiness }
    "AuditRunoffCacheEligibility" { Run-AuditRunoffCacheEligibility }
    "BuildRainfallInterfaceCache" { Run-BuildRainfallInterfaceCache }
    "BuildRunoffInterfaceCache" { Run-BuildRunoffInterfaceCache }
    "AuditRunoffInterfaceEquivalence" { Run-AuditRunoffInterfaceEquivalence }
    "EvaluateRunoffCacheGate" { Run-EvaluateRunoffCacheGate }
    "BuildReferenceBranchCache" { Run-BuildReferenceBranchCache }
    "RunCandidatePrefilterAudit" { Run-RunCandidatePrefilterAudit }
    "BenchmarkReplayAcceleration" { Run-BenchmarkReplayAcceleration }
    "EvaluateReplayAccelerationGate" { Run-EvaluateReplayAccelerationGate }
    "AuditPrompt2Entry" { Run-AuditPrompt2Entry }
    "PlanPrompt2FitEventExpansion" { Run-PlanPrompt2FitEventExpansion }
    "AuditPrompt2FitEventExpansion" { Run-AuditPrompt2FitEventExpansion }
    "PlanPrompt2BaselineExpansion" { Run-PlanPrompt2BaselineExpansion }
    "GeneratePrompt2BaselineExpansion" { Run-GeneratePrompt2BaselineExpansion }
    "AuditPrompt2BaselineExpansion" { Run-AuditPrompt2BaselineExpansion }
    "BuildPrompt2ControlCheckpointCandidates" { Run-BuildPrompt2ControlCheckpointCandidates }
    "SelectPrompt2ControlCheckpoints" { Run-SelectPrompt2ControlCheckpoints }
    "AuditPrompt2ControlCheckpointSupport" { Run-AuditPrompt2ControlCheckpointSupport }
    "BuildPrompt2StateInputManifest" { Run-BuildPrompt2StateInputManifest }
    "BuildPrompt2StateFeatures" { Run-BuildPrompt2StateFeatures }
    "AuditPrompt2StateCoverage" { Run-AuditPrompt2StateCoverage }
    "EvaluatePrompt2CheckpointSupportGate" { Run-EvaluatePrompt2CheckpointSupportGate }
    "BuildControlAlignedCheckpointCatalog" { Run-BuildControlAlignedCheckpointCatalog }
    "AuditControlAlignedCheckpointCatalog" { Run-AuditControlAlignedCheckpointCatalog }
    "BuildRound0CoverageContract" { Run-BuildRound0CoverageContract }
    "AuditRound0Manifest" { Run-AuditRound0Manifest }
    "PlanRound0HydraulicDryRun" { Run-PlanRound0HydraulicDryRun }
    "RunRound0HydraulicDryRun" { Run-RunRound0HydraulicDryRun }
    "EvaluateRound0HydraulicDryRunGate" { Run-EvaluateRound0HydraulicDryRunGate }
    "ApproveRound0Manifest" { Run-ApproveRound0Manifest }
    "GenerateRound0Pilot" { Run-GenerateRound0Pilot }
    "EvaluateRound0Pilot" { Run-EvaluateRound0Pilot }
    "ReplanRound0Adaptive" { Run-ReplanRound0Adaptive }
    "GenerateRound0Batch" { Run-GenerateRound0Batch }
    "BuildRound0Dataset" { Run-BuildRound0Dataset }
    "AuditRound0Dataset" { Run-AuditRound0Dataset }
    "EvaluateRound0DataGate" { Run-EvaluateRound0DataGate }
    "EvaluateActionEffectTrainingReadiness" { Run-EvaluateActionEffectTrainingReadiness }
    "PlanRound1" { Run-PlanRound1 }
    "AuditRound1Manifest" { Run-AuditRound1Manifest }
    "ApproveRound1Manifest" { Run-ApproveRound1Manifest }
    "GenerateRound1" { Run-GenerateRound1 }
    "GenerateRound1Pilot" { Run-GenerateRound1Pilot }
    "EvaluateRound1Pilot" { Run-EvaluateRound1Pilot }
    "GenerateRound1Batch" { Run-GenerateRound1Batch }
    "BuildRound1Dataset" { Run-BuildRound1Dataset }
    "AuditRound1Dataset" { Run-AuditRound1Dataset }
    "EvaluateRound1DataGate" { Run-EvaluateRound1DataGate }
    "EvaluateRound1" { Run-EvaluateRound1 }
    "PlanRound2" { Run-PlanRound2 }
    "AuditRound2Manifest" { Run-AuditRound2Manifest }
    "ApproveRound2Manifest" { Run-ApproveRound2Manifest }
    "GenerateRound2" { Run-GenerateRound2 }
    "GenerateRound2Pilot" { Run-GenerateRound2Pilot }
    "EvaluateRound2Pilot" { Run-EvaluateRound2Pilot }
    "GenerateRound2Batch" { Run-GenerateRound2Batch }
    "BuildRound2Dataset" { Run-BuildRound2Dataset }
    "AuditRound2Dataset" { Run-AuditRound2Dataset }
    "EvaluateRound2DataGate" { Run-EvaluateRound2DataGate }
    "EvaluateRound2" { Run-EvaluateRound2 }
    "AuditPrompt3Entry" { Run-AuditPrompt3Entry }
    "EvaluatePrompt3EntryGate" { Run-EvaluatePrompt3EntryGate }
    "BuildActionEffectDataset" { Run-BuildActionEffectDataset }
    "AuditActionEffectDataset" { Run-AuditActionEffectDataset }
    "EvaluateActionEffectDatasetGate" { Run-EvaluateActionEffectDatasetGate }
    "TrainActionEffectBaselineModels" { Run-TrainActionEffectBaselineModels }
    "TrainActionEffectEnsemble" { Run-TrainActionEffectEnsemble }
    "EvaluateActionEffectModelGate" { Run-EvaluateActionEffectModelGate }
    "CalibrateDevelopmentUncertainty" { Run-CalibrateDevelopmentUncertainty }
    "EvaluateUncertaintyGate" { Run-EvaluateUncertaintyGate }
    "TrainOODModel" { Run-TrainOODModel }
    "EvaluateOODGate" { Run-EvaluateOODGate }
    "TrainSafetyClassifier" { Run-TrainSafetyClassifier }
    "EvaluateSafetyClassifierGate" { Run-EvaluateSafetyClassifierGate }
    "TrainFallbackSelector" { Run-TrainFallbackSelector }
    "EvaluatePrompt3ModelGate" { Run-EvaluatePrompt3ModelGate }
    "BuildPFVFirstDualFallbackMPC" { Run-BuildPFVFirstDualFallbackMPC }
    "AuditMPCContract" { Run-AuditMPCContract }
    "RunMPCUnitSmoke" { Run-RunMPCUnitSmoke }
    "EvaluateMPCUnitGate" { Run-EvaluateMPCUnitGate }
    "RunMPCShadowSmoke" { Run-RunMPCShadowSmoke }
    "RunMPCShadowDevelopment" { Run-RunMPCShadowDevelopment }
    "EvaluateMPCShadowGate" { Run-EvaluateMPCShadowGate }
    "RunMPCClosedLoopSmoke" { Run-RunMPCClosedLoopSmoke }
    "EvaluateMPCClosedLoopSmokeGate" { Run-EvaluateMPCClosedLoopSmokeGate }
    "AuditAuthoritativeClosedLoopReadiness" { Run-AuditAuthoritativeClosedLoopReadiness }
    "RunAuthoritativeClosedLoopDev" { Run-RunAuthoritativeClosedLoopDev }
    "EvaluateAuthoritativeClosedLoopDevGate" { Run-EvaluateAuthoritativeClosedLoopDevGate }
    "RunPairedClosedLoopDev" { Run-RunPairedClosedLoopDev }
    "EvaluatePairedClosedLoopDevGate" { Run-EvaluatePairedClosedLoopDevGate }
    "BuildEvaluationEventSplits" { Run-BuildEvaluationEventSplits }
    "AuditEvaluationEventSplits" { Run-AuditEvaluationEventSplits }
    "CalibrationA" { Run-CalibrationA }
    "EvaluateCalibrationAGate" { Run-EvaluateCalibrationAGate }
    "LockedValidationB" { Run-LockedValidationB }
    "EvaluateLockedValidationBGate" { Run-EvaluateLockedValidationBGate }
    "PolicyLock" { Run-PolicyLock }
    "AuditPolicyLock" { Run-AuditPolicyLock }
    "FormalBlind" { Run-FormalBlind }
    "BuildFormalPairedComparison" { Run-BuildFormalPairedComparison }
    "EvaluateFormalPerformanceGate" { Run-EvaluateFormalPerformanceGate }
    "ExportFormalPaperTables" { Run-ExportFormalPaperTables }
    "DiagnoseFormalFailuresV31" { Run-DiagnoseFormalFailuresV31 }
    "PlanRound3HardNegativesV31" { Run-PlanRound3HardNegativesV31 }
    "GenerateRound3HardNegativesV31" { Run-GenerateRound3HardNegativesV31 }
    "BuildRound3DatasetV31" { Run-BuildRound3DatasetV31 }
    "AuditRound3DatasetV31" { Run-AuditRound3DatasetV31 }
    "TrainActionEffectV31" { Run-TrainActionEffectV31 }
    "CalibrateUncertaintyV31" { Run-CalibrateUncertaintyV31 }
    "TrainOODSafetyFallbackV31" { Run-TrainOODSafetyFallbackV31 }
    "EvaluateModelGateV31" { Run-EvaluateModelGateV31 }
    "RunClosedLoopDevV31" { Run-RunClosedLoopDevV31 }
    "BuildEvaluationRainfallAssetsV31" { Run-BuildEvaluationRainfallAssetsV31 }
    "BuildEvaluationSplitsV31" { Run-BuildEvaluationSplitsV31 }
    "AuditEvaluationSplitsV31" { Run-AuditEvaluationSplitsV31 }
    "CalibrationAV31" { Run-CalibrationAV31 }
    "LockedValidationBV31" { Run-LockedValidationBV31 }
    "PolicyLockV31" { Run-PolicyLockV31 }
    "AuditPolicyLockV31" { Run-AuditPolicyLockV31 }
    "FormalBlindV31" { Run-FormalBlindV31 }
    "BuildFormalComparisonV31" { Run-BuildFormalComparisonV31 }
    "EvaluateFormalPerformanceV31" { Run-EvaluateFormalPerformanceV31 }
    "ExportFormalTablesV31" { Run-ExportFormalTablesV31 }
    "DiagnoseFormalFailuresV32" { Run-DiagnoseFormalFailuresV32 }
    "PlanRound4HardNegativesV32" { Run-PlanRound4HardNegativesV32 }
    "GenerateRound4HardNegativesV32" { Run-GenerateRound4HardNegativesV32 }
    "BuildRound4DatasetV32" { Run-BuildRound4DatasetV32 }
    "AuditRound4DatasetV32" { Run-AuditRound4DatasetV32 }
    "TrainActionEffectV32" { Run-TrainActionEffectV32 }
    "CalibrateUncertaintyV32" { Run-CalibrateUncertaintyV32 }
    "TrainOODSafetyFallbackV32" { Run-TrainOODSafetyFallbackV32 }
    "EvaluateModelGateV32" { Run-EvaluateModelGateV32 }
    "RunClosedLoopDevV32" { Run-RunClosedLoopDevV32 }
    "BuildEvaluationRainfallAssetsV32" { Run-BuildEvaluationRainfallAssetsV32 }
    "BuildEvaluationSplitsV32" { Run-BuildEvaluationSplitsV32 }
    "AuditEvaluationSplitsV32" { Run-AuditEvaluationSplitsV32 }
    "CalibrationAV32" { Run-CalibrationAV32 }
    "LockedValidationBV32" { Run-LockedValidationBV32 }
    "PolicyLockV32" { Run-PolicyLockV32 }
    "AuditPolicyLockV32" { Run-AuditPolicyLockV32 }
    "FormalBlindV32" { Run-FormalBlindV32 }
    "RunFormalExtraBaselinesV32" { Run-RunFormalExtraBaselinesV32 }
    "BuildFormalComparisonV32" { Run-BuildFormalComparisonV32 }
    "EvaluateFormalPerformanceV32" { Run-EvaluateFormalPerformanceV32 }
    "ExportFormalTablesV32" { Run-ExportFormalTablesV32 }
    "DiagnoseV32RegressionV33" { Run-DiagnoseV32RegressionV33 }
    "RunModuleAblationV33" { Run-RunModuleAblationV33 }
    "PlanRound5HardNegativesV33" { Run-PlanRound5HardNegativesV33 }
    "GenerateRound5HardNegativesV33" { Run-GenerateRound5HardNegativesV33 }
    "BuildRound5DatasetV33" { Run-BuildRound5DatasetV33 }
    "AuditRound5DatasetV33" { Run-AuditRound5DatasetV33 }
    "TrainActionEffectV33" { Run-TrainActionEffectV33 }
    "CalibrateUncertaintyV33" { Run-CalibrateUncertaintyV33 }
    "TrainOODSafetyFallbackV33" { Run-TrainOODSafetyFallbackV33 }
    "EvaluateModelGateV33" { Run-EvaluateModelGateV33 }
    "RunClosedLoopDevV33" { Run-RunClosedLoopDevV33 }
    "BuildEvaluationRainfallAssetsV33" { Run-BuildEvaluationRainfallAssetsV33 }
    "BuildEvaluationSplitsV33" { Run-BuildEvaluationSplitsV33 }
    "AuditEvaluationSplitsV33" { Run-AuditEvaluationSplitsV33 }
    "CalibrationAV33" { Run-CalibrationAV33 }
    "LockedValidationBV33" { Run-LockedValidationBV33 }
    "PolicyLockV33" { Run-PolicyLockV33 }
    "AuditPolicyLockV33" { Run-AuditPolicyLockV33 }
    "FormalBlindV33" { Run-FormalBlindV33 }
    "RunFormalExtraBaselinesV33" { Run-RunFormalExtraBaselinesV33 }
    "BuildFormalComparisonV33" { Run-BuildFormalComparisonV33 }
    "EvaluateFormalPerformanceV33" { Run-EvaluateFormalPerformanceV33 }
    "ExportFormalTablesV33" { Run-ExportFormalTablesV33 }
    "EvaluatePrompt3Completion" { Run-EvaluatePrompt3Completion }
    "EvaluatePrompt2Completion" { Run-EvaluatePrompt2Completion }
    "EvaluatePrompt2GATReadiness" { Run-EvaluatePrompt2GATReadiness }
    "ImportPrompt2Artifacts" { Run-ImportPrompt2Artifacts }
    "FatalAudit" { Run-FatalAudit }
    "AuditReferencesFallbacks" { Run-AuditReferencesFallbacks }
    "RebuildContract" { Run-RebuildContract }
    "AuditNativeRules" { Run-AuditNativeRules }
    "AuditFallbacks" { Run-AuditFallbacks }
    "BuildFormalRainfallAssets" { Run-BuildFormalRainfallAssets }
    "BuildRainfallAssetIndex" { Run-BuildRainfallAssetIndex }
    "BuildEventCatalog" { Run-BuildEventCatalog }
    "PlanBaselineTrajectories" { Run-PlanBaselineTrajectories }
    "GenerateBaselineTrajectories" { Run-GenerateBaselineTrajectories }
    "BuildCheckpointCatalog" { Run-BuildCheckpointCatalog }
    "BuildCoverageContract" { Run-BuildCoverageContract }
    "AuditCurrentTruth" { Run-AuditCurrentTruth }
    "EvaluatePrompt3AEngineeringGate" { Run-EvaluatePrompt3AEngineeringGate }
    "EvaluatePrompt3ARuntimeGate" { Run-EvaluatePrompt3ARuntimeGate }
    "PlanRound0" { Run-PlanRound0 }
    "DryRunRound0" { Run-DryRunRound0 }
    "EvaluatePrompt3ACompletion" { Run-EvaluatePrompt3ACompletion }
    default { Disable-Stage -Name $stage }
  }
} catch [DisabledStageError] {
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 2
} catch [BlockedStageError] {
  $s = $_.Exception.Stage
  Write-StageStatus -Stage $s -Status "blocked" -ExitCode 3 -FailureReason $_.Exception.Message -CompletionMarker $null
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 3
} catch [GateFailedError] {
  $s = $_.Exception.Stage
  Write-StageStatus -Stage $s -Status "failed_gate" -ExitCode 5 -FailureReason $_.Exception.Message -CompletionMarker $null
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 5
} catch [ContractMismatchError] {
  $s = $_.Exception.Stage
  Write-StageStatus -Stage $s -Status "contract_mismatch" -ExitCode 6 -FailureReason $_.Exception.Message -CompletionMarker $null
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 6
} catch [CliContractError] {
  $s = $_.Exception.Stage
  Write-StageStatus -Stage $s -Status "failed" -ExitCode 7 -FailureReason $_.Exception.Message -CompletionMarker $null
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 7
} catch [RuntimeStageError] {
  $s = $_.Exception.Stage
  Write-StageStatus -Stage $s -Status "failed" -ExitCode 4 -FailureReason $_.Exception.Message -CompletionMarker $null
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 4
} catch {
  Write-StageStatus -Stage "unknown" -Status "failed" -ExitCode 4 -FailureReason $_.Exception.Message -CompletionMarker $null
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 4
}
