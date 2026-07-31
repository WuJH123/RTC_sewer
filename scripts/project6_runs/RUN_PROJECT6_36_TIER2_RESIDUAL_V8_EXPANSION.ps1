[CmdletBinding()]
param(
  [switch]$PlanExpansion,
  [switch]$DryRunExpansion,
  [switch]$RunExpansion,
  [switch]$BuildExpansionDataset,
  [switch]$MergeExpandedDataset,
  [switch]$AuditExpandedDataset,
  [switch]$BuildDeploymentGateSplit,
  [switch]$UseDeploymentGateSplit,
  [switch]$TrainV8,
  [switch]$GateV8,
  [switch]$Resume,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_hierarchical_residual_v7.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$TargetCases = 1050,
  [int]$FitEvents = 14,
  [int]$CalibrationEvents = 6,
  [int]$ValidationEvents = 8,
  [int]$TrainBoundaryCases = 360,
  [int]$CalibrationBoundaryCases = 180,
  [int]$ValidationBoundaryCases = 240,
  [int]$EffectEpochs = 160,
  [int]$EffectBatchSize = 64
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 Tier2 residual-v8 expansion] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

$RootOut = "outputs\project6_36_tier2_residual_v8_expansion_balanced"
$BaseDataset = "outputs\project6_36_tier2_residual_v7\effect_dataset\same_state_raw_joint_36_tier2_residual_v7.npz"
$ReferenceBank = "outputs\data_bank_train_v8_storage_variablepump\trajectories"
$Manifest = "$RootOut\paired_plan\tier2_residual_v8_expansion_manifest.csv"
$CaseDir = "$RootOut\paired_cases"
$ExpansionDatasetDir = "$RootOut\effect_dataset_supplement"
$ExpansionDataset = "$ExpansionDatasetDir\same_state_tier2_residual_v8_expansion_supplement.npz"
$FinalDatasetDir = "$RootOut\effect_dataset"
$FinalDataset = "$FinalDatasetDir\same_state_raw_joint_36_tier2_residual_v8_expanded.npz"
$DeploymentGateDatasetDir = "$RootOut\effect_dataset_deployment_gate"
$DeploymentGateDataset = "$DeploymentGateDatasetDir\same_state_raw_joint_36_tier2_residual_v8_deployment_gate.npz"
$DeploymentGateSplitReport = "$DeploymentGateDatasetDir\deployment_gate_split_report.json"
$CalibrationFile = "$RootOut\calibration_events.txt"
$ValidationFile = "$RootOut\locked_validation_events.txt"
$ModelDir = "outputs\models_temporal_joint_36_tier2_residual_v8_expanded"
$Model = "$ModelDir\raw_joint_36_tier2_residual_v8_expanded.pt"
$Report = "$ModelDir\raw_joint_36_tier2_residual_v8_expanded_train_report.json"
$GateJson = "$RootOut\tier2_mpc_gate_v8_expanded.json"
$WarmStart = "outputs\models_temporal_joint_36_tier2_residual_v7\raw_joint_36_tier2_residual_v7.pt"
$WarmReport = "outputs\models_temporal_joint_36_tier2_residual_v7\raw_joint_36_tier2_residual_v7_train_report.json"

Invoke-Python "environment_preflight" @(
  "-c", "import torch,numpy,pandas,yaml; print(torch.__version__, torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

if ($PlanExpansion) {
  Invoke-Python "plan_v8_expansion_manifest" @(
    "scripts\108_plan_tier2_residual_v8_expansion.py",
    "--config", $Config,
    "--base-dataset", $BaseDataset,
    "--reference-bank", $ReferenceBank,
    "--out-dir", $RootOut,
    "--target-cases", "$TargetCases",
    "--fit-events", "$FitEvents",
    "--calibration-events", "$CalibrationEvents",
    "--validation-events", "$ValidationEvents",
    "--train-boundary-cases", "$TrainBoundaryCases",
    "--calibration-boundary-cases", "$CalibrationBoundaryCases",
    "--validation-boundary-cases", "$ValidationBoundaryCases"
  )
}

if ($DryRunExpansion -or $RunExpansion) {
  if (-not (Test-Path $Manifest)) { throw "Missing v8 expansion manifest. Run -PlanExpansion first." }
  $Expected = @((Import-Csv $Manifest) | Where-Object { $_.branch -eq "B" }).Count
  if ($Expected -le 0) { throw "Manifest has no candidate branch rows: $Manifest" }
  $argsRun = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $Manifest,
    "--reference-bank", $ReferenceBank,
    "--out-dir", $CaseDir,
    "--workers", "$Workers",
    "--max-cases", "$Expected",
    "--preflight-noop-filter"
  )
  if ($Resume) { $argsRun += "--resume" }
  if ($DryRunExpansion) { $argsRun += "--dry-run" }
  Invoke-Python $(if ($DryRunExpansion) { "dry_run_v8_expansion_pairs" } else { "run_v8_expansion_pairs" }) $argsRun
}

if ($BuildExpansionDataset) {
  $Results = Join-Path $CaseDir "paired_candidate_results.csv"
  $Failures = Join-Path $CaseDir "failures.csv"
  if (-not (Test-Path $Results)) { throw "Missing v8 expansion SWMM results: $Results" }
  $Expected = @((Import-Csv $Manifest) | Where-Object { $_.branch -eq "B" }).Count
  $Completed = @(Import-Csv $Results).Count
  if ($Completed -ne $Expected) { throw "v8 expansion cases incomplete: $Completed/$Expected. Resume -RunExpansion first." }
  if ((Test-Path $Failures) -and @(Import-Csv $Failures).Count -gt 0) { throw "v8 expansion failures remain: $Failures" }
  Invoke-Python "build_v8_expansion_dataset" @(
    "scripts\89_build_same_state_raw_joint_dataset.py",
    "--config", $Config,
    "--case-dir", $CaseDir,
    "--out-dir", $ExpansionDatasetDir,
    "--dataset-name", "same_state_tier2_residual_v8_expansion_supplement.npz"
  )
}

if ($MergeExpandedDataset) {
  foreach ($Required in @($BaseDataset, $ExpansionDataset, $ValidationFile)) {
    if (-not (Test-Path $Required)) { throw "Missing merge input: $Required" }
  }
  Invoke-Python "merge_base_and_v8_expansion" @(
    "scripts\104_merge_same_state_effect_datasets.py",
    "--base-dataset", $BaseDataset,
    "--supplement-dataset", $ExpansionDataset,
    "--out-npz", $FinalDataset,
    "--base-split-policy", "all_train",
    "--locked-validation-events-file", $ValidationFile
  )
}

if ($AuditExpandedDataset) {
  foreach ($Required in @($FinalDataset, $Manifest)) {
    if (-not (Test-Path $Required)) { throw "Missing audit input: $Required" }
  }
  Invoke-Python "audit_v8_expanded_dataset" @(
    "scripts\90_audit_temporal_joint_information.py",
    "--config", $Config,
    "--dataset", $FinalDataset,
    "--manifest", $Manifest,
    "--out-dir", "$RootOut\effect_dataset_audit"
  )
}

if ($TrainV8) {
  $TrainingDataset = $FinalDataset
  if ($UseDeploymentGateSplit) {
    $TrainingDataset = $DeploymentGateDataset
  }
  foreach ($Required in @($TrainingDataset, $WarmStart, $WarmReport)) {
    if (-not (Test-Path $Required)) { throw "Missing training input: $Required" }
  }
  if ($Resume -and (Test-Path $Model) -and (Test-Path $Report)) {
    Write-Host "[Project6 Tier2 residual-v8 expansion] reuse completed model=$Model"
  } else {
    Invoke-Python "warm_start_train_v8_expanded_effect_model" @(
      "scripts\93_train_raw_joint_action_surrogate_v3.py",
      "--config", $Config,
      "--dataset", $TrainingDataset,
      "--warm-start", $WarmStart,
      "--v2-report", $WarmReport,
      "--architecture-version", "causal_phase_direction_v6",
      "--epochs", "$EffectEpochs",
      "--batch-size", "$EffectBatchSize",
      "--device", $Device,
      "--learning-rate", "0.0001",
      "--fine-tune-action-encoder",
      "--fine-tune-state-interaction",
      "--action-learning-rate-scale", "0.10",
      "--state-learning-rate-scale", "0.02",
      "--direction-loss-weight", "2.0",
      "--direction-classification-loss-weight", "2.0",
      "--peak-direction-loss-multiplier", "4.0",
      "--peak-aggregate-loss-multiplier", "2.0",
      "--peak-sequence-loss-multiplier", "2.0",
      "--peak-direction-sample-weight", "3.0",
      "--direction-eval-exclude-candidate-kinds", "strong_counterfactual,strong_single_or_pair",
      "--pairwise-ranking-loss-weight", "1.0",
      "--classification-loss-weight", "1.50",
      "--reference-loss-weight", "0.05",
      "--balanced-sampling",
      "--balanced-epoch-multiplier", "2.5",
      "--deployment-source-token", "project6_36_tier2_residual_v8_expansion",
      "--legacy-replay-weight", "0.10",
      "--calibration-events-file", $CalibrationFile,
      "--selection-objective", "gate_aligned",
      "--offline-safety-sample-weight", "0.20",
      "--uncertainty-coverage", "0.90",
      "--selection-every", "2",
      "--lr-plateau-patience", "2",
      "--lr-plateau-factor", "0.5",
      "--early-stopping-patience", "10",
      "--seed", "20260715",
      "--out-dir", $ModelDir,
      "--model-name", "raw_joint_36_tier2_residual_v8_expanded.pt",
      "--report-name", "raw_joint_36_tier2_residual_v8_expanded_train_report.json"
    )
  }
}

if ($GateV8) {
  if (-not (Test-Path $Report)) { throw "Missing v8 report. Run -TrainV8 first: $Report" }
  Invoke-Python "strict_v8_tier2_gate" @(
    "scripts\99_mpc_gate_preflight.py",
    "--config", $Config,
    "--model-report", $Report,
    "--out-json", $GateJson,
    "--enforce",
    "--require-tier2"
  )
}

if ($BuildDeploymentGateSplit) {
  if (-not (Test-Path $FinalDataset)) { throw "Missing expanded dataset. Run -MergeExpandedDataset first: $FinalDataset" }
  Invoke-Python "build_deployment_gate_event_split" @(
    "scripts\115_resplit_effect_dataset_for_deployment_gate.py",
    "--config", $Config,
    "--dataset", $FinalDataset,
    "--out-npz", $DeploymentGateDataset,
    "--out-report", $DeploymentGateSplitReport,
    "--validation-events", "$ValidationEvents"
  )
}

if (-not ($PlanExpansion -or $DryRunExpansion -or $RunExpansion -or $BuildExpansionDataset -or $MergeExpandedDataset -or $AuditExpandedDataset -or $BuildDeploymentGateSplit -or $TrainV8 -or $GateV8)) {
  Write-Host "Select: -PlanExpansion -DryRunExpansion -RunExpansion -BuildExpansionDataset -MergeExpandedDataset -AuditExpandedDataset -BuildDeploymentGateSplit -TrainV8 -GateV8"
}
