param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_hierarchical_eventbudget_h120_v2.yaml",
  [string]$FitConfig = "configs\wuhan_project6_36_hierarchical_eventbudget_h120_v2_residualfit.yaml",
  [string]$RunTag = "project6_36_hierarchical_eventbudget_h120_v2_dev4",
  [string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$ProposedWorkers = 4,
  [int]$MaxEvents = 48,
  [int]$MaxLogicalPairs = 720,
  [int]$Epochs = 100,
  [int]$BatchSize = 32,
  [switch]$PlanResidual10,
  [switch]$RunResidual10,
  [switch]$BuildDataset,
  [switch]$BuildGuard,
  [switch]$TrainResidual10,
  [switch]$GateResidual10,
  [switch]$EnforceGate,
  [switch]$PreflightDev4,
  [switch]$RunDev4,
  [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $Root

function Run-Step {
  param([string]$Label, [string[]]$StepArgs)
  Write-Host "[Project6 hierarchical residual10 H120] step=$Label"
  & $Python @StepArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

$PlanDir = "outputs/project6_36_residual10_core_paired_h120_v1/paired_plan"
$CaseDir = "outputs/project6_36_residual10_core_paired_h120_v1/paired_cases"
$DatasetDir = "outputs/project6_36_residual10_core_paired_h120_v1/effect_dataset"
$Dataset = "$DatasetDir/same_state_residual10_core_h120_v1.npz"
$GuardDir = "outputs/project6_36_residual10_core_paired_h120_v1/empirical_guard"
$Guard = "$GuardDir/residual10_fit_only_empirical_guard.csv"
$ModelDir = "outputs/models_hierarchical_residual10_h120_v2"
$Model = "$ModelDir/raw_joint_residual10_core_h120_v1.pt"
$Report = "$ModelDir/raw_joint_residual10_core_h120_v1_train_report.json"
$Uncertainty = "$ModelDir/raw_joint_residual10_core_h120_v1_conformal_uncertainty.json"
$DevEvents = "T10_D150_chicago_center,T30_D240_block,T50_D300_chicago_late,T100_D300_chicago_late"

if ($PlanResidual10) {
  Run-Step "plan_residual10_core_paired_cases" @(
    "scripts/119_plan_residual10_core_paired_cases.py",
    "--config", $Config,
    "--out-dir", $PlanDir,
    "--max-events", [string]$MaxEvents,
    "--max-logical-pairs", [string]$MaxLogicalPairs
  )
}

if ($RunResidual10) {
  $runArgs = @(
    "scripts/120_generate_residual10_core_paired_cases.py",
    "--config", $Config,
    "--manifest", "$PlanDir/residual10_core_paired_manifest.csv",
    "--out-dir", $CaseDir,
    "--workers", [string]$Workers,
    "--max-logical-pairs", [string]$MaxLogicalPairs
  )
  if ($Resume) { $runArgs += "--resume" }
  Run-Step "run_residual10_core_paired_cases" $runArgs
}

if ($BuildDataset) {
  Run-Step "build_residual10_core_effect_dataset" @(
    "scripts/121_build_residual10_core_effect_dataset.py",
    "--config", $Config,
    "--case-dir", $CaseDir,
    "--out-dir", $DatasetDir,
    "--dataset-name", "same_state_residual10_core_h120_v1.npz"
  )
}

if ($BuildGuard) {
  Run-Step "build_fit_only_residual10_empirical_guard" @(
    "scripts/122_build_fit_only_residual10_empirical_guard.py",
    "--manifest", "$PlanDir/residual10_core_paired_manifest.csv",
    "--dataset-audit", "$DatasetDir/residual10_core_effect_audit.csv",
    "--out-dir", $GuardDir
  )
}

if ($TrainResidual10) {
  Run-Step "train_residual10_effect_model" @(
    "scripts/93_train_raw_joint_action_surrogate_v3.py",
    "--config", $Config,
    "--dataset", $Dataset,
    "--warm-start", "outputs/models_temporal_joint_36_tier2_residual_v8_expanded/raw_joint_36_tier2_residual_v8_expanded.pt",
    "--epochs", [string]$Epochs,
    "--batch-size", [string]$BatchSize,
    "--device", $Device,
    "--fine-tune-action-encoder",
    "--fine-tune-state-interaction",
    "--balanced-sampling",
    "--selection-objective", "gate_aligned",
    "--calibration-event-fraction", "0.20",
    "--direction-loss-weight", "1.0",
    "--direction-classification-loss-weight", "1.5",
    "--classification-loss-weight", "1.0",
    "--architecture-version", "causal_phase_direction_v6",
    "--peak-direction-loss-multiplier", "2.0",
    "--peak-direction-sample-weight", "2.0",
    "--uncertainty-coverage", "0.90",
    "--early-stopping-patience", "12",
    "--out-dir", $ModelDir,
    "--model-name", "raw_joint_residual10_core_h120_v1.pt",
    "--report-name", "raw_joint_residual10_core_h120_v1_train_report.json"
  )
  Run-Step "extract_residual10_uncertainty" @(
    "scripts/123_extract_residual10_uncertainty.py",
    "--report", $Report,
    "--out-json", $Uncertainty
  )
}

if ($GateResidual10) {
  $gateArgs = @(
    "scripts/99_mpc_gate_preflight.py",
    "--config", $FitConfig,
    "--model-report", $Report,
    "--out-json", "outputs/project6_36_residual10_core_paired_h120_v1/gate/mpc_gate_preflight.json"
  )
  if ($EnforceGate) {
    $gateArgs += "--enforce"
    $gateArgs += "--require-tier2"
  }
  Run-Step "gate_residual10_effect_model" $gateArgs
}

if ($PreflightDev4) {
  Run-Step "strict_hierarchical_preflight_dev4" @(
    "scripts/08_run_closed_loop.py",
    "--config", $FitConfig,
    "--mode", "formal",
    "--run-tag", $RunTag,
    "--device", $Device,
    "--workers", [string]$Workers,
    "--proposed-workers", [string]$ProposedWorkers,
    "--event-ids", $DevEvents,
    "--proposed-controller", "hierarchical_core26_residual10",
    "--proposed-base", "clean",
    "--baseline-policies", "no_control",
    "--skip-baselines",
    "--skip-proposed"
  )
}

if ($RunDev4) {
  if (!(Test-Path $Report)) { throw "Missing residual model report: $Report" }
  $R = Get-Content $Report -Raw | ConvertFrom-Json
  if (-not [bool]$R.validation_gate_passed) { throw "Residual model gate is false; not running Dev4." }
  if (-not [bool]$R.rolling_horizon_smoke_eligibility.passed) { throw "Residual smoke eligibility is false; not running Dev4." }
  Run-Step "run_hierarchical_dev4" @(
    "scripts/08_run_closed_loop.py",
    "--config", $FitConfig,
    "--mode", "formal",
    "--run-tag", $RunTag,
    "--device", $Device,
    "--workers", [string]$Workers,
    "--proposed-workers", [string]$ProposedWorkers,
    "--event-ids", $DevEvents,
    "--proposed-controller", "hierarchical_core26_residual10",
    "--proposed-base", "clean",
    "--baseline-policies", "no_control",
    "--skip-existing"
  )
}
