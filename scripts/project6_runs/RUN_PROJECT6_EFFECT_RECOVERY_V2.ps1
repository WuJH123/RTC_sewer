[CmdletBinding()]
param(
  [switch]$TrainEffect,
  [switch]$Gate,
  [switch]$All,
  [switch]$Resume,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_temporal_joint_recovery_v2.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$EffectEpochs = 160,
  [int]$EffectBatchSize = 32,
  [double]$CalibrationEventFraction = 0.20,
  [double]$BalancedEpochMultiplier = 2.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 effect-recovery-v2] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

$Dataset = "outputs\project6_36_temporal_joint_peakfixed_v1\effect_dataset\same_state_raw_joint_36_peakfixed_v1.npz"
$WarmStart = "outputs\models_temporal_joint_36_peakfixed_v1\raw_joint_36_same_state_peakfixed_v1.pt"
$PreviousReport = "outputs\models_temporal_joint_36_peakfixed_v1\raw_joint_36_same_state_v3_train_report.json"
$OutDir = "outputs\models_temporal_joint_36_recovery_v2"
$Model = "$OutDir\raw_joint_36_same_state_recovery_v2.pt"
$Report = "$OutDir\raw_joint_36_same_state_recovery_v2_train_report.json"
$GateJson = "outputs\project6_36_temporal_joint_recovery_v2\mpc_gate_preflight.json"

Invoke-Python "environment_preflight" @(
  "-c", "import torch; print(torch.__version__, torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

if ($All) { $TrainEffect = $true; $Gate = $true }

if ($TrainEffect) {
  if (-not (Test-Path $Dataset)) { throw "Missing same-state dataset: $Dataset" }
  if (-not (Test-Path $WarmStart)) { throw "Missing peak-fixed warm start: $WarmStart" }
  if ($Resume -and (Test-Path $Model) -and (Test-Path $Report)) {
    Write-Host "[Project6 effect-recovery-v2] reuse completed model=$Model"
  } else {
    Invoke-Python "train_balanced_calibrated_v4_effect" @(
      "scripts\93_train_raw_joint_action_surrogate_v3.py",
      "--config", $Config,
      "--dataset", $Dataset,
      "--warm-start", $WarmStart,
      "--v2-report", $PreviousReport,
      "--architecture-version", "priority_aware_safety_v4",
      "--epochs", "$EffectEpochs",
      "--batch-size", "$EffectBatchSize",
      "--device", $Device,
      "--learning-rate", "0.0003",
      "--fine-tune-action-encoder",
      "--action-learning-rate-scale", "0.03",
      "--direction-loss-weight", "2.0",
      "--classification-loss-weight", "1.25",
      "--reference-loss-weight", "0.05",
      "--balanced-sampling",
      "--balanced-epoch-multiplier", "$BalancedEpochMultiplier",
      "--calibration-event-fraction", "$CalibrationEventFraction",
      "--uncertainty-coverage", "0.90",
      "--selection-every", "5",
      "--seed", "20260714",
      "--out-dir", $OutDir,
      "--model-name", "raw_joint_36_same_state_recovery_v2.pt",
      "--report-name", "raw_joint_36_same_state_recovery_v2_train_report.json"
    )
  }
}

if ($Gate) {
  if (-not (Test-Path $Report)) { throw "Missing recovery report. Run -TrainEffect first: $Report" }
  Write-Host "[Project6 effect-recovery-v2] step=strict_mpc_gate"
  & $Python "scripts\99_mpc_gate_preflight.py" `
    "--config" $Config `
    "--model-report" $Report `
    "--out-json" $GateJson `
    "--enforce"
  if ($LASTEXITCODE -eq 2) {
    Write-Host "[Project6 effect-recovery-v2] strict gate returned false. This is a model-validation result, not a Python crash. Smoke remains blocked."
    exit 2
  }
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [strict_mpc_gate] with exit code $LASTEXITCODE" }
}

if (-not ($TrainEffect -or $Gate -or $All)) {
  Write-Host "Select -TrainEffect, -Gate, or -All. This recovery runner intentionally has no Smoke stage."
}
