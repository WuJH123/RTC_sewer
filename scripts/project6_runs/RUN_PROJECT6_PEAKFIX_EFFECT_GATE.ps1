[CmdletBinding()]
param(
  [switch]$RepairLabels,
  [switch]$ValidatePretrain,
  [switch]$TrainEffect,
  [switch]$Gate,
  [switch]$All,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_temporal_joint_peakfixed.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$EffectEpochs = 120,
  [int]$EffectBatchSize = 32
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 peakfix-effect] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

$SourceDataset = "outputs\project6_36_temporal_joint_v4\effect_dataset\same_state_raw_joint_36_v3.npz"
$PeakfixedDataset = "outputs\project6_36_temporal_joint_peakfixed_v1\effect_dataset\same_state_raw_joint_36_peakfixed_v1.npz"
$DynamicsModel = "outputs\models_temporal_action_pretrain_36_actionaware_v2\raw_joint_36_actionaware_observational_dynamics.pt"
$PretrainAuditDir = "outputs\models_temporal_action_pretrain_36_peakfixed_audit_v1"
$EffectOutDir = "outputs\models_temporal_joint_36_peakfixed_v1"
$EffectReport = "$EffectOutDir\raw_joint_36_same_state_v3_train_report.json"

Invoke-Python "environment_preflight" @(
  "-c", "import torch; print(torch.__version__, torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

if ($All) { $RepairLabels = $true; $ValidatePretrain = $true; $TrainEffect = $true; $Gate = $true }

if ($RepairLabels) {
  if (Test-Path $PeakfixedDataset) {
    Write-Host "[Project6 peakfix-effect] reuse repaired dataset=$PeakfixedDataset"
  } else {
    Invoke-Python "repair_peak_labels" @(
      "scripts\103_repair_peak_label_semantics.py",
      "--dataset", $SourceDataset,
      "--out-dataset", $PeakfixedDataset
    )
  }
}

if ($ValidatePretrain) {
  Invoke-Python "validate_current_pretrain_with_fixed_peak_metrics" @(
    "scripts\102_train_temporal_action_dynamics_pretrain.py",
    "--config", $Config,
    "--dataset-index", "outputs\cache_temporal_action_pretrain_36\temporal_action_pretrain_36.npz",
    "--out-dir", $PretrainAuditDir,
    "--model-name", "raw_joint_36_actionaware_peakfixed_audit.pt",
    "--warm-start", $DynamicsModel,
    "--epochs", "0",
    "--batch-size", "128",
    "--max-validation-samples", "16384",
    "--scale-samples", "5000",
    "--device", $Device,
    "--amp", "--tf32",
    "--prefetch-depth", "8",
    "--cpu-threads", "12"
  )
}

if ($TrainEffect) {
  if (-not (Test-Path $PeakfixedDataset)) { throw "Missing repaired same-state dataset. Run -RepairLabels first." }
  Invoke-Python "train_same_state_effect" @(
    "scripts\93_train_raw_joint_action_surrogate_v3.py",
    "--config", $Config,
    "--dataset", $PeakfixedDataset,
    "--warm-start", $DynamicsModel,
    "--v2-report", "$PretrainAuditDir\temporal_action_dynamics_pretrain_report.json",
    "--epochs", "$EffectEpochs",
    "--batch-size", "$EffectBatchSize",
    "--device", $Device,
    "--learning-rate", "0.0005",
    "--fine-tune-action-encoder",
    "--action-learning-rate-scale", "0.05",
    "--direction-loss-weight", "1.0",
    "--classification-loss-weight", "1.0",
    "--reference-loss-weight", "0.10",
    "--out-dir", $EffectOutDir,
    "--model-name", "raw_joint_36_same_state_peakfixed_v1.pt"
  )
}

if ($Gate) {
  if (-not (Test-Path $EffectReport)) { throw "Missing effect report. Run -TrainEffect first." }
  Invoke-Python "strict_mpc_gate" @(
    "scripts\99_mpc_gate_preflight.py",
    "--config", $Config,
    "--model-report", $EffectReport,
    "--out-json", "outputs\project6_36_temporal_joint_peakfixed_v1\mpc_gate_preflight.json",
    "--enforce"
  )
}

if (-not ($RepairLabels -or $ValidatePretrain -or $TrainEffect -or $Gate -or $All)) {
  Write-Host "Select -RepairLabels -ValidatePretrain -TrainEffect -Gate, or -All. This script intentionally has no Smoke stage."
}
