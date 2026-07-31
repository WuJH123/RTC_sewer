[CmdletBinding()]
param(
  [switch]$PlanCoverage,
  [switch]$DryRunPaired,
  [switch]$RunPaired,
  [switch]$BuildDataset,
  [switch]$TrainEffect,
  [switch]$Gate,
  [switch]$Resume,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_causal_effect_v3.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$MaxCandidateCases = 720,
  [int]$MinTrainEventsPerCell = 3,
  [int]$MinValidationEventsPerCell = 2,
  [int]$EffectEpochs = 120,
  [int]$EffectBatchSize = 32
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 36 causal-effect-v3] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

$BaseDataset = "outputs\project6_36_temporal_joint_peakfixed_v1\effect_dataset\same_state_raw_joint_36_peakfixed_v1.npz"
$PlanDir = "outputs\project6_36_causal_effect_v3\paired_plan"
$Manifest = "$PlanDir\targeted_causal_effect_manifest.csv"
$CaseDir = "outputs\project6_36_causal_effect_v3\paired_cases"
$SupplementDir = "outputs\project6_36_causal_effect_v3\effect_dataset_supplement"
$Supplement = "$SupplementDir\same_state_raw_joint_36_causal_supplement_v3.npz"
$CombinedDir = "outputs\project6_36_causal_effect_v3\effect_dataset"
$Combined = "$CombinedDir\same_state_raw_joint_36_causal_effect_v3.npz"
$WarmStart = "outputs\models_temporal_joint_36_recovery_v2\raw_joint_36_same_state_recovery_v2.pt"
$PriorReport = "outputs\models_temporal_joint_36_recovery_v2\raw_joint_36_same_state_recovery_v2_train_report.json"
$ModelDir = "outputs\models_temporal_joint_36_causal_effect_v3"
$Model = "$ModelDir\raw_joint_36_causal_effect_v3.pt"
$Report = "$ModelDir\raw_joint_36_causal_effect_v3_train_report.json"
$GateJson = "outputs\project6_36_causal_effect_v3\mpc_gate_preflight.json"

Invoke-Python "environment_preflight" @(
  "-c", "import torch,numpy,pandas,yaml; print(torch.__version__, torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

if ($PlanCoverage) {
  Invoke-Python "audit_and_plan_facility_direction_phase_event_coverage" @(
    "scripts\103_plan_causal_effect_coverage.py",
    "--config", $Config,
    "--dataset", $BaseDataset,
    "--out-dir", $PlanDir,
    "--min-train-events", "$MinTrainEventsPerCell",
    "--min-validation-events", "$MinValidationEventsPerCell",
    "--max-candidate-cases", "$MaxCandidateCases"
  )
}

if ($DryRunPaired -or $RunPaired) {
  if (-not (Test-Path $Manifest)) { throw "Missing targeted manifest. Run -PlanCoverage first: $Manifest" }
  $pairedArgs = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $Manifest,
    "--out-dir", $CaseDir,
    "--workers", "$Workers",
    "--max-cases", "$MaxCandidateCases",
    "--preflight-noop-filter"
  )
  if ($Resume) { $pairedArgs += "--resume" }
  if ($DryRunPaired) { $pairedArgs += "--dry-run" }
  Invoke-Python $(if ($DryRunPaired) { "paired_preflight" } else { "run_targeted_same_state_pairs" }) $pairedArgs
}

if ($BuildDataset) {
  Invoke-Python "build_causal_supplement_dataset" @(
    "scripts\89_build_same_state_raw_joint_dataset.py",
    "--config", $Config,
    "--case-dir", $CaseDir,
    "--out-dir", $SupplementDir,
    "--dataset-name", "same_state_raw_joint_36_causal_supplement_v3.npz"
  )
  Invoke-Python "merge_base_and_causal_supplement" @(
    "scripts\104_merge_same_state_effect_datasets.py",
    "--base-dataset", $BaseDataset,
    "--supplement-dataset", $Supplement,
    "--out-npz", $Combined
  )
}

if ($TrainEffect) {
  if (-not (Test-Path $Combined)) { throw "Missing combined causal dataset. Run -BuildDataset first: $Combined" }
  if (-not (Test-Path $WarmStart)) { throw "Missing warm-start checkpoint: $WarmStart" }
  if ($Resume -and (Test-Path $Model) -and (Test-Path $Report)) {
    Write-Host "[Project6 36 causal-effect-v3] reuse completed model=$Model"
  } else {
    Invoke-Python "train_phase_conditioned_causal_effect" @(
      "scripts\93_train_raw_joint_action_surrogate_v3.py",
      "--config", $Config,
      "--dataset", $Combined,
      "--warm-start", $WarmStart,
      "--v2-report", $PriorReport,
      "--architecture-version", "causal_phase_safety_v5",
      "--epochs", "$EffectEpochs",
      "--batch-size", "$EffectBatchSize",
      "--device", $Device,
      "--learning-rate", "0.0003",
      "--fine-tune-action-encoder",
      "--action-learning-rate-scale", "0.05",
      "--direction-loss-weight", "1.5",
      "--classification-loss-weight", "1.5",
      "--reference-loss-weight", "0.05",
      "--balanced-sampling",
      "--balanced-epoch-multiplier", "2.0",
      "--calibration-event-fraction", "0.20",
      "--uncertainty-coverage", "0.90",
      "--selection-every", "5",
      "--seed", "20260714",
      "--out-dir", $ModelDir,
      "--model-name", "raw_joint_36_causal_effect_v3.pt",
      "--report-name", "raw_joint_36_causal_effect_v3_train_report.json"
    )
  }
}

if ($Gate) {
  if (-not (Test-Path $Report)) { throw "Missing v3 report. Run -TrainEffect first: $Report" }
  & $Python "scripts\99_mpc_gate_preflight.py" --config $Config --model-report $Report --out-json $GateJson --enforce
  if ($LASTEXITCODE -eq 2) {
    Write-Host "[Project6 36 causal-effect-v3] validation gate is false; rolling-horizon Smoke remains blocked."
    exit 2
  }
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [strict_mpc_gate] with exit code $LASTEXITCODE" }
}

if (-not ($PlanCoverage -or $DryRunPaired -or $RunPaired -or $BuildDataset -or $TrainEffect -or $Gate)) {
  Write-Host "Select: -PlanCoverage -DryRunPaired -RunPaired -BuildDataset -TrainEffect -Gate"
}
