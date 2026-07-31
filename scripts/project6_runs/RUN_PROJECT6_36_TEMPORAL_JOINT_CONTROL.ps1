[CmdletBinding()]
param(
  [switch]$Audit,
  [switch]$Tests,
  [switch]$BuildPairedData,
  [switch]$TrainEffect,
  [switch]$Smoke,
  [switch]$Calibration,
  [switch]$Formal,
  [switch]$Evaluate,
  [switch]$Resume,
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6_36_temporal_joint.yaml",
  [string]$RunTag = "project6_36_temporal_joint_v1",
  [string]$ModelReport = "outputs\models_temporal_joint_36_peakfixed_v1\raw_joint_36_same_state_v3_train_report.json",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$Workers = 8,
  [int]$Epochs = 80
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

if (-not $Python) {
  $candidates = @()
  if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX "python.exe") }
  $candidates += "E:\RTC_sewer\Project\env\Scripts\python.exe"
  $command = Get-Command python -ErrorAction SilentlyContinue
  if ($command) { $candidates += $command.Source }
  $Python = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path $Python)) { throw "No usable Python interpreter. Pass -Python explicitly." }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 temporal-joint] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

Invoke-Python "environment_preflight" @("-c", "import sys, numpy, pandas, torch, pyswmm; print(sys.executable); print('torch',torch.__version__,'cuda',torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available(), 'Requested CUDA but this interpreter has no CUDA runtime'")

if ($Audit) {
  Invoke-Python "audit_reuse" @("scripts\85_audit_36_fulltrain_reuse.py", "--config", "configs\wuhan_project6_v8_storage_36.yaml", "--stage", "Audit")
  Invoke-Python "canonical_mapping" @("scripts\86_canonicalize_36_action_order_and_plan.py", "--config", "configs\wuhan_project6_v8_storage_36.yaml", "--stage", "CanonicalizeActionOrder")
  Invoke-Python "correct_paired_manifest" @("scripts\87_build_temporal_joint_paired_manifest.py")
}

if ($Tests) {
  Invoke-Python "unit_tests" @("-m", "pytest", "tests\test_temporal_joint_36_control.py", "tests\test_raw_joint_action_surrogate.py", "tests\test_raw_joint_training_metrics.py", "tests\test_project6_canonical_mapping_integration.py", "-q")
}

if ($BuildPairedData) {
  $arguments = @("scripts\88_generate_same_state_temporal_joint_cases.py", "--config", $Config, "--workers", "$Workers", "--max-cases", "387")
  if ($Resume) { $arguments += "--resume" }
  Invoke-Python "same_state_paired_swmm" $arguments
}

if ($TrainEffect) {
  $effectDataset = "outputs\project6_36_temporal_joint_v1\effect_dataset\same_state_raw_joint_36.npz"
  if ($Resume -and (Test-Path $effectDataset)) {
    Write-Host "[Project6 temporal-joint] reuse effect_dataset=$effectDataset"
  } else {
    Invoke-Python "build_same_state_tensor" @("scripts\89_build_same_state_raw_joint_dataset.py", "--config", $Config)
  }
  Invoke-Python "train_raw_joint_effect" @(
    "scripts\79_train_raw_joint_action_surrogate.py", "--config", $Config,
    "--dataset", $effectDataset,
    "--epochs", "$Epochs", "--batch-size", "16", "--hidden-dim", "96", "--device", $Device,
    "--out-dir", "outputs\models_temporal_joint_36", "--model-name", "raw_joint_36_same_state_v2.pt",
    "--dynamics-warm-start", "outputs\models_temporal_action_pretrain_36\raw_joint_36_observational_dynamics.pt",
    "--require-same-state"
  )
  $report = Get-Content "outputs\models_temporal_joint_36\raw_joint_36_same_state_v2_train_report.json" | ConvertFrom-Json
  if (-not $report.validation_gate_passed) {
    Write-Host "[Project6 temporal-joint] validation_failures=$($report.validation_gate_failures -join ',')"
    throw "Raw-joint validation gate failed. Closed-loop smoke is blocked; inspect the v2 train report before adding targeted paired data."
  }
}

$SmokeEvents = "T10_D150_chicago_late,T20_D150_chicago_center,T100_D300_double_peak"
if ($Smoke) {
  if (-not (Test-Path $ModelReport)) { throw "Missing raw-joint validation report. Run the peakfix effect training first." }
  Invoke-Python "strict_smoke_preflight" @(
    "scripts\99_mpc_gate_preflight.py", "--config", $Config,
    "--model-report", $ModelReport,
    "--out-json", "outputs\project6_36_temporal_joint_peakfixed_v1\smoke_gate_preflight.json",
    "--enforce"
  )
  Invoke-Python "smoke_full36" @(
    "scripts\08_run_closed_loop.py", "--config", $Config, "--mode", "formal",
    "--run-tag", "${RunTag}_smoke_full36", "--event-ids", $SmokeEvents,
    "--baseline-policies", "no_control", "--proposed-controller", "temporal_joint_36",
    "--proposed-base", "clean", "--device", $Device, "--workers", "$Workers",
    "--proposed-workers", "1", "--skip-existing"
  )
  Invoke-Python "smoke_26only" @(
    "scripts\08_run_closed_loop.py", "--config", "configs\wuhan_project6_26_temporal_joint_ablation.yaml", "--mode", "formal",
    "--run-tag", "${RunTag}_smoke_26only", "--event-ids", $SmokeEvents,
    "--skip-baselines", "--proposed-controller", "temporal_joint_36", "--proposed-base", "clean",
    "--device", $Device, "--proposed-workers", "1", "--skip-existing"
  )
}

if ($Calibration) {
  throw "Calibration remains blocked until the three-event smoke action audit passes pump/dwell, simultaneous-action, direction, and fallback checks. Run -Evaluate after -Smoke."
}

if ($Formal) {
  throw "Formal is intentionally blocked: the current GAT has seen every T5-T100 library event. Create a genuinely untouched rainfall holdout before publication formal runs."
}

if ($Evaluate) {
  $runDir = "outputs\closed_loop_paired_no_controls\formal\${RunTag}_smoke_full36"
  if (-not (Test-Path $runDir)) { throw "Missing smoke run: $runDir" }
  Invoke-Python "accepted_action_audit" @("scripts\73_audit_accepted_mpc_actions.py", "--config", $Config, "--run-dir", $runDir, "--reference-policy", "no_control")
}

if (-not ($Audit -or $Tests -or $BuildPairedData -or $TrainEffect -or $Smoke -or $Calibration -or $Formal -or $Evaluate)) {
  Write-Host "Select one or more stages: -Audit -Tests -BuildPairedData -TrainEffect -Smoke -Calibration -Formal -Evaluate. Add -Resume for paired SWMM continuation."
}
