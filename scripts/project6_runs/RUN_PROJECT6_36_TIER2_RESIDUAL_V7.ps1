[CmdletBinding()]
param(
  [switch]$FreezeAndPlanTier1,
  [switch]$DryRunTier1,
  [switch]$RunTier1,
  [switch]$PlanResidual,
  [switch]$DryRunResidual,
  [switch]$RunResidual,
  [switch]$BuildDataset,
  [switch]$TrainV7,
  [switch]$TrainV7Directional,
  [switch]$Gate,
  [switch]$GateDirectional,
  [switch]$SmokeTier2,
  [switch]$Resume,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_hierarchical_residual_v7.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$FitEvents = 12,
  [int]$CalibrationEvents = 4,
  [int]$ValidationEvents = 8,
  [int]$MaxResidualCases = 760,
  [int]$MinResidualCases = 700,
  [int]$EffectEpochs = 120,
  [int]$EffectBatchSize = 64,
  [string]$SmokeRunTag = "project6_36_tier2_residual_v7_smoke"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 Tier2 residual-v7] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

$RootOut = "outputs\project6_36_tier2_residual_v7"
$BaseDataset = "outputs\project6_36_causal_effect_coverage_v2\effect_dataset_boundary_v6_round2\same_state_raw_joint_36_causal_effect_boundary_v6_round2.npz"
$ReferenceBank = "outputs\data_bank_train_v8_storage_variablepump\trajectories"
$Tier1Manifest = "$RootOut\tier1_screen_plan\tier1_base_manifest.csv"
$Tier1Cases = "$RootOut\tier1_screen_cases"
$ResidualManifest = "$RootOut\residual_plan\tier2_residual_manifest.csv"
$ResidualCases = "$RootOut\residual_cases"
$Tier1DatasetDir = "$RootOut\tier1_screen_effect_dataset"
$Tier1Dataset = "$Tier1DatasetDir\same_state_tier1_screen_v7.npz"
$ResidualDatasetDir = "$RootOut\residual_effect_dataset"
$ResidualDataset = "$ResidualDatasetDir\same_state_tier2_residual_v7.npz"
$NewSupplementDir = "$RootOut\new_deployment_supplement"
$NewSupplement = "$NewSupplementDir\same_state_deployment_supplement_v7.npz"
$FinalDatasetDir = "$RootOut\effect_dataset"
$FinalDataset = "$FinalDatasetDir\same_state_raw_joint_36_tier2_residual_v7.npz"
$CalibrationFile = "$RootOut\calibration_events.txt"
$ValidationFile = "$RootOut\locked_validation_events.txt"
$WarmStart = "outputs\models_temporal_joint_36_causal_effect_boundary_v6_round2\raw_joint_36_causal_effect_boundary_v6_round2.pt"
$WarmReport = "outputs\models_temporal_joint_36_causal_effect_boundary_v6_round2\raw_joint_36_causal_effect_boundary_v6_round2_train_report.json"
$ModelDir = "outputs\models_temporal_joint_36_tier2_residual_v7"
$Model = "$ModelDir\raw_joint_36_tier2_residual_v7.pt"
$Report = "$ModelDir\raw_joint_36_tier2_residual_v7_train_report.json"
$GateJson = "$RootOut\tier2_mpc_gate.json"
$DirectionalModelDir = "outputs\models_temporal_joint_36_tier2_residual_v7_directional"
$DirectionalModel = "$DirectionalModelDir\raw_joint_36_tier2_residual_v7_directional.pt"
$DirectionalReport = "$DirectionalModelDir\raw_joint_36_tier2_residual_v7_directional_train_report.json"
$DirectionalGateJson = "$RootOut\tier2_mpc_gate_directional.json"

Invoke-Python "environment_preflight" @(
  "-c", "import torch,numpy,pandas,yaml; print(torch.__version__, torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

if ($FreezeAndPlanTier1) {
  Invoke-Python "freeze_1451_and_plan_fresh_tier1" @(
    "scripts\107_plan_tier2_residual_v7.py",
    "--stage", "freeze_and_plan_tier1",
    "--config", $Config,
    "--base-dataset", $BaseDataset,
    "--reference-bank", $ReferenceBank,
    "--out-dir", $RootOut,
    "--frozen-rows", "1451",
    "--fit-events", "$FitEvents",
    "--calibration-events", "$CalibrationEvents",
    "--validation-events", "$ValidationEvents"
  )
}

if ($DryRunTier1 -or $RunTier1) {
  if (-not (Test-Path $Tier1Manifest)) { throw "Missing Tier 1 manifest. Run -FreezeAndPlanTier1 first." }
  $ExpectedTier1 = @((Import-Csv $Tier1Manifest) | Where-Object { $_.branch -eq "B" }).Count
  $tier1Args = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $Tier1Manifest,
    "--reference-bank", $ReferenceBank,
    "--out-dir", $Tier1Cases,
    "--workers", "$Workers",
    "--max-cases", "$ExpectedTier1",
    "--preflight-noop-filter"
  )
  if ($Resume) { $tier1Args += "--resume" }
  if ($DryRunTier1) { $tier1Args += "--dry-run" }
  Invoke-Python $(if ($DryRunTier1) { "tier1_preflight" } else { "run_tier1_screen" }) $tier1Args
}

if ($PlanResidual) {
  Invoke-Python "select_safe_tier1_and_plan_residual" @(
    "scripts\107_plan_tier2_residual_v7.py",
    "--stage", "plan_residual",
    "--config", $Config,
    "--base-dataset", $BaseDataset,
    "--reference-bank", $ReferenceBank,
    "--out-dir", $RootOut,
    "--max-residual-cases", "$MaxResidualCases",
    "--min-residual-cases", "$MinResidualCases"
  )
}

if ($DryRunResidual -or $RunResidual) {
  if (-not (Test-Path $ResidualManifest)) { throw "Missing residual manifest. Run -PlanResidual first." }
  $ExpectedResidual = @((Import-Csv $ResidualManifest) | Where-Object { $_.branch -eq "B" }).Count
  $residualArgs = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $ResidualManifest,
    "--reference-bank", $ReferenceBank,
    "--out-dir", $ResidualCases,
    "--workers", "$Workers",
    "--max-cases", "$ExpectedResidual",
    "--preflight-noop-filter"
  )
  if ($Resume) { $residualArgs += "--resume" }
  if ($DryRunResidual) { $residualArgs += "--dry-run" }
  Invoke-Python $(if ($DryRunResidual) { "residual_preflight" } else { "run_deployment_residual_pairs" }) $residualArgs
}

if ($BuildDataset) {
  foreach ($Item in @(
    @{ Manifest = $Tier1Manifest; Cases = $Tier1Cases; Label = "Tier 1" },
    @{ Manifest = $ResidualManifest; Cases = $ResidualCases; Label = "residual" }
  )) {
    $Expected = @((Import-Csv $Item.Manifest) | Where-Object { $_.branch -eq "B" }).Count
    $Results = Join-Path $Item.Cases "paired_candidate_results.csv"
    $Failures = Join-Path $Item.Cases "failures.csv"
    if (-not (Test-Path $Results)) { throw "Missing $($Item.Label) results: $Results" }
    $Completed = @(Import-Csv $Results).Count
    if ($Completed -ne $Expected) { throw "$($Item.Label) cases incomplete: $Completed/$Expected. Resume the SWMM stage." }
    if ((Test-Path $Failures) -and @(Import-Csv $Failures).Count -gt 0) { throw "$($Item.Label) failures remain: $Failures" }
  }
  Invoke-Python "build_tier1_screen_dataset" @(
    "scripts\89_build_same_state_raw_joint_dataset.py", "--config", $Config,
    "--case-dir", $Tier1Cases, "--out-dir", $Tier1DatasetDir,
    "--dataset-name", "same_state_tier1_screen_v7.npz"
  )
  Invoke-Python "build_residual_dataset" @(
    "scripts\89_build_same_state_raw_joint_dataset.py", "--config", $Config,
    "--case-dir", $ResidualCases, "--out-dir", $ResidualDatasetDir,
    "--dataset-name", "same_state_tier2_residual_v7.npz"
  )
  Invoke-Python "merge_new_deployment_rows" @(
    "scripts\104_merge_same_state_effect_datasets.py",
    "--base-dataset", $Tier1Dataset,
    "--supplement-dataset", $ResidualDataset,
    "--out-npz", $NewSupplement
  )
  Invoke-Python "merge_frozen_base_with_locked_validation" @(
    "scripts\104_merge_same_state_effect_datasets.py",
    "--base-dataset", $BaseDataset,
    "--supplement-dataset", $NewSupplement,
    "--out-npz", $FinalDataset,
    "--base-split-policy", "all_train",
    "--locked-validation-events-file", $ValidationFile
  )
}

if ($TrainV7) {
  foreach ($Required in @($FinalDataset, $CalibrationFile, $ValidationFile, $WarmStart, $WarmReport)) {
    if (-not (Test-Path $Required)) { throw "Missing v7 training input: $Required" }
  }
  if ($Resume -and (Test-Path $Model) -and (Test-Path $Report)) {
    Write-Host "[Project6 Tier2 residual-v7] reuse completed model=$Model"
  } else {
    Invoke-Python "warm_start_train_v7" @(
      "scripts\93_train_raw_joint_action_surrogate_v3.py",
      "--config", $Config,
      "--dataset", $FinalDataset,
      "--warm-start", $WarmStart,
      "--v2-report", $WarmReport,
      "--architecture-version", "causal_phase_safety_v5",
      "--epochs", "$EffectEpochs",
      "--batch-size", "$EffectBatchSize",
      "--device", $Device,
      "--learning-rate", "0.0002",
      "--fine-tune-action-encoder",
      "--action-learning-rate-scale", "0.05",
      "--direction-loss-weight", "1.5",
      "--classification-loss-weight", "1.5",
      "--reference-loss-weight", "0.05",
      "--balanced-sampling",
      "--balanced-epoch-multiplier", "2.0",
      "--calibration-events-file", $CalibrationFile,
      "--selection-objective", "gate_aligned",
      "--offline-safety-sample-weight", "0.25",
      "--uncertainty-coverage", "0.90",
      "--selection-every", "5",
      "--seed", "20260714",
      "--out-dir", $ModelDir,
      "--model-name", "raw_joint_36_tier2_residual_v7.pt",
      "--report-name", "raw_joint_36_tier2_residual_v7_train_report.json"
    )
  }
}

if ($TrainV7Directional) {
  foreach ($Required in @($FinalDataset, $CalibrationFile, $ValidationFile, $Model, $Report)) {
    if (-not (Test-Path $Required)) { throw "Missing directional training input: $Required" }
  }
  if ($Resume -and (Test-Path $DirectionalModel) -and (Test-Path $DirectionalReport)) {
    Write-Host "[Project6 Tier2 residual-v7] reuse completed directional model=$DirectionalModel"
  } else {
    Invoke-Python "deployment_focused_directional_warm_start" @(
      "scripts\93_train_raw_joint_action_surrogate_v3.py",
      "--config", $Config,
      "--dataset", $FinalDataset,
      "--warm-start", $Model,
      "--v2-report", $Report,
      "--architecture-version", "causal_phase_safety_v5",
      "--epochs", "$EffectEpochs",
      "--batch-size", "$EffectBatchSize",
      "--device", $Device,
      "--learning-rate", "0.0001",
      "--fine-tune-action-encoder",
      "--fine-tune-state-interaction",
      "--action-learning-rate-scale", "0.10",
      "--state-learning-rate-scale", "0.02",
      "--direction-loss-weight", "2.0",
      "--pairwise-ranking-loss-weight", "1.0",
      "--classification-loss-weight", "1.25",
      "--reference-loss-weight", "0.05",
      "--balanced-sampling",
      "--balanced-epoch-multiplier", "2.0",
      "--deployment-source-token", "project6_36_tier2_residual_v7",
      "--legacy-replay-weight", "0.10",
      "--calibration-events-file", $CalibrationFile,
      "--selection-objective", "gate_aligned",
      "--offline-safety-sample-weight", "0.15",
      "--uncertainty-coverage", "0.90",
      "--selection-every", "2",
      "--lr-plateau-patience", "2",
      "--lr-plateau-factor", "0.5",
      "--early-stopping-patience", "8",
      "--seed", "20260715",
      "--out-dir", $DirectionalModelDir,
      "--model-name", "raw_joint_36_tier2_residual_v7_directional.pt",
      "--report-name", "raw_joint_36_tier2_residual_v7_directional_train_report.json"
    )
  }
}

if ($Gate -or $SmokeTier2) {
  if (-not (Test-Path $Report)) { throw "Missing v7 report. Train v7 first: $Report" }
  Invoke-Python "strict_tier2_gate" @(
    "scripts\99_mpc_gate_preflight.py",
    "--config", $Config,
    "--model-report", $Report,
    "--out-json", $GateJson,
    "--enforce",
    "--require-tier2"
  )
}

if ($GateDirectional) {
  if (-not (Test-Path $DirectionalReport)) { throw "Missing directional report. Run -TrainV7Directional first." }
  Invoke-Python "strict_directional_tier2_gate" @(
    "scripts\99_mpc_gate_preflight.py",
    "--config", $Config,
    "--model-report", $DirectionalReport,
    "--out-json", $DirectionalGateJson,
    "--enforce",
    "--require-tier2"
  )
}

if ($SmokeTier2) {
  $GateResult = Get-Content $GateJson -Raw | ConvertFrom-Json
  if (-not $GateResult.tier2_residual_allowed) { throw "Tier 2 gate is false; closed-loop remains blocked." }
  $SmokeEvents = ((Get-Content $ValidationFile | Where-Object { $_.Trim() }) | Select-Object -First 3) -join ","
  if (-not $SmokeEvents) { throw "Locked validation event file is empty." }
  Invoke-Python "tier2_closed_loop_smoke" @(
    "scripts\08_run_closed_loop.py",
    "--config", $Config,
    "--mode", "debug",
    "--run-tag", $SmokeRunTag,
    "--device", $Device,
    "--workers", "$Workers",
    "--proposed-workers", "1",
    "--event-ids", $SmokeEvents,
    "--baseline-policies", "no_control",
    "--proposed-controller", "temporal_joint_36",
    "--raw-joint-model", $Model,
    "--skip-existing"
  )
}

if (-not ($FreezeAndPlanTier1 -or $DryRunTier1 -or $RunTier1 -or $PlanResidual -or $DryRunResidual -or $RunResidual -or $BuildDataset -or $TrainV7 -or $TrainV7Directional -or $Gate -or $GateDirectional -or $SmokeTier2)) {
  Write-Host "Select: -FreezeAndPlanTier1 -DryRunTier1 -RunTier1 -PlanResidual -DryRunResidual -RunResidual -BuildDataset -TrainV7 -TrainV7Directional -Gate -GateDirectional -SmokeTier2"
}
