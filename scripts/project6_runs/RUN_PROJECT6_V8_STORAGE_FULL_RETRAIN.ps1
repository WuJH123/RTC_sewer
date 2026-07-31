param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6_v8_storage.yaml",
  [ValidateSet("Smoke", "Train", "Calibration", "Formal35", "Formal70", "Evaluate", "Full")]
  [string[]]$Stage = @("Smoke"),
  [string]$RunTag = "project6_v8_storage_variablepump_T5_T100_v3",
  [string]$Device = "cuda",
  [int]$Workers = 12,
  [int]$ProposedWorkers = 1,
  [int]$GatEpochs = 150,
  [int]$SurrogateEpochs = 180,
  [int]$GatMaxTrainSamplesPerEpoch = 0,
  [int]$MaxTrajectories = 0,
  [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location $Root
if (-not $Python) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
if ($Device -eq "cuda") {
  & $Python -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 2)"
  if ($LASTEXITCODE -ne 0) { throw "CUDA was requested but is unavailable in $Python" }
}

$Stage = @($Stage | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Stage -contains "Full") { $Stage = @("Train", "Calibration", "Formal35", "Evaluate") }
$allowed = @("Smoke", "Train", "Calibration", "Formal35", "Formal70", "Evaluate")
$invalid = @($Stage | Where-Object { $_ -notin $allowed })
if ($invalid.Count) { throw "Unknown stage(s): $($invalid -join ', ')" }

function Invoke-Py([string]$Label, [string[]]$PythonArgs) {
  Write-Host "[Project6 v8-storage full retrain] step=$Label"
  & $Python @PythonArgs
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

function Get-EventIds([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Event table not found: $Path" }
  return ((Import-Csv -LiteralPath $Path).event_id -join ",")
}

$CfgPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Root $Config }
$ScenarioDir = Join-Path $Root "outputs\storage_retrofit\$RunTag"
$FormalRunDir = Join-Path $Root "outputs\closed_loop_paired_no_controls\formal\$RunTag"
$EvalDir = Join-Path $Root "outputs\evaluation_$RunTag"
$DataDir = Join-Path $Root "data\surrogate_v8_storage_variablepump"
$GatFeatures = Join-Path $Root "outputs\gat_reconstructed_features_v8_storage_variablepump"
$Model = Join-Path $Root "outputs\models_v8_storage_variablepump\horizon_temporal_gnn.pt"

function Prepare-Scenario {
  Invoke-Py "audit_inp" @("scripts\00_audit_inp.py", "--config", $CfgPath)
  Invoke-Py "generate_rainfall_library" @("scripts\01_generate_rainfall_library.py", "--config", $CfgPath, "--mode", "formal")
  Invoke-Py "select_priority_and_sensors" @("scripts\02_select_priority_and_sensors.py", "--config", $CfgPath)
  Invoke-Py "build_influence_domains" @("scripts\49_build_influence_domains.py", "--config", $CfgPath, "--khop", "3", "--fallback-khop", "30", "--max-candidates-per-priority", "80", "--max-storage-controls-per-priority", "8", "--max-regulators-per-priority", "32", "--max-pumps-per-priority", "12")
  Invoke-Py "build_storage_scenario" @("scripts\84_build_v8_storage_retrofit_scenario.py", "--config", $CfgPath, "--run-tag", $RunTag)
}

function Train-FromScratch {
  Prepare-Scenario
  $TrajectoryArgs = @("scripts\03_generate_generic_trajectories.py", "--config", $CfgPath, "--mode", "full", "--workers", "$Workers")
  if ($MaxTrajectories -gt 0) { $TrajectoryArgs += @("--max-trajectories", "$MaxTrajectories") }
  if ($Resume) { $TrajectoryArgs += "--resume" }
  Invoke-Py "generate_36asset_trajectories" $TrajectoryArgs
  Invoke-Py "clean_trajectory_bank" @("scripts\64_clean_current_trajectory_bank.py", "--config", $CfgPath, "--apply", "--quarantine-stale-details")
  # Effect labels must share the same no-control reference used by online MPC.
  # EFD/Auto-RBC are retained as trajectories and formal comparison baselines.
  Invoke-Py "build_36asset_tensor_cache" @("scripts\04_build_tensor_cache.py", "--config", $CfgPath, "--max-files", "0", "--time-stride", "1", "--horizon-steps", "6", "--reference-policies", "no_control")
  $GatArgs = @("scripts\05_train_gat.py", "--config", $CfgPath, "--epochs", "$GatEpochs", "--device", $Device, "--eval-every", "5", "--patience", "20", "--score-full-weight", "0.70", "--score-priority-weight", "0.30")
  if ($GatMaxTrainSamplesPerEpoch -gt 0) { $GatArgs += @("--max-train-samples-per-epoch", "$GatMaxTrainSamplesPerEpoch") }
  Invoke-Py "train_gat" $GatArgs
  $FeatureArgs = @("scripts\41_build_gat_reconstructed_feature_cache.py", "--config", $CfgPath, "--device", $Device, "--batch-size", "32", "--out-dir", $GatFeatures)
  if ($Resume) { $FeatureArgs += "--resume" }
  Invoke-Py "build_gat_feature_cache" $FeatureArgs
  $HorizonArgs = @("scripts\42_build_horizon_surrogate_dataset.py", "--config", $CfgPath, "--horizon-steps", "6", "--history-steps", "3", "--stride", "3", "--workers", "$Workers", "--source-scope", "generic_trajectories", "--gat-feature-cache-dir", $GatFeatures, "--require-gat-features", "--trust-gat-feature-cache")
  if ($Resume) { $HorizonArgs += "--resume" }
  Invoke-Py "build_horizon_dataset" $HorizonArgs
  # Prefer the complete parquet dataset.  A CSV may be a compatibility
  # fallback or an interrupted export and must not silently replace it.
  $Dataset = Join-Path $DataDir "horizon_mpc_dataset.parquet"
  if (-not (Test-Path -LiteralPath $Dataset)) { $Dataset = Join-Path $DataDir "horizon_mpc_dataset.csv" }
  if (-not (Test-Path -LiteralPath $Dataset)) { throw "Horizon dataset is missing under $DataDir" }
  Invoke-Py "train_temporal_horizon_surrogate" @("scripts\43_train_horizon_surrogate.py", "--config", $CfgPath, "--dataset", $Dataset, "--model-kind", "temporal_gnn", "--epochs", "$SurrogateEpochs", "--batch-size", "256", "--patience", "20", "--device", $Device, "--model-output", $Model, "--report-dir", (Join-Path $Root "outputs\surrogate_v8_storage_variablepump"))
  Invoke-Py "validate_horizon_surrogate" @("scripts\44_validate_horizon_surrogate.py", "--config", $CfgPath, "--dataset", $Dataset, "--model", $Model, "--out-dir", (Join-Path $Root "outputs\surrogate_v8_storage_variablepump"), "--fail-on-quality")
  Invoke-Py "train_uncertainty" @("scripts\45_train_uncertainty_heads.py", "--config", $CfgPath, "--dataset", $Dataset, "--model", $Model)
  Invoke-Py "validate_uncertainty" @("scripts\46_validate_uncertainty_gate.py", "--config", $CfgPath, "--dataset", $Dataset, "--model", $Model)
}

function Run-ClosedLoop([string]$Kind) {
  $table = switch ($Kind) {
    "Calibration" { Join-Path $ScenarioDir "calibration_events.csv" }
    "Formal35" { Join-Path $ScenarioDir "formal35_events.csv" }
    "Formal70" { Join-Path $ScenarioDir "formal70_events.csv" }
    default { throw "Unsupported closed-loop kind: $Kind" }
  }
  $tag = if ($Kind -eq "Formal35") { $RunTag } else { "$RunTag`_$($Kind.ToLower())" }
  $args = @("scripts\08_run_closed_loop.py", "--config", $CfgPath, "--mode", "formal", "--run-tag", $tag, "--device", $Device, "--workers", "$Workers", "--proposed-workers", "$ProposedWorkers", "--event-ids", (Get-EventIds $table), "--proposed-controller", "generic_gat_mpc")
  if ($Resume) { $args += "--skip-existing" }
  Invoke-Py "closed_loop_$($Kind.ToLower())" $args
}

foreach ($item in $Stage) {
  switch ($item) {
    "Smoke" {
      Prepare-Scenario
      Invoke-Py "smoke_trajectories" @("scripts\03_generate_generic_trajectories.py", "--config", $CfgPath, "--mode", "debug", "--max-trajectories", "24", "--workers", "$Workers")
    }
    "Train" { Train-FromScratch }
    "Calibration" { Run-ClosedLoop "Calibration" }
    "Formal35" { Run-ClosedLoop "Formal35" }
    "Formal70" {
      Invoke-Py "build_extended_event_split" @("scripts\84_build_v8_storage_retrofit_scenario.py", "--config", $CfgPath, "--run-tag", $RunTag, "--extended")
      Run-ClosedLoop "Formal70"
    }
    "Evaluate" {
      if (-not (Test-Path -LiteralPath $FormalRunDir)) { throw "Formal35 run is absent: $FormalRunDir" }
      Invoke-Py "recalculate_priority_metrics" @("scripts\61_recalculate_project2_priority_zone_metrics.py", "--config", $CfgPath, "--run-dir", $FormalRunDir, "--out-dir", $EvalDir)
      $metrics = Get-ChildItem -LiteralPath $EvalDir -Filter "*event_policy_metrics_main.csv" | Select-Object -First 1
      if ($null -eq $metrics) { throw "Missing main policy metrics in $EvalDir" }
      Invoke-Py "formal_gate" @("scripts\75_no_control_repair_gate.py", "--config", $CfgPath, "--event-policy-metrics", $metrics.FullName, "--out-dir", $EvalDir)
    }
  }
}
