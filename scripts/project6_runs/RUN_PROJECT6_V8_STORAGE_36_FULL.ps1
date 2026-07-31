param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6_v8_storage_36.yaml",
  [ValidateSet("Audit", "Build", "Train", "Calibration", "Formal35", "Formal70", "Evaluate", "Full")]
  [string[]]$Stage = @("Audit"),
  [string]$RunTag = "project6_v8_storage_36_full_v1",
  [string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$ProposedWorkers = 1,
  [int]$GatEpochs = 150,
  [int]$SurrogateEpochs = 180,
  [int]$GatMaxTrainSamplesPerEpoch = 0,
  [int]$MaxTrajectories = 0,
  [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root
if (-not $Python) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }

function Invoke-Py([string]$Label, [string[]]$Args) {
  Write-Host "[Project6 v8-storage-36] step=$Label"
  & $Python @Args
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

function EventIds([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Event split not found: $Path" }
  return ((Import-Csv -LiteralPath $Path).event_id -join ",")
}

$cfgPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Root $Config }
$scenarioDir = Join-Path $Root "outputs\storage_retrofit\$RunTag"
$formalDir = Join-Path $Root "outputs\closed_loop_paired_no_controls\formal\$RunTag"
$evalDir = Join-Path $Root "outputs\evaluation_$RunTag"
$dataDir = Join-Path $Root "data\surrogate_v8_storage_36"
$featureDir = Join-Path $Root "outputs\gat_reconstructed_features_v8_storage_36"
$model = Join-Path $Root "outputs\models_v8_storage_36\horizon_temporal_gnn.pt"

function Prepare {
  Invoke-Py "audit_inp" @("scripts\00_audit_inp.py", "--config", $cfgPath)
  Invoke-Py "generate_rainfall_library" @("scripts\01_generate_rainfall_library.py", "--config", $cfgPath, "--mode", "formal")
  Invoke-Py "select_priority_and_sensors" @("scripts\02_select_priority_and_sensors.py", "--config", $cfgPath)
  Invoke-Py "build_influence_domains" @("scripts\49_build_influence_domains.py", "--config", $cfgPath, "--khop", "3", "--fallback-khop", "30", "--max-candidates-per-priority", "80", "--max-storage-controls-per-priority", "8", "--max-regulators-per-priority", "32", "--max-pumps-per-priority", "12")
  Invoke-Py "build_36_asset_scenario" @("scripts\84_build_v8_storage_retrofit_scenario.py", "--config", $cfgPath, "--run-tag", $RunTag, "--extended")
}

function Train {
  Prepare
  $traj = @("scripts\03_generate_generic_trajectories.py", "--config", $cfgPath, "--mode", "full", "--workers", "$Workers")
  if ($MaxTrajectories -gt 0) { $traj += @("--max-trajectories", "$MaxTrajectories") }
  if ($Resume) { $traj += "--resume" }
  Invoke-Py "generate_36_asset_trajectories" $traj
  Invoke-Py "clean_36_asset_trajectory_bank" @("scripts\64_clean_current_trajectory_bank.py", "--config", $cfgPath, "--apply", "--quarantine-stale-details")
  Invoke-Py "build_36_asset_transition_cache" @("scripts\04_build_tensor_cache.py", "--config", $cfgPath, "--max-files", "0", "--time-stride", "1", "--horizon-steps", "6", "--reference-policies", "no_control")

  $gat = @("scripts\05_train_gat.py", "--config", $cfgPath, "--epochs", "$GatEpochs", "--device", $Device, "--eval-every", "5", "--patience", "20", "--score-full-weight", "0.70", "--score-priority-weight", "0.30")
  if ($GatMaxTrainSamplesPerEpoch -gt 0) { $gat += @("--max-train-samples-per-epoch", "$GatMaxTrainSamplesPerEpoch") }
  Invoke-Py "train_36_scope_gat" $gat
  $features = @("scripts\41_build_gat_reconstructed_feature_cache.py", "--config", $cfgPath, "--device", $Device, "--batch-size", "32", "--out-dir", $featureDir)
  if ($Resume) { $features += "--resume" }
  Invoke-Py "build_36_scope_gat_features" $features
  $horizon = @("scripts\42_build_horizon_surrogate_dataset.py", "--config", $cfgPath, "--horizon-steps", "6", "--history-steps", "3", "--stride", "3", "--workers", "$Workers", "--source-scope", "generic_trajectories", "--gat-feature-cache-dir", $featureDir, "--require-gat-features", "--trust-gat-feature-cache")
  if ($Resume) { $horizon += "--resume" }
  Invoke-Py "build_36_scope_horizon_dataset" $horizon
  $dataset = Join-Path $dataDir "horizon_mpc_dataset.parquet"
  if (-not (Test-Path -LiteralPath $dataset)) { $dataset = Join-Path $dataDir "horizon_mpc_dataset.csv" }
  if (-not (Test-Path -LiteralPath $dataset)) { throw "Horizon dataset missing: $dataDir" }
  Invoke-Py "train_36_scope_temporal_surrogate" @("scripts\43_train_horizon_surrogate.py", "--config", $cfgPath, "--dataset", $dataset, "--model-kind", "temporal_gnn", "--epochs", "$SurrogateEpochs", "--batch-size", "256", "--patience", "20", "--device", $Device, "--model-output", $model, "--report-dir", (Join-Path $Root "outputs\surrogate_v8_storage_36"))
  Invoke-Py "validate_36_scope_temporal_surrogate" @("scripts\44_validate_horizon_surrogate.py", "--config", $cfgPath, "--dataset", $dataset, "--model", $model, "--out-dir", (Join-Path $Root "outputs\surrogate_v8_storage_36"), "--fail-on-quality")
  Invoke-Py "train_36_scope_uncertainty" @("scripts\45_train_uncertainty_heads.py", "--config", $cfgPath, "--dataset", $dataset, "--model", $model)
  Invoke-Py "validate_36_scope_uncertainty" @("scripts\46_validate_uncertainty_gate.py", "--config", $cfgPath, "--dataset", $dataset, "--model", $model)
}

function ClosedLoop([string]$Kind) {
  $events = switch ($Kind) {
    "Calibration" { Join-Path $scenarioDir "calibration_events.csv" }
    "Formal35" { Join-Path $scenarioDir "formal35_events.csv" }
    "Formal70" { Join-Path $scenarioDir "formal70_events.csv" }
    default { throw "Unsupported event split: $Kind" }
  }
  $tag = if ($Kind -eq "Formal35") { $RunTag } else { "$RunTag`_$($Kind.ToLower())" }
  $args = @("scripts\08_run_closed_loop.py", "--config", $cfgPath, "--mode", "formal", "--run-tag", $tag, "--device", $Device, "--workers", "$Workers", "--proposed-workers", "$ProposedWorkers", "--event-ids", (EventIds $events), "--proposed-controller", "generic_gat_mpc", "--baseline-policies", "no_control,internal_rules,efd_storage_priority,auto_rbc")
  if ($Resume) { $args += "--skip-existing" }
  Invoke-Py "closed_loop_$($Kind.ToLower())" $args
}

$requested = @($Stage | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($requested -contains "Full") { $requested = @("Train", "Calibration", "Formal70", "Evaluate") }
foreach ($stageName in $requested) {
  switch ($stageName) {
    "Audit" { Prepare }
    "Build" { Prepare }
    "Train" { Train }
    "Calibration" { ClosedLoop "Calibration" }
    "Formal35" { ClosedLoop "Formal35" }
    "Formal70" { if (-not (Test-Path (Join-Path $scenarioDir "formal70_events.csv"))) { Prepare }; ClosedLoop "Formal70" }
    "Evaluate" {
      if (-not (Test-Path -LiteralPath $formalDir)) { throw "Formal35 result directory missing: $formalDir" }
      Invoke-Py "recalculate_36_scope_metrics" @("scripts\61_recalculate_project2_priority_zone_metrics.py", "--config", $cfgPath, "--run-dir", $formalDir, "--out-dir", $evalDir)
      $metrics = Get-ChildItem -LiteralPath $evalDir -Filter "*event_policy_metrics_main.csv" | Select-Object -First 1
      if ($null -eq $metrics) { throw "Main event-policy metrics missing under $evalDir" }
      Invoke-Py "formal_gate_36_scope" @("scripts\75_no_control_repair_gate.py", "--config", $cfgPath, "--event-policy-metrics", $metrics.FullName, "--out-dir", $evalDir)
    }
    default { throw "Unknown stage: $stageName" }
  }
}

