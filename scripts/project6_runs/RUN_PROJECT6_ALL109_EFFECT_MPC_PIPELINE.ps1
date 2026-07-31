param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6.yaml",
  [string]$Project5Root = "E:\RTC_sewer\Project5",
  [string]$Project5Gat = "E:\RTC_sewer\Project5\outputs\models_paired_no_controls\gat_sr0p10.pt",
  [ValidateSet("smoke", "formal")]
  [string]$RunLevel = "smoke",
  [string]$RunTag = "",
  [string]$Device = "cuda",
  [int]$Workers = 8,
  [int]$ProposedWorkers = 4,
  [int]$GatEpochs = 100,
  [int]$SurrogateEpochs = 180,
  [int]$AblationEvents = 10,
  [int]$AblationSamplesPerPhase = 1,
  [int]$AblationMaxActuators = 0,
  [string]$MultiScaleDeltaLevels = "0.05,0.10,0.20,0.40",
  [string]$MultiScaleAbsoluteLevels = "0,0.25,0.50,0.75,1.00",
  [int]$JointMaxGroupSize = 4,
  [int]$JointMaxCombinationsPerPhase = 24,
  [int]$RepresentativeEvents = 30,
  [string]$EventIds = "",
  [switch]$SkipGatTraining,
  [switch]$SkipAblation,
  [switch]$SkipMultiScaleAblation,
  [switch]$SkipJointAblation,
  [switch]$SkipClosedLoop,
  [switch]$AllowGateFail
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param([string]$Label, [string[]]$Arguments)
  Write-Host "[Project6 all109 effect-MPC] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

Push-Location $Root
try {
  if (-not $Python) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
  if (-not (Test-Path $Python)) { throw "Missing Project6 Python: $Python" }
  if ($Device -eq "cuda") {
    & $Python -c "import sys,torch;sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) { throw "CUDA requested but unavailable in Project6 .venv" }
  }
  $ConfigPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Root $Config }
  if (-not $RunTag) {
    $RunTag = if ($RunLevel -eq "formal") { "project6_all109_effect_mpc_formal_v2" } else { "project6_all109_effect_mpc_smoke_v2" }
  }
  $IsFormal = $RunLevel -eq "formal"
  $HorizonMaxFiles = if ($IsFormal) { 0 } else { 120 }
  $AblationMaxEvents = if ($IsFormal) { $AblationEvents } else { [Math]::Min(2, $AblationEvents) }
  $AblationOut = Join-Path $Root "outputs\ablation_all109"
  $ExactEffects = Join-Path $AblationOut "exact_no_control_action_effect_dataset.csv"
  $HorizonDataset = Join-Path $Root "data\surrogate_all109\horizon_mpc_dataset.parquet"
  $RunDir = Join-Path $Root "outputs\closed_loop_paired_no_controls\formal\$RunTag"
  $EvalDir = Join-Path $Root "outputs\evaluation_$RunTag"

  Invoke-Step "audit_inp" @("scripts\00_audit_inp.py", "--config", $ConfigPath)
  Invoke-Step "generate_formal_rainfall" @("scripts\01_generate_rainfall_library.py", "--config", $ConfigPath, "--mode", "formal")
  Invoke-Step "select_priority_sensors" @("scripts\02_select_priority_and_sensors.py", "--config", $ConfigPath)
  Invoke-Step "import_project5_all109" @(
    "scripts\65_import_verified_project5_trajectories.py", "--config", $ConfigPath,
    "--source-root", $Project5Root, "--include-source-events-for-gat", "--resume"
  )
  Invoke-Step "build_all_event_gat_cache" @(
    "scripts\04_build_tensor_cache.py", "--config", $ConfigPath, "--max-files", "0",
    "--time-stride", "1", "--no-current-event-filter",
    "--reference-policies", "no_control,official_mpc,efd_storage_priority,auto_rbc"
  )
  if (-not $SkipGatTraining) {
    $GatArgs = @(
      "scripts\05_train_gat.py", "--config", $ConfigPath, "--epochs", [string]$GatEpochs,
      "--device", $Device, "--max-train-samples-per-epoch", "0", "--eval-every", "5",
      "--patience", "20", "--score-full-weight", "0.90", "--score-priority-weight", "0.10"
    )
    $LocalGat = Join-Path $Root "outputs\models_all109\gat_sr0p10.pt"
    if (Test-Path $LocalGat) {
      $GatArgs += @("--init-checkpoint", $LocalGat)
    }
    elseif (Test-Path $Project5Gat) {
      $GatArgs += @("--init-checkpoint", $Project5Gat)
    }
    Invoke-Step "finetune_gat_all_270_events" $GatArgs
  }
  Invoke-Step "build_influence_domains" @(
    "scripts\49_build_influence_domains.py", "--config", $ConfigPath, "--khop", "3",
    "--fallback-khop", "30", "--max-candidates-per-priority", "160",
    "--max-storage-controls-per-priority", "10", "--max-regulators-per-priority", "48",
    "--max-pumps-per-priority", "57"
  )
  Invoke-Step "build_formal_gat_feature_cache" @(
    "scripts\41_build_gat_reconstructed_feature_cache.py", "--config", $ConfigPath,
    "--device", $Device, "--batch-size", "16", "--max-files", [string]$HorizonMaxFiles,
    "--out-dir", "outputs\gat_reconstructed_features_all109", "--resume"
  )

  if (-not $EventIds) {
    Invoke-Step "select_representative_events" @(
      "scripts\62_select_representative_events.py", "--config", $ConfigPath,
      "--max-events", [string]$RepresentativeEvents, "--out-dir", "outputs\design"
    )
    $EventIds = ((Get-Content "outputs\design\representative_event_ids.txt" | Where-Object { $_.Trim() }) -join ",")
  }
  if (-not $SkipAblation) {
    Invoke-Step "local_single_actuator_screen" @(
      "scripts\76_generate_no_control_single_actuator_ablation.py", "--config", $ConfigPath,
      "--event-ids", $EventIds, "--max-events", [string]$AblationMaxEvents,
      "--max-actuators", [string]$AblationMaxActuators,
      "--samples-per-phase", [string]$AblationSamplesPerPhase, "--delta", "0.05",
      "--hold-steps", "2", "--workers", [string]$Workers, "--resume", "--out-dir", $AblationOut
    )
  }
  if (-not $SkipMultiScaleAblation) {
    Invoke-Step "multiscale_single_actuator_response" @(
      "scripts\76_generate_no_control_single_actuator_ablation.py", "--config", $ConfigPath,
      "--event-ids", $EventIds, "--max-events", [string]$AblationMaxEvents,
      "--max-actuators", [string]$AblationMaxActuators, "--samples-per-phase", [string]$AblationSamplesPerPhase,
      "--delta-levels", $MultiScaleDeltaLevels, "--absolute-levels", $MultiScaleAbsoluteLevels,
      "--hold-steps", "2", "--workers", [string]$Workers, "--resume", "--out-dir", $AblationOut
    )
  }
  if (-not $SkipJointAblation) {
    Invoke-Step "direction_safe_joint_action_ablation" @(
      "scripts\77_generate_no_control_joint_action_ablation.py", "--config", $ConfigPath,
      "--event-ids", $EventIds, "--max-events", [string]$AblationMaxEvents,
      "--samples-per-phase", [string]$AblationSamplesPerPhase, "--max-group-size", [string]$JointMaxGroupSize,
      "--max-combinations-per-phase", [string]$JointMaxCombinationsPerPhase,
      "--max-action-amplitude", "0.10", "--hold-steps", "2", "--workers", [string]$Workers,
      "--resume", "--out-dir", $AblationOut
    )
  }
  if (-not (Test-Path $ExactEffects)) { throw "Missing exact action-effect dataset: $ExactEffects" }

  Invoke-Step "build_paired_horizon_dataset" @(
    "scripts\42_build_horizon_surrogate_dataset.py", "--config", $ConfigPath,
    "--horizon-steps", "6", "--history-steps", "3", "--stride", "3",
    "--max-detail-files", [string]$HorizonMaxFiles, "--workers", [string]$Workers,
    "--source-scope", "generic_trajectories", "--gat-feature-cache-dir",
    "outputs\gat_reconstructed_features_all109", "--require-gat-features",
    "--trust-gat-feature-cache", "--resume"
  )
  if (-not (Test-Path $HorizonDataset)) {
    $CsvFallback = [IO.Path]::ChangeExtension($HorizonDataset, ".csv")
    if (Test-Path $CsvFallback) { $HorizonDataset = $CsvFallback } else { throw "Missing horizon dataset" }
  }
  Invoke-Step "train_temporal_effect_surrogate" @(
    "scripts\43_train_horizon_surrogate.py", "--config", $ConfigPath,
    "--dataset", $HorizonDataset, "--exact-effect-dataset", $ExactEffects,
    "--model-kind", "temporal_gnn", "--epochs", [string]$SurrogateEpochs,
    "--batch-size", "256", "--device", $Device, "--min-samples", "1000"
  )
  $ValidateArgs = @(
    "scripts\44_validate_horizon_surrogate.py", "--config", $ConfigPath,
    "--dataset", $HorizonDataset, "--exact-effect-dataset", $ExactEffects
  )
  if ($IsFormal) { $ValidateArgs += "--fail-on-quality" } else { $ValidateArgs += "--allow-quality-fail" }
  Invoke-Step "validate_exact_action_effect" $ValidateArgs
  Invoke-Step "calibrate_effect_uncertainty" @(
    "scripts\45_train_uncertainty_heads.py", "--config", $ConfigPath,
    "--dataset", $HorizonDataset, "--exact-effect-dataset", $ExactEffects
  )
  Invoke-Step "validate_effect_uncertainty" @(
    "scripts\46_validate_uncertainty_gate.py", "--config", $ConfigPath,
    "--dataset", $HorizonDataset, "--exact-effect-dataset", $ExactEffects
  )

  if (-not $SkipClosedLoop) {
    Invoke-Step "run_closed_loop" @(
      "scripts\08_run_closed_loop.py", "--config", $ConfigPath, "--mode", "formal",
      "--run-tag", $RunTag, "--device", $Device, "--workers", [string]$Workers,
      "--proposed-workers", [string]$ProposedWorkers, "--proposed-controller", "generic_gat_mpc",
      "--proposed-base", "clean", "--event-ids", $EventIds,
      "--baseline-policies", "no_control,internal_rules,efd_storage_priority,auto_rbc", "--skip-existing"
    )
    Invoke-Step "recalculate_metrics" @(
      "scripts\61_recalculate_project2_priority_zone_metrics.py", "--config", $ConfigPath,
      "--run-dir", $RunDir, "--out-dir", $EvalDir
    )
    Invoke-Step "audit_accepted_actions" @(
      "scripts\73_audit_accepted_mpc_actions.py", "--config", $ConfigPath,
      "--run-dir", $RunDir, "--reference-policy", "no_control",
      "--out-dir", (Join-Path $EvalDir "accepted_action_audit")
    )
    Invoke-Step "build_repair_supervision" @(
      "scripts\74_build_no_control_repair_supervision.py", "--config", $ConfigPath,
      "--event-policy-metrics", (Join-Path $EvalDir "project5_priority_event_policy_metrics_main.csv"),
      "--out-dir", (Join-Path $EvalDir "no_control_repair_supervision")
    )
    $GateArgs = @(
      "scripts\75_no_control_repair_gate.py", "--config", $ConfigPath,
      "--event-policy-metrics", (Join-Path $EvalDir "project5_priority_event_policy_metrics_main.csv"),
      "--out-dir", $EvalDir
    )
    if (-not $AllowGateFail) { $GateArgs += "--fail-on-block" }
    Invoke-Step "formal_gate" $GateArgs
  }
  Write-Host "[Project6 all109 effect-MPC] finished run_tag=$RunTag"
}
finally {
  Pop-Location
}
