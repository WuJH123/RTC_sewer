param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6.yaml",
  [ValidateSet("smoke", "formal")]
  [string]$RunLevel = "smoke",
  [string]$RunTag = "",
  [string]$Device = "cpu",
  [int]$Workers = 8,
  [int]$ProposedWorkers = 1,
  [int]$GatEpochs = 5,
  [int]$GatMaxTrainSamplesPerEpoch = 20000,
  [int]$GatEvalEvery = 5,
  [int]$GatPatience = 8,
  [double]$GatScoreFullWeight = 0.80,
  [double]$GatScorePriorityWeight = 0.20,
  [int]$SurrogateEpochs = 5,
  [int]$MaxTrajectories = 24,
  [int]$MaxSteps = 0,
  [string]$EventIds = "",
  [string]$BaselinePolicies = "",
  [int]$RepresentativeEvents = 0,
  [switch]$TrainLegacyGraphSurrogate,
  [switch]$SkipTraining,
  [switch]$SkipClosedLoop,
  [switch]$SkipActionAudits,
  [switch]$AllowGateFail
)

$ErrorActionPreference = "Stop"

function Invoke-PythonStep {
  param(
    [string]$Label,
    [string[]]$Arguments
  )
  Write-Host "[Project6 No-control Repair MPC] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

function Get-ConfigPathOptional {
  param(
    [string]$Dotted
  )
  $Script = @'
import sys
sys.path.insert(0, sys.argv[3])
from sewerrtc.io.project_paths import load_config, cfg_path
try:
    cfg = load_config(sys.argv[1])
    print(cfg_path(cfg, sys.argv[2]))
except Exception:
    print("")
'@
  $Tmp = [System.IO.Path]::GetTempFileName() + ".py"
  try {
    Set-Content -LiteralPath $Tmp -Value $Script -Encoding UTF8
    $Value = (& $Python $Tmp $ConfigPath $Dotted $Root)
    if ($LASTEXITCODE -ne 0) { return "" }
    return ([string]$Value).Trim()
  }
  finally {
    Remove-Item -LiteralPath $Tmp -Force -ErrorAction SilentlyContinue
  }
}

function Get-ConfigValueOptional {
  param(
    [string]$Dotted
  )
  $Script = @'
import sys
sys.path.insert(0, sys.argv[3])
from sewerrtc.io.project_paths import load_config
try:
    cfg = load_config(sys.argv[1])
    obj = cfg
    for key in sys.argv[2].split("."):
        obj = obj[key]
    print(obj)
except Exception:
    print("")
'@
  $Tmp = [System.IO.Path]::GetTempFileName() + ".py"
  try {
    Set-Content -LiteralPath $Tmp -Value $Script -Encoding UTF8
    $Value = (& $Python $Tmp $ConfigPath $Dotted $Root)
    if ($LASTEXITCODE -ne 0) { return "" }
    return ([string]$Value).Trim()
  }
  finally {
    Remove-Item -LiteralPath $Tmp -Force -ErrorAction SilentlyContinue
  }
}

Push-Location $Root

try {
  if (-not $Python) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
  }
  if (-not (Test-Path $Python)) {
    throw "Project6 Python environment not found: $Python. Create .venv and install requirements.txt first."
  }
  if ($Device -eq "cuda") {
    & $Python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) {
      throw "-Device cuda requested, but Project6 Python has no CUDA-enabled PyTorch. Install the matching CUDA wheel or use -Device cpu."
    }
  }
  $ConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $Root $Config }
  if (-not $RunTag) {
    $RunTag = if ($RunLevel -eq "formal") { "project6_no_control_repair_formal_v1" } else { "project6_no_control_repair_smoke_v1" }
  }
  $TrajectoryMode = if ($RunLevel -eq "formal") { "full" } else { "debug" }
  $RainMode = if ($RunLevel -eq "formal") { "formal" } else { "train" }
  $TrajectoryLimit = if ($RunLevel -eq "formal") { 0 } else { $MaxTrajectories }
  $HorizonMaxFiles = if ($RunLevel -eq "formal") { 0 } else { 80 }
  $HorizonMinSamples = if ($RunLevel -eq "formal") { 1000 } else { 50 }
  $HorizonModelKind = if ($RunLevel -eq "formal") { "temporal_gnn" } else { "ridge_baseline" }
  $ClosedLoopMode = "formal"
  $ClosedLoopRoot = Get-ConfigPathOptional "outputs.closed_loop"
  if (-not $ClosedLoopRoot) { $ClosedLoopRoot = Join-Path $Root "outputs\closed_loop_paired_no_controls" }
  $RunDir = Join-Path $ClosedLoopRoot "$ClosedLoopMode\$RunTag"
  $EvalRoot = Get-ConfigPathOptional "outputs.evaluation"
  if ($EvalRoot) {
    $EvalDir = Join-Path $EvalRoot $RunTag
  }
  else {
    $EvalDir = Join-Path $Root "outputs\evaluation_$RunTag"
  }
  $SurrogateRoot = Get-ConfigPathOptional "outputs.surrogate"
  if (-not $SurrogateRoot) { $SurrogateRoot = Join-Path $Root "outputs\surrogate" }
  $GatFeatureRoot = Get-ConfigPathOptional "outputs.gat_features"
  if (-not $GatFeatureRoot) { $GatFeatureRoot = Join-Path $Root "outputs\gat_reconstructed_features" }
  $BenchmarkSource = Get-ConfigValueOptional "benchmark.source"
  $ObjectiveMode = (Get-ConfigValueOptional "controller.objective_mode").ToLower()

  Write-Host "[Project6 No-control Repair MPC] root=$Root"
  Write-Host "[Project6 No-control Repair MPC] run_level=$RunLevel run_tag=$RunTag"

  Invoke-PythonStep "audit_inp" @("scripts\00_audit_inp.py", "--config", $ConfigPath)
  Invoke-PythonStep "generate_rainfall_library" @("scripts\01_generate_rainfall_library.py", "--config", $ConfigPath, "--mode", $RainMode)
  Invoke-PythonStep "select_priority_and_sensors" @("scripts\02_select_priority_and_sensors.py", "--config", $ConfigPath)

  if (-not $SkipTraining) {
    Invoke-PythonStep "generate_generic_trajectories" @(
      "scripts\03_generate_generic_trajectories.py",
      "--config", $ConfigPath,
      "--mode", $TrajectoryMode,
      "--resume",
      "--max-trajectories", [string]$TrajectoryLimit,
      "--max-steps", [string]$MaxSteps,
      "--workers", [string]$Workers
    )

    Invoke-PythonStep "clean_current_trajectory_bank" @(
      "scripts\64_clean_current_trajectory_bank.py",
      "--config", $ConfigPath,
      "--apply",
      "--quarantine-stale-details"
    )

    Invoke-PythonStep "build_tensor_cache" @(
      "scripts\04_build_tensor_cache.py",
      "--config", $ConfigPath,
      "--max-files", "0",
      "--time-stride", "1",
      "--reference-policies", "no_control,official_mpc,efd_storage_priority,auto_rbc"
    )

    Invoke-PythonStep "train_gat" @(
      "scripts\05_train_gat.py",
      "--config", $ConfigPath,
      "--epochs", [string]$GatEpochs,
      "--device", $Device,
      "--max-train-samples-per-epoch", [string]$GatMaxTrainSamplesPerEpoch,
      "--eval-every", [string]$GatEvalEvery,
      "--patience", [string]$GatPatience,
      "--score-full-weight", [string]$GatScoreFullWeight,
      "--score-priority-weight", [string]$GatScorePriorityWeight
    )

    if ($TrainLegacyGraphSurrogate) {
      Invoke-PythonStep "train_graph_surrogate" @(
        "scripts\06_train_surrogate.py",
        "--config", $ConfigPath,
        "--epochs", [string]$SurrogateEpochs,
        "--device", $Device,
        "--batch-size", "64",
        "--eval-batch-size", "64",
        "--val-every", "1",
        "--save-every", "5",
        "--val-max-samples", "8192",
        "--no-full-val-at-end",
        "--max-train-samples-per-epoch", "65536"
      )
    }
    else {
      Write-Host "[Project6 No-control Repair MPC] step=train_graph_surrogate skipped (legacy; use -TrainLegacyGraphSurrogate to run)"
    }
  }

  Invoke-PythonStep "build_influence_domains" @(
    "scripts\49_build_influence_domains.py",
    "--config", $ConfigPath,
    "--khop", "3",
    "--fallback-khop", "30",
    "--max-candidates-per-priority", "160",
    "--max-storage-controls-per-priority", "10",
    "--max-regulators-per-priority", "48",
    "--max-pumps-per-priority", "57"
  )

  Invoke-PythonStep "build_gat_reconstructed_feature_cache" @(
    "scripts\41_build_gat_reconstructed_feature_cache.py",
    "--config", $ConfigPath,
    "--device", $Device,
    "--batch-size", "16",
    "--max-files", [string]$HorizonMaxFiles,
    "--out-dir", $GatFeatureRoot,
    "--resume"
  )

  Invoke-PythonStep "build_horizon_dataset" @(
    "scripts\42_build_horizon_surrogate_dataset.py",
    "--config", $ConfigPath,
    "--horizon-steps", "6",
    "--history-steps", "3",
    "--stride", "3",
    "--max-detail-files", [string]$HorizonMaxFiles,
    "--workers", [string]$Workers,
    "--source-scope", "generic_trajectories",
    "--gat-feature-cache-dir", $GatFeatureRoot,
    "--require-gat-features",
    "--trust-gat-feature-cache",
    "--resume"
  )

  $HorizonAuditName = if ($RunLevel -eq "smoke" -and $HorizonMaxFiles -gt 0) { "horizon_dataset_audit_smoke$HorizonMaxFiles.json" } else { "horizon_dataset_audit.json" }
  $HorizonAudit = Join-Path $SurrogateRoot $HorizonAuditName
  $HorizonDataset = ""
  if (Test-Path $HorizonAudit) {
    $AuditObj = Get-Content -Raw $HorizonAudit | ConvertFrom-Json
    if ($AuditObj.output_dataset -and (Test-Path ([string]$AuditObj.output_dataset))) {
      $HorizonDataset = [string]$AuditObj.output_dataset
    }
  }
  if (-not $HorizonDataset) {
    $FallbackDataset = Join-Path $Root "data\surrogate\horizon_mpc_dataset.csv"
    if (Test-Path $FallbackDataset) { $HorizonDataset = $FallbackDataset }
  }
  if (-not $HorizonDataset -or -not (Test-Path $HorizonDataset)) {
    throw "Missing horizon dataset after build_horizon_dataset; audit=$HorizonAudit"
  }
  Write-Host "[Project6 No-control Repair MPC] horizon_dataset=$HorizonDataset"

  $TrainHorizonArgs = @(
    "scripts\43_train_horizon_surrogate.py",
    "--config", $ConfigPath,
    "--dataset", $HorizonDataset,
    "--model-kind", $HorizonModelKind,
    "--epochs", [string]$SurrogateEpochs,
    "--device", $Device,
    "--min-samples", [string]$HorizonMinSamples
  )
  if ($RunLevel -eq "smoke") { $TrainHorizonArgs += "--allow-small-dataset" }
  Invoke-PythonStep "train_horizon_surrogate" $TrainHorizonArgs

  $ValidateHorizonArgs = @(
    "scripts\44_validate_horizon_surrogate.py",
    "--config", $ConfigPath,
    "--dataset", $HorizonDataset
  )
  if ($RunLevel -eq "formal") { $ValidateHorizonArgs += "--fail-on-quality" }
  if ($RunLevel -eq "smoke") { $ValidateHorizonArgs += "--allow-quality-fail" }
  Invoke-PythonStep "validate_horizon_surrogate" $ValidateHorizonArgs

  Invoke-PythonStep "train_uncertainty_heads" @(
    "scripts\45_train_uncertainty_heads.py",
    "--config", $ConfigPath,
    "--dataset", $HorizonDataset
  )

  Invoke-PythonStep "validate_uncertainty_gate" @(
    "scripts\46_validate_uncertainty_gate.py",
    "--config", $ConfigPath,
    "--dataset", $HorizonDataset
  )

  if (-not $SkipClosedLoop) {
    $ClosedLoopEventIds = $EventIds
    if (-not $ClosedLoopEventIds) {
      $RepresentativeArgs = @(
        "scripts\62_select_representative_events.py",
        "--config", $ConfigPath,
        "--out-dir", (Join-Path $Root "outputs\design")
      )
      if ($RepresentativeEvents -gt 0) {
        $RepresentativeArgs += @("--max-events", [string]$RepresentativeEvents)
      }
      Invoke-PythonStep "select_representative_events" $RepresentativeArgs
      $RepresentativeEventIdPath = Join-Path $Root "outputs\design\representative_event_ids.txt"
      if (-not (Test-Path $RepresentativeEventIdPath)) {
        throw "Missing representative event id file: $RepresentativeEventIdPath"
      }
      $ClosedLoopEventIds = ((Get-Content $RepresentativeEventIdPath | Where-Object { $_.Trim() }) -join ",")
    }
    Write-Host "[Project6 No-control Repair MPC] closed_loop_event_ids=$ClosedLoopEventIds"

    $RunClosedLoopArgs = @(
      "scripts\08_run_closed_loop.py",
      "--config", $ConfigPath,
      "--mode", $ClosedLoopMode,
      "--run-tag", $RunTag,
      "--device", $Device,
      "--workers", [string]$Workers,
      "--proposed-workers", [string]$ProposedWorkers,
      "--proposed-controller", "generic_gat_mpc",
      "--proposed-base", "clean",
      "--event-ids", $ClosedLoopEventIds,
      "--max-steps", [string]$MaxSteps,
      "--skip-existing"
    )
    if ($BaselinePolicies.Trim()) {
      $RunClosedLoopArgs += @("--baseline-policies", $BaselinePolicies)
    }
    Invoke-PythonStep "run_closed_loop" $RunClosedLoopArgs

    if ((-not $SkipActionAudits) -and ($BenchmarkSource -eq "pystorms_scenario_beta")) {
      Invoke-PythonStep "single_actuator_ablation" @(
        "scripts\72_pystorms_beta_single_actuator_ablation.py",
        "--config", $ConfigPath,
        "--event-ids", $ClosedLoopEventIds,
        "--resume",
        "--out-dir", (Join-Path $EvalDir "single_actuator_ablation")
      )

      Invoke-PythonStep "accepted_mpc_action_audit" @(
        "scripts\73_audit_accepted_mpc_actions.py",
        "--config", $ConfigPath,
        "--run-dir", $RunDir,
        "--reference-policy", "no_control",
        "--out-dir", (Join-Path $EvalDir "accepted_action_audit")
      )
    }

    Invoke-PythonStep "recalculate_priority_metrics" @(
      "scripts\61_recalculate_project2_priority_zone_metrics.py",
      "--config", $ConfigPath,
      "--run-dir", $RunDir,
      "--out-dir", $EvalDir
    )

    if ($ObjectiveMode -eq "pfv_preserving_system_repair") {
      Invoke-PythonStep "build_no_control_repair_supervision" @(
        "scripts\74_build_no_control_repair_supervision.py",
        "--config", $ConfigPath,
        "--event-policy-metrics", (Join-Path $EvalDir "project5_priority_event_policy_metrics_main.csv"),
        "--out-dir", (Join-Path $EvalDir "no_control_repair_supervision")
      )

      $GateArgs = @(
        "scripts\75_no_control_repair_gate.py",
        "--config", $ConfigPath,
        "--event-policy-metrics", (Join-Path $EvalDir "project5_priority_event_policy_metrics_main.csv"),
        "--out-dir", $EvalDir
      )
      if (-not $AllowGateFail) {
        $GateArgs += "--fail-on-block"
      }
      Invoke-PythonStep "no_control_repair_gate" $GateArgs
    }
    else {
      $GateArgs = @(
        "scripts\63_project5_formal_gate.py",
        "--config", $ConfigPath,
        "--paired-metrics", (Join-Path $EvalDir "project5_priority_paired_metrics_main.csv"),
        "--out-dir", $EvalDir
      )
      if (-not $AllowGateFail) {
        $GateArgs += "--fail-on-block"
      }
      Invoke-PythonStep "project5_formal_gate" $GateArgs
    }
  }

  Write-Host "[Project6 No-control Repair MPC] finished"
  Write-Host "[Project6 No-control Repair MPC] run_dir=$RunDir"
  Write-Host "[Project6 No-control Repair MPC] eval_dir=$EvalDir"
}
finally {
  Pop-Location
}
