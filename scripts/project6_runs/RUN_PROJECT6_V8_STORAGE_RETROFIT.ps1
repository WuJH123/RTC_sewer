param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_v8_storage.yaml",
  [string]$RunTag = "project6_v8_storage_T5_T100_v1",
  [string[]]$Stage = @("Audit", "BuildScenario"),
  [string]$Device = "cuda",
  [int]$Workers = 8,
  [int]$ProposedWorkers = 1,
  [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python executable not found: $Python" }
$allowedStages = @("Audit", "BuildScenario", "CheckReuse", "FineTuneIfNeeded", "Smoke", "Calibration", "Formal35", "Formal70", "Evaluate", "FullFormal")
$Stage = @($Stage | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$invalidStages = @($Stage | Where-Object { $_ -notin $allowedStages })
if ($invalidStages.Count -gt 0) { throw "Unknown stage(s): $($invalidStages -join ', ')" }
if ($Stage -contains "FullFormal") {
  $Stage = @("Audit", "BuildScenario", "CheckReuse", "Smoke", "Calibration", "Formal35", "Evaluate")
}

function Invoke-Py([string]$Label, [string[]]$PythonArgs) {
  Write-Host "[Project6 v8 storage] step=$Label"
  & $Python @PythonArgs
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

function Get-EventIds([string]$Path) {
  if (-not (Test-Path $Path)) { throw "Event table not found: $Path. Run -Stage BuildScenario first." }
  return ((Import-Csv $Path).event_id -join ",")
}

$scenarioDir = Join-Path $Root "outputs\storage_retrofit\$RunTag"
$runDir = Join-Path $Root "outputs\closed_loop_paired_no_controls\formal\$RunTag"
$evalDir = Join-Path $Root "outputs\evaluation_$RunTag"

foreach ($current in $Stage) {
  switch ($current) {
    "Audit" {
      Invoke-Py "audit_inp" @("scripts\00_audit_inp.py", "--config", $Config)
    }
    "BuildScenario" {
      Invoke-Py "generate_rainfall_library" @("scripts\01_generate_rainfall_library.py", "--config", $Config, "--mode", "formal")
      Invoke-Py "select_priority_and_sensors" @("scripts\02_select_priority_and_sensors.py", "--config", $Config)
      Invoke-Py "build_influence_domains" @("scripts\49_build_influence_domains.py", "--config", $Config)
      Invoke-Py "build_storage_retrofit_scenario" @("scripts\84_build_v8_storage_retrofit_scenario.py", "--config", $Config, "--run-tag", $RunTag)
    }
    "CheckReuse" {
      $required = @(
        "outputs\models_all109\gat_sr0p10.pt",
        "outputs\models_all109\horizon_temporal_gnn.pt",
        "outputs\cache_all109\transition_cache.npz",
        "outputs\storage_retrofit\$RunTag\scenario_manifest.json"
      )
      $missing = @($required | Where-Object { -not (Test-Path (Join-Path $Root $_)) })
      if ($missing.Count -gt 0) { throw "Reuse preflight failed. Missing: $($missing -join '; ')" }
      Get-FileHash ($required | ForEach-Object { Join-Path $Root $_ }) -Algorithm SHA256 | Format-Table Path, Hash -AutoSize
    }
    "FineTuneIfNeeded" {
      $cache = Join-Path $Root "outputs\cache_v8_storage\transition_cache.npz"
      if (-not (Test-Path $cache)) {
        throw "No isolated retrofit cache exists. Do not retrain yet: generate only approved targeted SWMM cases, then rebuild the cache."
      }
      Invoke-Py "finetune_gat" @("scripts\05_train_gat.py", "--config", $Config, "--epochs", "10", "--device", $Device, "--init-checkpoint", "outputs\models_all109\gat_sr0p10.pt")
    }
    "Smoke" {
      $events = "T5_D75_chicago_center,T100_D300_double_peak"
      $args = @("scripts\08_run_closed_loop.py", "--config", $Config, "--mode", "formal", "--run-tag", "$RunTag`_smoke", "--device", $Device, "--workers", "$Workers", "--proposed-workers", "$ProposedWorkers", "--event-ids", $events, "--proposed-controller", "generic_gat_mpc")
      if ($Resume) { $args += "--skip-existing" }
      Invoke-Py "closed_loop_smoke" $args
    }
    "Calibration" {
      $events = Get-EventIds (Join-Path $scenarioDir "calibration_events.csv")
      $args = @("scripts\08_run_closed_loop.py", "--config", $Config, "--mode", "formal", "--run-tag", "$RunTag`_calibration", "--device", $Device, "--workers", "$Workers", "--proposed-workers", "$ProposedWorkers", "--event-ids", $events, "--proposed-controller", "generic_gat_mpc")
      if ($Resume) { $args += "--skip-existing" }
      Invoke-Py "closed_loop_calibration" $args
    }
    "Formal35" {
      $events = Get-EventIds (Join-Path $scenarioDir "formal35_events.csv")
      $args = @("scripts\08_run_closed_loop.py", "--config", $Config, "--mode", "formal", "--run-tag", $RunTag, "--device", $Device, "--workers", "$Workers", "--proposed-workers", "$ProposedWorkers", "--event-ids", $events, "--proposed-controller", "generic_gat_mpc")
      if ($Resume) { $args += "--skip-existing" }
      Invoke-Py "closed_loop_formal35" $args
    }
    "Formal70" {
      Invoke-Py "build_extended_event_split" @("scripts\84_build_v8_storage_retrofit_scenario.py", "--config", $Config, "--run-tag", $RunTag, "--extended")
      $events = Get-EventIds (Join-Path $scenarioDir "formal70_events.csv")
      $args = @("scripts\08_run_closed_loop.py", "--config", $Config, "--mode", "formal", "--run-tag", "$RunTag`_formal70", "--device", $Device, "--workers", "$Workers", "--proposed-workers", "$ProposedWorkers", "--event-ids", $events, "--proposed-controller", "generic_gat_mpc")
      if ($Resume) { $args += "--skip-existing" }
      Invoke-Py "closed_loop_formal70" $args
    }
    "Evaluate" {
      if (-not (Test-Path $runDir)) { throw "Formal run directory not found: $runDir" }
      Invoke-Py "recalculate_priority_metrics" @("scripts\61_recalculate_project2_priority_zone_metrics.py", "--config", $Config, "--run-dir", $runDir, "--out-dir", $evalDir)
      $metrics = Get-ChildItem $evalDir -Filter "*event_policy_metrics_main.csv" | Select-Object -First 1
      if ($null -eq $metrics) { throw "No main event-policy metrics found in $evalDir" }
      Invoke-Py "formal_gate" @("scripts\75_no_control_repair_gate.py", "--config", $Config, "--event-policy-metrics", $metrics.FullName, "--out-dir", $evalDir)
    }
  }
}
