[CmdletBinding()]
param(
  [switch]$Smoke,
  [switch]$Formal8,
  [switch]$Formal20,
  [switch]$Evaluate,
  [switch]$Resume,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_actual_repair_v1.yaml",
  [string]$RunTag = "project6_36_actual_repair_v1",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$ProposedWorkers = 4,
  [string]$EventIds = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 36 actual-repair] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

Invoke-Python "environment_preflight" @(
  "-c", "import sys, torch, pandas, numpy; print(sys.executable); print(torch.__version__, torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

$SmokeEvents = if ($EventIds) {
  $EventIds
} else {
  "T5_D75_block,T20_D150_chicago_center,T50_D210_chicago_center,T100_D240_block"
}
$Formal8Events = if ($EventIds) {
  $EventIds
} else {
  "T5_D75_block,T10_D105_chicago_early,T20_D150_chicago_center,T30_D240_chicago_early,T50_D210_chicago_center,T50_D300_double_peak,T75_D210_chicago_center,T100_D240_block"
}
$Formal20Events = if ($EventIds) {
  $EventIds
} else {
  "T5_D75_block,T5_D105_chicago_center,T10_D105_chicago_early,T10_D150_chicago_center,T20_D150_chicago_center,T20_D210_chicago_center,T20_D300_double_peak,T30_D75_chicago_early,T30_D240_chicago_early,T30_D300_chicago_early,T50_D105_block,T50_D210_chicago_center,T50_D300_double_peak,T75_D210_chicago_center,T75_D300_chicago_late,T100_D105_block,T100_D150_chicago_center,T100_D240_block,T100_D300_chicago_late,T50_D300_chicago_center"
}

function Run-ClosedLoop([string]$Tag, [string]$Events) {
  $args = @(
    "scripts\08_run_closed_loop.py",
    "--config", $Config,
    "--mode", "formal",
    "--run-tag", $Tag,
    "--device", $Device,
    "--workers", "$Workers",
    "--proposed-workers", "$ProposedWorkers",
    "--event-ids", $Events,
    "--proposed-controller", "temporal_joint_36",
    "--proposed-base", "clean",
    "--baseline-policies", "no_control,internal_rules,efd_storage_priority,auto_rbc"
  )
  if ($Resume) { $args += "--skip-existing" }
  Invoke-Python "run_closed_loop_$Tag" $args
}

if ($Smoke) {
  Run-ClosedLoop "${RunTag}_smoke" $SmokeEvents
}

if ($Formal8) {
  Run-ClosedLoop "${RunTag}_formal8" $Formal8Events
}

if ($Formal20) {
  Run-ClosedLoop "${RunTag}_formal20" $Formal20Events
}

if ($Evaluate) {
  foreach ($Suffix in @("smoke", "formal8", "formal20")) {
    $runDir = "outputs\closed_loop_paired_no_controls\formal\${RunTag}_$Suffix"
    if (Test-Path $runDir) {
      Invoke-Python "recalculate_metrics_$Suffix" @(
        "scripts\61_recalculate_project2_priority_zone_metrics.py",
        "--config", $Config,
        "--run-dir", $runDir,
        "--out-dir", "outputs\evaluation_${RunTag}_$Suffix"
      )
      Invoke-Python "accepted_action_audit_$Suffix" @(
        "scripts\73_audit_accepted_mpc_actions.py",
        "--config", $Config,
        "--run-dir", $runDir,
        "--reference-policy", "no_control"
      )
    }
  }
}

if (-not ($Smoke -or $Formal8 -or $Formal20 -or $Evaluate)) {
  Write-Host "Select -Smoke, -Formal8, -Formal20, and/or -Evaluate. Use -Resume to skip existing event files."
}
