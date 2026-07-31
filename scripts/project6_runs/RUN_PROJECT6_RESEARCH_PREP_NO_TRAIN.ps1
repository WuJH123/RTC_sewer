[CmdletBinding()]
param(
  [switch]$PlanReuse,
  [switch]$Review,
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6_36_temporal_joint.yaml",
  [string]$OutDir = "outputs\research_reuse_plan",
  [string]$SensorRatios = "0.05,0.10,0.15,0.20,0.30",
  [string]$TrajectoryRoots = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

if (-not $Python) {
  $candidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    "E:\RTC_sewer\Project\env\Scripts\python.exe"
  )
  if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX "python.exe") }
  $Python = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path $Python)) {
  throw "No usable Python interpreter. Pass -Python explicitly."
}

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 research-prep no-train] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

if ($PlanReuse) {
  $arguments = @(
    "scripts\96_plan_research_data_reuse.py",
    "--config", $Config,
    "--out-dir", $OutDir,
    "--sensor-ratios", $SensorRatios
  )
  if ($TrajectoryRoots) {
    $arguments += @("--trajectory-roots", $TrajectoryRoots)
  }
  Invoke-Python "plan_historical_trajectory_reuse" $arguments
}

if ($Review) {
  $summaryPath = Join-Path $Root (Join-Path $OutDir "research_reuse_summary.json")
  if (-not (Test-Path $summaryPath)) {
    throw "Missing research reuse summary. Run -PlanReuse first."
  }
  $summary = Get-Content $summaryPath | ConvertFrom-Json
  Write-Host "[Project6 research-prep no-train] inventory_rows=$($summary.inventory_rows)"
  Write-Host "[Project6 research-prep no-train] gat_manifest_rows=$($summary.gat_manifest_rows) events=$($summary.gat_events)"
  Write-Host "[Project6 research-prep no-train] action_learning_rows=$($summary.action_learning_rows) events=$($summary.action_learning_events)"
  Write-Host "[Project6 research-prep no-train] same_state_effect_rows=$($summary.same_state_effect_rows)"
  Write-Host "[Project6 research-prep no-train] outputs=$($summary.outputs | ConvertTo-Json -Compress)"
}

if (-not ($PlanReuse -or $Review)) {
  Write-Host "Select -PlanReuse and/or -Review. This runner only audits and writes manifests; it does not train models or run SWMM."
}
