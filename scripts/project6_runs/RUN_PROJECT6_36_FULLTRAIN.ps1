param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6_v8_storage_36.yaml",
  [ValidateSet("Audit", "ReusePlan", "CanonicalizeActionOrder", "PrepareJointDataPlan")]
  [string[]]$Stage = @("Audit"),
  [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root
if (-not $Python) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }

function Invoke-Stage([string]$Name) {
  if ($Name -in @("CanonicalizeActionOrder", "PrepareJointDataPlan")) {
    $args = @("scripts\86_canonicalize_36_action_order_and_plan.py", "--config", $Config, "--stage", $Name)
  } else {
    $args = @("scripts\85_audit_36_fulltrain_reuse.py", "--config", $Config, "--stage", $Name)
  }
  if ($Force) { $args += "--force" }
  & $Python @args
  if ($LASTEXITCODE -ne 0) { throw "Stage failed [$Name] with exit code $LASTEXITCODE" }
}

# Stages are deliberately independent.  ReusePlan will reject execution until
# Audit is complete; this runner never invokes a downstream stage implicitly.
foreach ($item in $Stage) { Invoke-Stage $item }
