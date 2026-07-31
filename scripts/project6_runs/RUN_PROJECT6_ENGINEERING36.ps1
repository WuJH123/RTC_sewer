param(
  [string]$Root = "E:\RTC_sewer\Project6",
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_engineering36.yaml",
  [switch]$InitContract,
  [switch]$AuditContract,
  [switch]$PlanSameState,
  [switch]$FormalBlindLock
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $Root

function Run-Step {
  param([string]$Label, [string[]]$StepArgs)
  Write-Host "[Project6 Engineering36] step=$Label"
  & $Python @StepArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

function Require-Contract {
  $manifest = Join-Path $Root "outputs\project6_engineering36\frozen\contract_manifest.json"
  if (!(Test-Path $manifest)) {
    throw "Missing Engineering36 contract manifest. Run -InitContract first."
  }
  $m = Get-Content $manifest -Raw | ConvertFrom-Json
  if (-not [bool]$m.passed) {
    throw "Engineering36 contract did not pass. Fix Stage 0 before continuing."
  }
  return $m
}

if ($InitContract) {
  Run-Step "init_engineering36_contract" @(
    "scripts/124_init_project6_engineering36.py",
    "--config", $Config
  )
}

if ($AuditContract) {
  $m = Require-Contract
  Write-Host "[Project6 Engineering36] contract=passed"
  Write-Host "[Project6 Engineering36] frozen_dir=$($m.frozen_dir)"
  Write-Host "[Project6 Engineering36] action_ids_sha256=$($m.action_ids_sha256)"
  Write-Host "[Project6 Engineering36] event_split_counts=$($m.event_split_counts | ConvertTo-Json -Compress)"
}

if ($PlanSameState) {
  Require-Contract | Out-Null
  throw @"
Stage 1 intentionally stops here until the new same-state generator is run.
Required next implementation target:
- No-control vs Core26 same-state cases
- Core26 vs Core26+Residual10 same-state cases
- PFV budget boundary, TFV/peak contrast, H30/H120 reversal strata
- exact 36-action hash from outputs/project6_engineering36/frozen/contract_manifest.json
Do not reuse trajectory files unless they match the frozen action hash and same retrofit INP.
"@
}

if ($FormalBlindLock) {
  $m = Require-Contract
  $lock = Join-Path $Root "outputs\project6_engineering36\frozen\formal_blind.lock.json"
  if (Test-Path $lock) {
    throw "FormalBlind is already locked: $lock"
  }
  $payload = [ordered]@{
    locked_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    config = $Config
    contract_manifest = "outputs/project6_engineering36/frozen/contract_manifest.json"
    action_ids_sha256 = $m.action_ids_sha256
    event_split_sha256 = $m.artifact_hashes.'event_split.csv'
    formal_blind_policy = "No rerun after unblinding; retraining must create a new frozen contract and new blind split."
  }
  $payload | ConvertTo-Json -Depth 8 | Set-Content -Path $lock -Encoding UTF8
  Write-Host "[Project6 Engineering36] FormalBlind locked: $lock"
}
