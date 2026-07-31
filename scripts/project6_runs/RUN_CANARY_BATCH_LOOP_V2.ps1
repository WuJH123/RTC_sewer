# Batch-loop driver for Gate 5R V2 canary: takes the next pending batch each
# iteration until scope_complete, mirroring the runbook loop with 16 workers.
$ErrorActionPreference = 'Stop'

$v4Runner = 'E:\RTC_sewer\Project6\scripts\project6_runs\RUN_PROJECT6_V4_GATE5R.ps1'
$v4Config = 'E:\RTC_sewer\Project6\configs\wuhan_project6_v4_gate5r.yaml'
$canaryProgress = 'E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\gate5r_informative_v2_no_dwf\canary\runs\run_progress.json'

do {
    & $v4Runner -Stage RunExcitationCanary `
        -Config $v4Config -Workers 16 -Limit 16 -Resume
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $progress = Get-Content -LiteralPath $canaryProgress -Raw |
        ConvertFrom-Json

    Write-Host "Canary completed=$($progress.completed_total) remaining=$($progress.remaining)"
} until ($progress.scope_complete)

Write-Host "Canary batches finished: scope_complete=$($progress.scope_complete)"
exit 0
