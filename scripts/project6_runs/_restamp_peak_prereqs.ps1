$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
# Silence benign numpy RuntimeWarnings (e.g. overflow in the logistic sigmoid,
# which correctly saturates to 0/1). The frozen Runner uses ErrorActionPreference
# Stop and would otherwise convert any native stderr line into a terminating
# error before reporting the real stage exit code.
$env:PYTHONWARNINGS = 'ignore'
$Runner = 'E:\RTC_sewer\Project6\scripts\project6_runs\RUN_PROJECT6_V4_FINAL.ps1'
$Cfg    = 'E:\RTC_sewer\Project6\configs\wuhan_project6_v4_final.yaml'

# Re-stamp the RunPeakBoundary prerequisite chain after code changes.
# ScanOpportunityPool resumes and reuses existing opportunity SWMM (0 new runs).
$plan = @(
    @{ Stage = 'AuditContracts';             Resume = $false },
    @{ Stage = 'BuildEventInventory';        Resume = $false },
    @{ Stage = 'PlanOpportunityPool';        Resume = $false },
    @{ Stage = 'ScanOpportunityPool';        Resume = $true  },
    @{ Stage = 'BuildOpportunityPool';       Resume = $false },
    @{ Stage = 'AuditOpportunityCoverage';   Resume = $false },
    @{ Stage = 'BuildPeakCandidateCatalog';  Resume = $false },
    @{ Stage = 'PlanPeakBoundary';           Resume = $false },
    @{ Stage = 'AuditPeakBoundaryPreflight'; Resume = $false }
)

foreach ($step in $plan) {
    if ($step.Resume) {
        & $Runner -Stage $step.Stage -Config $Cfg -Workers 16 -Resume
    } else {
        & $Runner -Stage $step.Stage -Config $Cfg -Workers 16
    }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "RESTAMP_STOP stage=$($step.Stage) exit=$code"
        exit $code
    }
}
Write-Host "RESTAMP_DONE all_prereqs_exit=0"
exit 0
