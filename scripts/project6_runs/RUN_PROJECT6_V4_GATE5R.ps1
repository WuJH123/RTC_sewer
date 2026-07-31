param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        'AuditContracts',
        'ReauditExistingGate5',
        'BuildEventInventory',
        'ScanOpportunities',
        'PlanExactPrefixTiny',
        'RunExactPrefixTiny',
        'AuditExactPrefixTiny',
        'PlanExcitationCanary',
        'RunExcitationCanary',
        'AuditExcitationCanary',
        'DiscoverExactAnchors',
        'AuditExactAnchors',
        'PlanPilot',
        'RunPilot',
        'BuildPilotDataset',
        'AuditPilotDataset',
        'TrainPilotBaselines',
        'EvaluatePilotGate',
        'PlanFormal1600',
        'RunFormal1600',
        'BuildFormal1600',
        'AuditFormal1600',
        'TrainV4Informative',
        'EvaluateV4InformativeGate'
    )]
    [string]$Stage,
    [string]$Python = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe',
    [string]$Config = 'E:\RTC_sewer\Project6\configs\wuhan_project6_v4_gate5r.yaml',
    [ValidateRange(1, 16)]
    [int]$Workers = 16,
    [ValidateRange(0, 100000)]
    [int]$Limit = 0,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'E:\RTC_sewer\Project6'
$EntryPoint = Join-Path $ProjectRoot 'scripts\248_v4_gate5r.py'

foreach ($requiredPath in @($Python, $Config, $EntryPoint)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required Gate 5R path not found: $requiredPath"
    }
}

$arguments = @(
    $EntryPoint,
    '--config', $Config,
    '--stage', $Stage,
    '--workers', $Workers,
    '--limit', $Limit
)
if ($Resume) { $arguments += '--resume' }

Write-Host "[Project6 V4 Gate5R] stage=$Stage workers=$Workers limit=$Limit resume=$Resume"
& $Python @arguments
$code = $LASTEXITCODE
Write-Host "[Project6 V4 Gate5R] stage=$Stage exit=$code"
exit $code
