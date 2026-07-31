param(
    [Parameter(Mandatory=$true)]
    [string]$Stage,
    [string]$Python = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe',
    [string]$Config = 'E:\RTC_sewer\Project6\configs\wuhan_project6_v4_final.yaml',
    [ValidateRange(1, 16)]
    [int]$Workers = 16,
    [ValidateRange(0, 1000000)]
    [int]$Limit = 0,
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'E:\RTC_sewer\Project6'
$EntryPoint = Join-Path $ProjectRoot 'scripts\project6_v4_final.py'

foreach ($requiredPath in @($Python, $Config, $EntryPoint)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        Write-Error "Required Final V4 path not found: $requiredPath"
        exit 2
    }
}

$arguments = @(
    $EntryPoint,
    '--stage', $Stage,
    '--config', $Config,
    '--workers', $Workers,
    '--limit', $Limit
)
if ($Resume) { $arguments += '--resume' }
if ($RetryFailed) { $arguments += '--retry-failed' }
if ($DryRun) { $arguments += '--dry-run' }

Write-Host "[Project6 V4 Final] stage=$Stage workers=$Workers limit=$Limit resume=$Resume retryFailed=$RetryFailed dryRun=$DryRun"
& $Python @arguments
$code = $LASTEXITCODE
Write-Host "[Project6 V4 Final] stage=$Stage exit=$code"
exit $code
