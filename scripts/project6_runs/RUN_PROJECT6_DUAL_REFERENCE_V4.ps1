param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('BuildV4Dataset','TrainV4','EvaluateV4ModelGate','BuildRunnerConfigV4','AuditV4Readiness','RunClosedLoopSmokeV4','EvaluateClosedLoopSmokeV4','DiagnoseV4FullEventPFVGate','PlanV4DualReferenceFullEventCases','GenerateV4DualReferenceFullEventCases','BuildV4AugmentedDataset','TrainV4Aug1','EvaluateV4Aug1ModelGate')]
    [string]$Stage,
    [string]$Python = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe',
    [string]$Config = 'E:\RTC_sewer\Project6\configs\wuhan_project6_dual_reference_v4.yaml',
    [int]$MaxEvents = 3,
    [int]$Workers = 3,
    [int]$MaxSamples = 0,
    [int]$EnsembleSize = 5,
    [switch]$Smoke,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'E:\RTC_sewer\Project6'
$Script = Join-Path $ProjectRoot 'scripts\205_prompt3_v4.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Config)) { throw "Config not found: $Config" }
if (-not (Test-Path -LiteralPath $Script)) { throw "V4 stage script not found: $Script" }

$arguments = @(
    $Script,
    '--config', $Config,
    '--stage', $Stage,
    '--max-events', $MaxEvents,
    '--workers', $Workers,
    '--max-samples', $MaxSamples,
    '--ensemble-size', $EnsembleSize
)
if ($Smoke) { $arguments += '--smoke' }
if ($Resume) { $arguments += '--resume' }

Write-Host "[Project6 V4 dual-reference] stage=$Stage"
# Intentionally no short PowerShell timeout. The Python V4 orchestrator uses
# heartbeat + stall detection + retry and writes event-level results for Resume.
& $Python @arguments
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "[Project6 V4 dual-reference] stage=$Stage exit=$code"
}
exit $code
