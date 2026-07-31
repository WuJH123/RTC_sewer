$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
$Py = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe'
$Cfg = 'configs\wuhan_project6_v4_final.yaml'
for ($i = 0; $i -lt 60; $i++) {
    $out = & $Py scripts\project6_v4_final.py --stage RunPilotFeasibilityMap --config $Cfg --workers 16 --resume 2>&1 | Out-String
    $j = $out | ConvertFrom-Json
    Write-Host ("iter=$i completed=" + $j.completed + " remaining=" + $j.remaining + " failed=" + $j.evidence.failed + " blocked=" + $j.evidence.resource_blocked)
    if ($j.remaining -eq 0) { Write-Host 'ROUND_A_COMPLETE'; break }
    if ($j.evidence.failed -gt 0) { Write-Host 'RUN_FAILED'; break }
    if ($j.exit_code -ne 3 -and $j.exit_code -ne 0) { Write-Host ("UNEXPECTED_EXIT=" + $j.exit_code); break }
}
