$ErrorActionPreference = 'Stop'
$root = 'e:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\final_v4'
Write-Host '--- pilot dirs ---'
Get-ChildItem -Path (Join-Path $root 'pilot') -Directory | ForEach-Object { Write-Host $_.Name }
Write-Host '--- planning ---'
Get-ChildItem (Join-Path $root 'pilot\planning') -File | ForEach-Object { Write-Host ($_.Name + '  ' + $_.Length) }
Write-Host '--- dataset ---'
Get-ChildItem (Join-Path $root 'pilot\dataset') -File | ForEach-Object { Write-Host ($_.Name + '  ' + $_.Length) }
Write-Host '--- runs top-level ---'
Get-ChildItem (Join-Path $root 'pilot\runs') | Select-Object -First 8 | ForEach-Object { Write-Host $_.Name }
Write-Host '--- runs size ---'
$s = Get-ChildItem (Join-Path $root 'pilot\runs') -Recurse -File | Measure-Object Length -Sum
Write-Host ('runs files=' + $s.Count + ' bytes=' + $s.Sum)
Write-Host '--- references dir? ---'
if (Test-Path (Join-Path $root 'pilot\references')) {
  $r = Get-ChildItem (Join-Path $root 'pilot\references') -Recurse -File | Measure-Object Length -Sum
  Write-Host ('references files=' + $r.Count + ' bytes=' + $r.Sum)
} else {
  Write-Host 'no pilot\references at top level'
}
Write-Host '--- stage_status pilot ---'
Get-ChildItem (Join-Path $root 'audits\stage_status') -Filter '*Pilot*' -File | ForEach-Object { Write-Host ($_.Name + '  ' + $_.Length) }
Write-Host '--- frozen_evidence existing ---'
if (Test-Path (Join-Path $root 'audits\frozen_evidence')) {
  Get-ChildItem (Join-Path $root 'audits\frozen_evidence') -Directory | ForEach-Object { Write-Host $_.Name }
}
Write-Host '--- disk free ---'
$d = Get-PSDrive -Name E
Write-Host ('E free GB=' + [math]::Round($d.Free/1GB,1))
