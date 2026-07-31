param(
    [ValidateSet("DryRun", "Apply")]
    [string]$Mode = "DryRun",
    [string]$Manifest = "E:\RTC_sewer\Project6\cleanup_manifests\outputs_cleanup_candidates_20260728.csv",
    [string[]]$Categories = @("A_swmm_out_rpt", "D_hotstart_hsf", "B_generated_case_inp")
)

# Manifest-driven cleanup executor for Step 1 (A + D) and Step 2 (B).
# Deletes ONLY the exact paths listed in the candidate manifest whose
# category is in $Categories. Mirrors the Assert-InRoot guard from the
# existing project scripts and adds a defense-in-depth protected-zone check.
# DryRun (default) deletes nothing; Apply performs deletion.

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath "E:\RTC_sewer\Project6\outputs").Path.TrimEnd("\")
$rootPrefix = $root + "\"

# Segments that must never be touched (defense in depth; manifest already excludes them)
$protectedSegments = @(
    "frozen_evidence", "audits", "references", "dataset", "dataset_v2",
    "dataset_v3", "evaluation", "map", "legacy_oracle", "planning",
    "inventory", "contracts", "golden_v4", "_cleanup_manifests"
)

function Assert-InRoot([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside outputs root: $full"
    }
    return $full
}

function Test-Protected([string]$FullPath) {
    $segs = $FullPath.ToLowerInvariant().Split('\')
    foreach ($s in $protectedSegments) {
        if ($segs -contains $s.ToLowerInvariant()) { return $true }
    }
    return $false
}

if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Manifest not found: $Manifest"
}

$rows = @(Import-Csv -LiteralPath $Manifest | Where-Object { $Categories -contains $_.category })

$driveBefore = (Get-PSDrive -Name E).Free
$stats = @{}
foreach ($c in $Categories) {
    $stats[$c] = [ordered]@{ present_files = 0; present_bytes = [int64]0; missing_files = 0; deleted_files = 0; deleted_bytes = [int64]0 }
}
$protectedHits = 0

foreach ($row in $rows) {
    $full = Assert-InRoot $row.path
    if (Test-Protected $full) {
        $protectedHits++
        Write-Warning "SKIP protected: $full"
        continue
    }
    $cat = $row.category
    if (Test-Path -LiteralPath $full) {
        $len = (Get-Item -LiteralPath $full).Length
        $stats[$cat].present_files++
        $stats[$cat].present_bytes += [int64]$len
        if ($Mode -eq "Apply") {
            Remove-Item -LiteralPath $full -Force
            $stats[$cat].deleted_files++
            $stats[$cat].deleted_bytes += [int64]$len
        }
    }
    else {
        $stats[$cat].missing_files++
    }
}

$driveAfter = (Get-PSDrive -Name E).Free
$totalPresentBytes = [int64]0
$totalDeletedBytes = [int64]0
foreach ($c in $Categories) {
    $totalPresentBytes += [int64]$stats[$c].present_bytes
    $totalDeletedBytes += [int64]$stats[$c].deleted_bytes
}

$out = [ordered]@{
    mode = $Mode
    manifest = $Manifest
    categories = $Categories
    protected_hits = $protectedHits
    per_category = $stats
    total_present_gb = [math]::Round($totalPresentBytes / 1GB, 3)
    total_deleted_gb = [math]::Round($totalDeletedBytes / 1GB, 3)
    free_before_gb = [math]::Round($driveBefore / 1GB, 2)
    free_after_gb = [math]::Round($driveAfter / 1GB, 2)
    observed_free_gain_gb = [math]::Round(($driveAfter - $driveBefore) / 1GB, 2)
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$statePath = Join-Path "E:\RTC_sewer\Project6\cleanup_manifests" "outputs_cleanup_exec_$($Mode)_$stamp.json"
$out | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
$out | ConvertTo-Json -Depth 6
"STATE_WRITTEN=$statePath"
