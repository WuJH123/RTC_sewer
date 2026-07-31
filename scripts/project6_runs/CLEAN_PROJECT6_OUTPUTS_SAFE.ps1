param(
  [string]$Root = "E:\RTC_sewer\Project6\outputs",
  [switch]$Apply,
  [switch]$RemoveSwmmOutRpt,
  [switch]$RemoveGeneratedInp,
  [switch]$IncludeDetails
)

$ErrorActionPreference = "Stop"

function Assert-UnderRoot {
  param(
    [string]$Path,
    [string]$RootPath
  )
  $resolved = (Resolve-Path -LiteralPath $Path).Path
  if (-not $resolved.StartsWith($RootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to operate outside root: $resolved"
  }
  return $resolved
}

function New-ManifestPath {
  param(
    [string]$CleanupDir,
    [string]$Name
  )
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  return (Join-Path $CleanupDir "$Name`_$stamp.csv")
}

function Summarize-Files {
  param([object[]]$Files)
  $bytes = ($Files | Measure-Object Length -Sum).Sum
  [pscustomobject]@{
    Files = $Files.Count
    GB = [math]::Round(($bytes / 1GB), 3)
  }
}

function Summarize-Dirs {
  param([object[]]$Dirs)
  $rows = @()
  foreach ($d in $Dirs) {
    $files = Get-ChildItem -LiteralPath $d.FullName -Recurse -File -ErrorAction SilentlyContinue
    $bytes = ($files | Measure-Object Length -Sum).Sum
    $rows += [pscustomobject]@{
      Path = $d.FullName
      Files = $files.Count
      GB = [math]::Round(($bytes / 1GB), 3)
    }
  }
  return $rows
}

$RootPath = Assert-UnderRoot -Path $Root -RootPath (Resolve-Path -LiteralPath $Root).Path
$CleanupDir = Join-Path $RootPath "_cleanup_manifests"
New-Item -ItemType Directory -Force -Path $CleanupDir | Out-Null

$plan = [ordered]@{
  root = $RootPath
  apply = [bool]$Apply
  remove_swmm_out_rpt = [bool]$RemoveSwmmOutRpt
  remove_generated_inp = [bool]$RemoveGeneratedInp
  include_details = [bool]$IncludeDetails
  cleanup_manifest_dir = $CleanupDir
}

if ($RemoveSwmmOutRpt) {
  $files = @(Get-ChildItem -LiteralPath $RootPath -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
      $_.FullName.StartsWith($RootPath, [System.StringComparison]::OrdinalIgnoreCase) -and
      ($_.Extension.ToLowerInvariant() -in @(".out", ".rpt"))
    })
  $manifest = New-ManifestPath -CleanupDir $CleanupDir -Name "swmm_out_rpt"
  $files | Select-Object FullName,Length,LastWriteTime | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
  $plan.swmm_out_rpt = Summarize-Files -Files $files
  $plan.swmm_out_rpt_manifest = $manifest

  if ($Apply) {
    foreach ($f in $files) {
      if (-not $f.FullName.StartsWith($RootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside root: $($f.FullName)"
      }
      Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
    }
  }
}

if ($RemoveGeneratedInp) {
  $dirNames = @("case_inp", "event_inp")
  if ($IncludeDetails) {
    $dirNames += "details"
  }

  $dirs = @(Get-ChildItem -LiteralPath $RootPath -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $dirNames -contains $_.Name -and $_.FullName.StartsWith($RootPath, [System.StringComparison]::OrdinalIgnoreCase) })

  $dirSummary = @(Summarize-Dirs -Dirs $dirs | Sort-Object GB -Descending)
  $manifest = New-ManifestPath -CleanupDir $CleanupDir -Name "generated_dirs"
  $dirSummary | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
  $plan.generated_dirs = $dirSummary
  $plan.generated_dirs_manifest = $manifest

  if ($Apply) {
    foreach ($d in $dirs) {
      $resolved = Assert-UnderRoot -Path $d.FullName -RootPath $RootPath
      if ($dirNames -notcontains (Split-Path -Leaf $resolved)) {
        throw "Refusing unexpected directory name: $resolved"
      }
      Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
}

$plan.free_gb_now = [math]::Round(((Get-PSDrive -Name E).Free / 1GB), 2)
$plan | ConvertTo-Json -Depth 6
