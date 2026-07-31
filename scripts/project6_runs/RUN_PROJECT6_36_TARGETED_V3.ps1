[CmdletBinding()]
param(
  [switch]$AuditExisting,
  [switch]$BuildManifest,
  [switch]$RunTargeted,
  [switch]$AuditAlignment,
  [switch]$RunAlignmentFix,
  [switch]$BuildPFVSupplement,
  [switch]$RunPFVSupplement,
  [switch]$BuildV3Dataset,
  [switch]$TrainV3,
  [switch]$Review,
  [switch]$Resume,
  [string]$Python = "",
  [string]$Config = "configs\wuhan_project6_36_temporal_joint.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$Epochs = 80
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

if (-not $Python) {
  $candidates = @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    "E:\RTC_sewer\Project\env\Scripts\python.exe"
  )
  if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX "python.exe") }
  $Python = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path $Python)) {
  throw "No usable Python interpreter. Pass -Python explicitly."
}

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 targeted-v3] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

Invoke-Python "environment_preflight" @(
  "-c",
  "import sys,numpy,pandas,torch,pyswmm; print(sys.executable); print('torch',torch.__version__,'cuda',torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

$V1Dataset = "outputs\project6_36_temporal_joint_v1\effect_dataset\same_state_raw_joint_36.npz"
$V2Manifest = "outputs\project6_36_temporal_joint_v2\joint_data_plan\targeted_informative_paired_manifest.csv"
$V2Cases = "outputs\project6_36_temporal_joint_v2\paired_cases"
$AlignmentPlan = "outputs\project6_36_temporal_joint_v2\joint_data_plan_alignment_fix\targeted_alignment_correction_manifest.csv"
$AlignmentCases = "outputs\project6_36_temporal_joint_v2\paired_cases_alignment_fix"
$PFVSupplementPlan = "outputs\project6_36_temporal_joint_v2\joint_data_plan_pfv_unsafe\pfv_unsafe_supplement_manifest.csv"
$CombinedManifest = "outputs\project6_36_temporal_joint_v2\joint_data_plan_pfv_unsafe\targeted_informative_combined_240_manifest.csv"
$PFVSupplementCases = "outputs\project6_36_temporal_joint_v2\paired_cases_pfv_unsafe"
$V3Dataset = "outputs\project6_36_temporal_joint_v2\effect_dataset\same_state_raw_joint_36_v3.npz"
$V3Report = "outputs\models_temporal_joint_36_v3\raw_joint_36_same_state_v3_train_report.json"

if ($AuditExisting) {
  Invoke-Python "audit_legacy_336" @(
    "scripts\90_audit_temporal_joint_information.py",
    "--config", $Config,
    "--dataset", $V1Dataset
  )
}

if ($BuildManifest) {
  Invoke-Python "build_targeted_manifest" @(
    "scripts\91_build_targeted_informative_manifest.py",
    "--config", $Config,
    "--device", $Device,
    "--candidates-per-checkpoint", "3",
    "--max-candidate-cases", "240"
  )
  $preflight = Get-Content "outputs\project6_36_temporal_joint_v2\joint_data_plan\targeted_manifest_preflight.json" | ConvertFrom-Json
  if (-not $preflight.passed) {
    throw "Targeted manifest preflight failed; SWMM remains blocked."
  }
}

if ($RunTargeted) {
  if (-not (Test-Path $V2Manifest)) { throw "Missing targeted manifest. Run -BuildManifest first." }
  $preflight = Get-Content "outputs\project6_36_temporal_joint_v2\joint_data_plan\targeted_manifest_preflight.json" | ConvertFrom-Json
  if (-not $preflight.passed) { throw "Targeted manifest preflight is false." }
  $arguments = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $V2Manifest,
    "--reference-bank", "outputs\data_bank_train_v8_storage_variablepump\trajectories",
    "--out-dir", $V2Cases,
    "--workers", "$Workers",
    "--max-cases", "180"
  )
  if ($Resume) { $arguments += "--resume" }
  Invoke-Python "run_targeted_same_state_swmm" $arguments
}

if ($AuditAlignment) {
  Invoke-Python "audit_realized_action_alignment" @(
    "scripts\94_audit_targeted_realized_sequences.py",
    "--config", $Config,
    "--dataset", $V1Dataset,
    "--case-dir", $V2Cases,
    "--manifest", $V2Manifest,
    "--out-dir", "outputs\project6_36_temporal_joint_v2\joint_data_plan_alignment_fix",
    "--max-correction-cases", "60"
  )
}

if ($RunAlignmentFix) {
  if (-not (Test-Path $AlignmentPlan)) { throw "Missing alignment correction manifest. Run -AuditAlignment first." }
  $preflight = Get-Content "outputs\project6_36_temporal_joint_v2\joint_data_plan_alignment_fix\alignment_correction_preflight.json" | ConvertFrom-Json
  if (-not $preflight.passed) { throw "Alignment correction preflight is false." }
  $arguments = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $AlignmentPlan,
    "--reference-bank", "outputs\data_bank_train_v8_storage_variablepump\trajectories",
    "--out-dir", $AlignmentCases,
    "--workers", "$Workers",
    "--max-cases", "$($preflight.correction_candidate_cases)"
  )
  if ($Resume) { $arguments += "--resume" }
  Invoke-Python "run_alignment_corrections" $arguments
}

if ($BuildPFVSupplement) {
  Invoke-Python "build_pfv_unsafe_supplement" @(
    "scripts\95_build_pfv_unsafe_supplement_manifest.py",
    "--config", $Config,
    "--base-manifest", $V2Manifest,
    "--out-dir", "outputs\project6_36_temporal_joint_v2\joint_data_plan_pfv_unsafe",
    "--max-candidate-cases", "60"
  )
}

if ($RunPFVSupplement) {
  if (-not (Test-Path $PFVSupplementPlan)) { throw "Missing PFV supplement manifest. Run -BuildPFVSupplement first." }
  $preflight = Get-Content "outputs\project6_36_temporal_joint_v2\joint_data_plan_pfv_unsafe\pfv_unsafe_supplement_preflight.json" | ConvertFrom-Json
  if (-not $preflight.passed) { throw "PFV supplement preflight is false." }
  $arguments = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $PFVSupplementPlan,
    "--reference-bank", "outputs\data_bank_train_v8_storage_variablepump\trajectories",
    "--out-dir", $PFVSupplementCases,
    "--workers", "$Workers",
    "--max-cases", "60"
  )
  if ($Resume) { $arguments += "--resume" }
  Invoke-Python "run_pfv_unsafe_supplement" $arguments
}

if ($BuildV3Dataset) {
  $datasetManifest = if (Test-Path $CombinedManifest) { $CombinedManifest } else { $V2Manifest }
  $arguments = @(
    "scripts\92_build_targeted_v3_dataset.py",
    "--config", $Config,
    "--old-dataset", $V1Dataset,
    "--case-dir", $V2Cases,
    "--manifest", $datasetManifest,
    "--out-dir", "outputs\project6_36_temporal_joint_v2\effect_dataset",
    "--noop-fraction", "0.06"
  )
  if (Test-Path "$AlignmentCases\paired_candidate_results.csv") {
    $arguments += @("--correction-case-dir", $AlignmentCases)
  }
  if (Test-Path "$PFVSupplementCases\paired_candidate_results.csv") {
    $arguments += @("--extra-case-dir", $PFVSupplementCases)
  }
  Invoke-Python "build_v3_dataset" $arguments
}

if ($TrainV3) {
  if (-not (Test-Path $V3Dataset)) { throw "Missing v3 dataset. Run -BuildV3Dataset first." }
  Invoke-Python "train_v3_heads" @(
    "scripts\93_train_raw_joint_action_surrogate_v3.py",
    "--config", $Config,
    "--dataset", $V3Dataset,
    "--warm-start", "outputs\models_temporal_joint_36\raw_joint_36_same_state_v2.pt",
    "--epochs", "$Epochs",
    "--batch-size", "16",
    "--device", $Device,
    "--out-dir", "outputs\models_temporal_joint_36_v3",
    "--model-name", "raw_joint_36_same_state_v3.pt"
  )
}

if ($Review) {
  if (-not (Test-Path $V3Report)) { throw "Missing v3 report. Run -TrainV3 first." }
  $report = Get-Content $V3Report | ConvertFrom-Json
  Write-Host "[Project6 targeted-v3] validation_gate_passed=$($report.validation_gate_passed)"
  Write-Host "[Project6 targeted-v3] acceptance=$($report.acceptance)"
  if (-not $report.validation_gate_passed) {
    $report.validation_gate_failures | ForEach-Object {
      Write-Host "[Project6 targeted-v3] failed=$($_.check) reason=$($_.reason)"
    }
  } else {
    Write-Host "[Project6 targeted-v3] Gate passed, but Smoke remains intentionally manual and blocked pending report review."
  }
}

if (-not ($AuditExisting -or $BuildManifest -or $RunTargeted -or $AuditAlignment -or $RunAlignmentFix -or $BuildPFVSupplement -or $RunPFVSupplement -or $BuildV3Dataset -or $TrainV3 -or $Review)) {
  Write-Host "Select: -AuditExisting -BuildManifest -RunTargeted -AuditAlignment -RunAlignmentFix -BuildPFVSupplement -RunPFVSupplement -BuildV3Dataset -TrainV3 -Review. Use -Resume with SWMM stages."
}
