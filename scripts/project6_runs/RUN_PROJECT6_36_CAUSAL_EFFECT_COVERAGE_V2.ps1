[CmdletBinding()]
param(
  [switch]$PlanCoverage,
  [switch]$DryRunPaired,
  [switch]$RunPaired,
  [switch]$BuildDataset,
  [switch]$AuditDataset,
  [switch]$PlanSafetyBoundary,
  [switch]$DryRunSafetyBoundary,
  [switch]$RunSafetyBoundary,
  [switch]$BuildSafetyBoundaryDataset,
  [switch]$PlanSafetyBoundaryRound2,
  [switch]$DryRunSafetyBoundaryRound2,
  [switch]$RunSafetyBoundaryRound2,
  [switch]$BuildSafetyBoundaryRound2Dataset,
  [switch]$TrainEffect,
  [switch]$Gate,
  [switch]$Resume,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_causal_effect_coverage_v2.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [int]$Workers = 16,
  [int]$MaxCandidateCases = 800,
  [int]$MinTrainEventsPerCell = 5,
  [int]$MinValidationEventsPerCell = 3,
  [string]$MagnitudeLevels = "0.05,0.10,0.20",
  [int]$ProfilesPerContext = 2,
  [double]$JointCaseFraction = 0.15,
  [int]$MaxSafetyBoundaryCases = 72,
  [int]$MaxSafetyBoundaryRound2Cases = 32,
  [int]$EffectEpochs = 120,
  [int]$EffectBatchSize = 64
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
if (-not (Test-Path $Python)) { throw "Python interpreter not found: $Python" }

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 causal-effect coverage-v2] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [$Label] with exit code $LASTEXITCODE" }
}

$BaseDataset = "outputs\project6_36_temporal_joint_peakfixed_v1\effect_dataset\same_state_raw_joint_36_peakfixed_v1.npz"
$RootOut = "outputs\project6_36_causal_effect_coverage_v2"
$PlanDir = "$RootOut\paired_plan"
$Manifest = "$PlanDir\targeted_causal_effect_manifest.csv"
$CaseDir = "$RootOut\paired_cases"
$CaseResults = "$CaseDir\paired_candidate_results.csv"
$CaseFailures = "$CaseDir\failures.csv"
$SupplementDir = "$RootOut\effect_dataset_supplement"
$Supplement = "$SupplementDir\same_state_raw_joint_36_causal_supplement_coverage_v2.npz"
$CombinedDir = "$RootOut\effect_dataset"
$Combined = "$CombinedDir\same_state_raw_joint_36_causal_effect_coverage_v2.npz"
$AuditDir = "$RootOut\effect_dataset_audit"
$BoundaryPlanDir = "$RootOut\safety_boundary_plan"
$BoundaryManifest = "$BoundaryPlanDir\pfv_safety_boundary_v5_manifest.csv"
$BoundaryCases = "$RootOut\safety_boundary_cases"
$BoundaryResults = "$BoundaryCases\paired_candidate_results.csv"
$BoundaryFailures = "$BoundaryCases\failures.csv"
$BoundarySupplementDir = "$RootOut\safety_boundary_effect_dataset"
$BoundarySupplement = "$BoundarySupplementDir\same_state_raw_joint_36_safety_boundary_v5.npz"
$BoundaryCombinedDir = "$RootOut\effect_dataset_boundary_v5"
$BoundaryCombined = "$BoundaryCombinedDir\same_state_raw_joint_36_causal_effect_coverage_boundary_v5.npz"
$BoundaryAuditDir = "$RootOut\effect_dataset_audit_boundary_v5"
$BoundaryAuditCsv = "$BoundaryAuditDir\paired_information_audit.csv"
$Round2PlanDir = "$RootOut\safety_boundary_round2_plan"
$Round2Manifest = "$Round2PlanDir\pfv_safety_boundary_v6_round2_manifest.csv"
$Round2Cases = "$RootOut\safety_boundary_round2_cases"
$Round2Results = "$Round2Cases\paired_candidate_results.csv"
$Round2Failures = "$Round2Cases\failures.csv"
$Round2SupplementDir = "$RootOut\safety_boundary_round2_effect_dataset"
$Round2Supplement = "$Round2SupplementDir\same_state_raw_joint_36_safety_boundary_v6_round2.npz"
$Round2CombinedDir = "$RootOut\effect_dataset_boundary_v6_round2"
$Round2Combined = "$Round2CombinedDir\same_state_raw_joint_36_causal_effect_boundary_v6_round2.npz"
$Round2AuditDir = "$RootOut\effect_dataset_audit_boundary_v6_round2"
$Round2AuditCsv = "$Round2AuditDir\paired_information_audit.csv"
$WarmStart = "outputs\models_temporal_joint_36_recovery_v2\raw_joint_36_same_state_recovery_v2.pt"
$PriorReport = "outputs\models_temporal_joint_36_recovery_v2\raw_joint_36_same_state_recovery_v2_train_report.json"
$ModelDir = "outputs\models_temporal_joint_36_causal_effect_boundary_v6_round2"
$Model = "$ModelDir\raw_joint_36_causal_effect_boundary_v6_round2.pt"
$Report = "$ModelDir\raw_joint_36_causal_effect_boundary_v6_round2_train_report.json"
$GateJson = "$RootOut\mpc_gate_preflight.json"

Invoke-Python "environment_preflight" @(
  "-c", "import torch,numpy,pandas,yaml; print(torch.__version__, torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

if ($PlanCoverage) {
  Invoke-Python "plan_feasible_causal_coverage" @(
    "scripts\103_plan_causal_effect_coverage.py",
    "--config", $Config,
    "--dataset", $BaseDataset,
    "--out-dir", $PlanDir,
    "--min-train-events", "$MinTrainEventsPerCell",
    "--min-validation-events", "$MinValidationEventsPerCell",
    "--max-candidate-cases", "$MaxCandidateCases",
    "--magnitude-levels", $MagnitudeLevels,
    "--profiles-per-context", "$ProfilesPerContext",
    "--joint-case-fraction", "$JointCaseFraction"
  )
}

if ($DryRunPaired -or $RunPaired) {
  if (-not (Test-Path $Manifest)) { throw "Missing coverage manifest. Run -PlanCoverage first: $Manifest" }
  $pairedArgs = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $Manifest,
    "--out-dir", $CaseDir,
    "--workers", "$Workers",
    "--max-cases", "$MaxCandidateCases",
    "--preflight-noop-filter"
  )
  if ($Resume) { $pairedArgs += "--resume" }
  if ($DryRunPaired) { $pairedArgs += "--dry-run" }
  Invoke-Python $(if ($DryRunPaired) { "paired_preflight" } else { "run_same_state_pairs" }) $pairedArgs
}

if ($BuildDataset) {
  if (-not (Test-Path $CaseResults)) { throw "Missing paired results. Run -RunPaired first: $CaseResults" }
  $CompletedCases = @(Import-Csv $CaseResults).Count
  if ($CompletedCases -lt $MaxCandidateCases) {
    throw "Paired dataset is incomplete: $CompletedCases/$MaxCandidateCases. Re-run -RunPaired -Resume."
  }
  if ((Test-Path $CaseFailures) -and @(Import-Csv $CaseFailures).Count -gt 0) {
    throw "Paired execution still has failures. Inspect $CaseFailures and resume before building the dataset."
  }
  Invoke-Python "build_coverage_supplement" @(
    "scripts\89_build_same_state_raw_joint_dataset.py",
    "--config", $Config,
    "--case-dir", $CaseDir,
    "--out-dir", $SupplementDir,
    "--dataset-name", "same_state_raw_joint_36_causal_supplement_coverage_v2.npz"
  )
  Invoke-Python "merge_base_and_coverage_supplement" @(
    "scripts\104_merge_same_state_effect_datasets.py",
    "--base-dataset", $BaseDataset,
    "--supplement-dataset", $Supplement,
    "--out-npz", $Combined
  )
}

if ($PlanSafetyBoundary) {
  if (-not (Test-Path $Combined)) { throw "Missing coverage dataset. Run -BuildDataset first: $Combined" }
  Invoke-Python "plan_pfv_safety_boundary_v5" @(
    "scripts\105_plan_pfv_safety_boundary_supplement.py",
    "--config", $Config,
    "--dataset", $Combined,
    "--audit", "$AuditDir\paired_information_audit.csv",
    "--out-dir", $BoundaryPlanDir,
    "--train-events", "8",
    "--validation-events", "8",
    "--target-validation-unsafe-rows", "20"
  )
}

if ($DryRunSafetyBoundary -or $RunSafetyBoundary) {
  if (-not (Test-Path $BoundaryManifest)) { throw "Missing safety-boundary manifest. Run -PlanSafetyBoundary first." }
  $boundaryArgs = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $BoundaryManifest,
    "--out-dir", $BoundaryCases,
    "--workers", "$Workers",
    "--max-cases", "$MaxSafetyBoundaryCases",
    "--preflight-noop-filter"
  )
  if ($Resume) { $boundaryArgs += "--resume" }
  if ($DryRunSafetyBoundary) { $boundaryArgs += "--dry-run" }
  Invoke-Python $(if ($DryRunSafetyBoundary) { "safety_boundary_preflight" } else { "run_safety_boundary_pairs" }) $boundaryArgs
}

if ($BuildSafetyBoundaryDataset) {
  if (-not (Test-Path $BoundaryResults)) { throw "Missing safety-boundary results. Run -RunSafetyBoundary first." }
  $CompletedBoundaryCases = @(Import-Csv $BoundaryResults).Count
  if ($CompletedBoundaryCases -lt $MaxSafetyBoundaryCases) {
    throw "Safety-boundary dataset is incomplete: $CompletedBoundaryCases/$MaxSafetyBoundaryCases. Re-run -RunSafetyBoundary -Resume."
  }
  if ((Test-Path $BoundaryFailures) -and @(Import-Csv $BoundaryFailures).Count -gt 0) {
    throw "Safety-boundary execution still has failures. Inspect $BoundaryFailures."
  }
  Invoke-Python "build_safety_boundary_supplement" @(
    "scripts\89_build_same_state_raw_joint_dataset.py",
    "--config", $Config,
    "--case-dir", $BoundaryCases,
    "--out-dir", $BoundarySupplementDir,
    "--dataset-name", "same_state_raw_joint_36_safety_boundary_v5.npz"
  )
  Invoke-Python "merge_coverage_and_safety_boundary" @(
    "scripts\104_merge_same_state_effect_datasets.py",
    "--base-dataset", $Combined,
    "--supplement-dataset", $BoundarySupplement,
    "--out-npz", $BoundaryCombined
  )
}

if ($PlanSafetyBoundaryRound2) {
  if (-not (Test-Path $BoundaryCombined)) { throw "Missing boundary-v5 dataset. Run -BuildSafetyBoundaryDataset first." }
  if (-not (Test-Path $BoundaryAuditCsv)) { throw "Missing boundary-v5 audit. Run -AuditDataset first." }
  Invoke-Python "plan_pfv_safety_boundary_v6_round2" @(
    "scripts\106_plan_pfv_safety_boundary_round2.py",
    "--config", $Config,
    "--dataset", $BoundaryCombined,
    "--audit", $BoundaryAuditCsv,
    "--selection-table", "$BoundaryPlanDir\selected_events_by_no_control_load.csv",
    "--out-dir", $Round2PlanDir,
    "--target-validation-unsafe-rows", "20"
  )
}

if ($DryRunSafetyBoundaryRound2 -or $RunSafetyBoundaryRound2) {
  if (-not (Test-Path $Round2Manifest)) { throw "Missing round-2 manifest. Run -PlanSafetyBoundaryRound2 first." }
  $round2Args = @(
    "scripts\88_generate_same_state_temporal_joint_cases.py",
    "--config", $Config,
    "--manifest", $Round2Manifest,
    "--out-dir", $Round2Cases,
    "--workers", "$Workers",
    "--max-cases", "$MaxSafetyBoundaryRound2Cases",
    "--preflight-noop-filter"
  )
  if ($Resume) { $round2Args += "--resume" }
  if ($DryRunSafetyBoundaryRound2) { $round2Args += "--dry-run" }
  Invoke-Python $(if ($DryRunSafetyBoundaryRound2) { "safety_boundary_round2_preflight" } else { "run_safety_boundary_round2_pairs" }) $round2Args
}

if ($BuildSafetyBoundaryRound2Dataset) {
  if (-not (Test-Path $Round2Results)) { throw "Missing round-2 results. Run -RunSafetyBoundaryRound2 first." }
  $CompletedRound2Cases = @(Import-Csv $Round2Results).Count
  if ($CompletedRound2Cases -lt $MaxSafetyBoundaryRound2Cases) {
    throw "Round-2 dataset is incomplete: $CompletedRound2Cases/$MaxSafetyBoundaryRound2Cases. Re-run -RunSafetyBoundaryRound2 -Resume."
  }
  if ((Test-Path $Round2Failures) -and @(Import-Csv $Round2Failures).Count -gt 0) {
    throw "Round-2 execution still has failures. Inspect $Round2Failures."
  }
  Invoke-Python "build_safety_boundary_round2_supplement" @(
    "scripts\89_build_same_state_raw_joint_dataset.py",
    "--config", $Config,
    "--case-dir", $Round2Cases,
    "--out-dir", $Round2SupplementDir,
    "--dataset-name", "same_state_raw_joint_36_safety_boundary_v6_round2.npz"
  )
  Invoke-Python "merge_boundary_v5_and_round2" @(
    "scripts\104_merge_same_state_effect_datasets.py",
    "--base-dataset", $BoundaryCombined,
    "--supplement-dataset", $Round2Supplement,
    "--out-npz", $Round2Combined
  )
}

if ($AuditDataset) {
  $AuditInput = if (Test-Path $Round2Combined) { $Round2Combined } elseif (Test-Path $BoundaryCombined) { $BoundaryCombined } else { $Combined }
  $AuditManifest = if (Test-Path $Round2Combined) { $Round2Manifest } elseif (Test-Path $BoundaryCombined) { $BoundaryManifest } else { $Manifest }
  $SelectedAuditDir = if (Test-Path $Round2Combined) { $Round2AuditDir } elseif (Test-Path $BoundaryCombined) { $BoundaryAuditDir } else { $AuditDir }
  if (-not (Test-Path $AuditInput)) { throw "Missing combined dataset. Run a dataset build stage first: $AuditInput" }
  Invoke-Python "audit_combined_causal_information" @(
    "scripts\90_audit_temporal_joint_information.py",
    "--config", $Config,
    "--dataset", $AuditInput,
    "--manifest", $AuditManifest,
    "--out-dir", $SelectedAuditDir
  )
}

if ($TrainEffect) {
  if (-not (Test-Path $Round2Combined)) {
    throw "Missing round-2 safety-boundary dataset. Run -BuildSafetyBoundaryRound2Dataset first: $Round2Combined"
  }
  if (-not (Test-Path $Round2AuditCsv)) {
    throw "Missing post-round2 audit. Run -AuditDataset before training: $Round2AuditCsv"
  }
  $BoundaryAuditRows = Import-Csv $Round2AuditCsv
  $ValidationUnsafeRows = @(
    $BoundaryAuditRows | Where-Object { $_.split -eq "validation" -and $_.PFV_noninferiority -eq "unsafe" }
  ).Count
  if ($ValidationUnsafeRows -lt 20) {
    throw "Validation PFV-unsafe support is still insufficient: $ValidationUnsafeRows/20. Do not train; design a further targeted boundary batch."
  }
  if (-not (Test-Path $WarmStart)) { throw "Missing warm-start checkpoint: $WarmStart" }
  if ($Resume -and (Test-Path $Model) -and (Test-Path $Report)) {
    Write-Host "[Project6 causal-effect coverage-v2] reuse completed model=$Model"
  } else {
    Invoke-Python "train_phase_conditioned_causal_effect" @(
      "scripts\93_train_raw_joint_action_surrogate_v3.py",
      "--config", $Config,
      "--dataset", $Round2Combined,
      "--warm-start", $WarmStart,
      "--v2-report", $PriorReport,
      "--architecture-version", "causal_phase_safety_v5",
      "--epochs", "$EffectEpochs",
      "--batch-size", "$EffectBatchSize",
      "--device", $Device,
      "--learning-rate", "0.0003",
      "--fine-tune-action-encoder",
      "--action-learning-rate-scale", "0.05",
      "--direction-loss-weight", "1.5",
      "--classification-loss-weight", "1.5",
      "--reference-loss-weight", "0.05",
      "--balanced-sampling",
      "--balanced-epoch-multiplier", "2.0",
      "--calibration-event-fraction", "0.20",
      "--uncertainty-coverage", "0.90",
      "--selection-every", "5",
      "--seed", "20260714",
      "--out-dir", $ModelDir,
      "--model-name", "raw_joint_36_causal_effect_boundary_v6_round2.pt",
      "--report-name", "raw_joint_36_causal_effect_boundary_v6_round2_train_report.json"
    )
  }
}

if ($Gate) {
  if (-not (Test-Path $Report)) { throw "Missing model report. Run -TrainEffect first: $Report" }
  & $Python "scripts\99_mpc_gate_preflight.py" --config $Config --model-report $Report --out-json $GateJson --enforce
  if ($LASTEXITCODE -eq 2) {
    Write-Host "[Project6 causal-effect coverage-v2] gate is false; rolling-horizon Smoke remains blocked."
    exit 2
  }
  if ($LASTEXITCODE -ne 0) { throw "Python step failed [strict_mpc_gate] with exit code $LASTEXITCODE" }
}

if (-not ($PlanCoverage -or $DryRunPaired -or $RunPaired -or $BuildDataset -or $AuditDataset -or $PlanSafetyBoundary -or $DryRunSafetyBoundary -or $RunSafetyBoundary -or $BuildSafetyBoundaryDataset -or $PlanSafetyBoundaryRound2 -or $DryRunSafetyBoundaryRound2 -or $RunSafetyBoundaryRound2 -or $BuildSafetyBoundaryRound2Dataset -or $TrainEffect -or $Gate)) {
  Write-Host "Select: -PlanCoverage -DryRunPaired -RunPaired -BuildDataset -PlanSafetyBoundary -DryRunSafetyBoundary -RunSafetyBoundary -BuildSafetyBoundaryDataset -PlanSafetyBoundaryRound2 -DryRunSafetyBoundaryRound2 -RunSafetyBoundaryRound2 -BuildSafetyBoundaryRound2Dataset -AuditDataset -TrainEffect -Gate"
}
