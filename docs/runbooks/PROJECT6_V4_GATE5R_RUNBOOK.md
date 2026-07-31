# Project6 V4 Gate 5R runbook

Gate 5R is the only authorised path from the failed legacy Gate 5 to an
informative H120 training dataset. It does not run full-event training,
closed-loop evaluation, calibration, policy lock, or Formal Blind.

## Frozen execution contract

- Network: `data/wuhan_v8_storage_retrofit.inp`.
- Network forcing is rainfall-only. The active INP contains no sanitary base
  inflow section or associated patterns. The byte-preserved original is
  `data/wuhan_v8_storage_retrofit_original_with_base_inflow_7ea1e133.inp`.
- Managed order: `data/project6_v8_storage_retrofit_control_enabled_ids.txt`.
- State output: 300 s; decisions: 600 s; H120: 12 decisions.
- Engineering36 only; `K <= 8`.
- Dynamic Internal: native SWMM rules from event start.
- All four branches use the same native SWMM prefix from event start.
- Candidate/No-control/Hold append time-gated priority-100 rules that are
  inactive before checkpoint; five-minute action-trace replay is forbidden.
- No-control holds Engineering36 at 1.0 after the checkpoint.
- KPI deltas are Candidate minus Reference. PFV/TFV are m3 and Peak is m3/s.
- Full-event labels remain disabled and NaN while Gate3_FULL is PARTIAL.
- Exit codes: 0 pass, 2 contract/input blocked, 3 incomplete, 4 runtime error,
  5 scientific gate failure.

Opportunity V3 uses facility flow and each facility's own upstream/downstream
head difference, plus current storage/capacity and the allowed rainfall
forecast. It does not use global topographic head spread and does not suppress
real flow/storage opportunity when rain and flooding are zero. A flat sample is
a realised single-inactive-facility action probe with no measurable response,
not a checkpoint label.

All outputs are isolated under:

```text
outputs/project6_dual_reference_v4/gate5r_informative_v3_exact_native_prefix/
```

## Tests and Gate 5R evidence

Open PowerShell and run:

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$v4Python = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe'
$v4Runner = 'E:\RTC_sewer\Project6\scripts\project6_runs\RUN_PROJECT6_V4_GATE5R.ps1'
$v4Config = 'E:\RTC_sewer\Project6\configs\wuhan_project6_v4_gate5r.yaml'

& $v4Python -m pytest -q `
  tests\test_project6_v4_dual_reference.py `
  tests\test_v4_reference_validity.py `
  tests\test_project6_v4_aug1.py `
  tests\test_v4_window_kpis.py `
  tests\test_v4_candidate_generator.py `
  tests\test_v4_schedule_projection.py `
  tests\test_v4_action_authority.py `
  tests\test_v4_opportunity_scan.py `
  tests\test_v4_gate5r_pipeline.py `
  tests\test_v4_runner_schedule.py `
  tests\test_v4_no_dwf_contract.py `
  tests\test_v4_native_rule_branch.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage AuditContracts -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage ReauditExistingGate5 -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage BuildEventInventory -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage ScanOpportunities -Config $v4Config -Workers 4 -Resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage PlanExactPrefixTiny -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage RunExactPrefixTiny -Config $v4Config -Workers 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage AuditExactPrefixTiny -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage PlanExcitationCanary -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage RunExcitationCanary -Config $v4Config -Workers 1 -Limit 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$v3Root = 'E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\gate5r_informative_v3_exact_native_prefix'
$canaryProgress = Join-Path $v3Root 'canary\runs\run_progress.json'
do {
  & $v4Runner -Stage RunExcitationCanary -Config $v4Config -Workers 4 -Resume
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  $progress = Get-Content -LiteralPath $canaryProgress -Raw | ConvertFrom-Json
  Write-Host "Canary completed=$($progress.completed_total) remaining=$($progress.remaining)"
} until ($progress.scope_complete)

& $v4Runner -Stage AuditExcitationCanary -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$anchorProgress = Join-Path $v3Root 'anchors\runs\run_progress.json'
do {
  & $v4Runner -Stage DiscoverExactAnchors -Config $v4Config -Workers 4 -Resume
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  if (-not (Test-Path -LiteralPath $anchorProgress)) { break }
  $progress = Get-Content -LiteralPath $anchorProgress -Raw | ConvertFrom-Json
  Write-Host "Anchor search completed=$($progress.completed_total) remaining=$($progress.remaining)"
} until ($progress.scope_complete)

& $v4Runner -Stage AuditExactAnchors -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage PlanPilot -Config $v4Config
exit $LASTEXITCODE
```

Stop here and inspect:

```text
stage_status/AuditExcitationCanary.json
anchors/anchor_science_audit.json
anchors/exact_anchor_manifest.csv
anchors/candidate_coverage_failure.json (only on scientific failure)
pilot/pilot_case_plan.csv
```

Do not proceed unless `AuditExactAnchors.json` has `exit_code = 0`.

## Pilot

```powershell
$pilotProgress = Join-Path $v3Root 'pilot\runs\run_progress.json'
do {
  & $v4Runner -Stage RunPilot -Config $v4Config -Workers 4 -Limit 10 -Resume
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  $progress = Get-Content -LiteralPath $pilotProgress -Raw | ConvertFrom-Json
  Write-Host "Pilot completed=$($progress.completed_total) remaining=$($progress.remaining)"
} until ($progress.scope_complete)

& $v4Runner -Stage BuildPilotDataset -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage AuditPilotDataset -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage TrainPilotBaselines -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage EvaluatePilotGate -Config $v4Config
exit $LASTEXITCODE
```

Do not proceed unless `EvaluatePilotGate.json` has `exit_code = 0`.

## Formal 1600 and informative model

```powershell
& $v4Runner -Stage PlanFormal1600 -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$formalProgress = Join-Path $v3Root 'formal1600\runs\run_progress.json'
do {
  & $v4Runner -Stage RunFormal1600 -Config $v4Config -Workers 4 -Limit 10 -Resume
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  $progress = Get-Content -LiteralPath $formalProgress -Raw | ConvertFrom-Json
  Write-Host "Formal1600 completed=$($progress.completed_total) remaining=$($progress.remaining)"
} until ($progress.scope_complete)

& $v4Runner -Stage BuildFormal1600 -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage AuditFormal1600 -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage TrainV4Informative -Config $v4Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage EvaluateV4InformativeGate -Config $v4Config
exit $LASTEXITCODE
```

No downstream stage may be started after a non-zero exit. A scientific failure
must be diagnosed from the evidence files; thresholds, K, labels, or event
partitions must not be changed merely to obtain a pass.
