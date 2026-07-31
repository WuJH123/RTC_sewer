# Project6 V4 Final pipeline runbook

This runbook is the sole Final-V4 execution route. The active network is the
frozen rainfall-only file:

```text
E:\RTC_sewer\Project6\data\wuhan_v8_storage_retrofit.inp
```

The runner never removes, scales, or edits DWF at runtime. `AuditContracts`
fails closed if active `[DWF]` FLOW rows appear or either frozen network SHA
changes.

## Environment and tests

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$v4Python = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe'
$v4Runner = 'E:\RTC_sewer\Project6\scripts\project6_runs\RUN_PROJECT6_V4_FINAL.ps1'
$v4Config = 'E:\RTC_sewer\Project6\configs\wuhan_project6_v4_final.yaml'

& $v4Python -m compileall -q `
  sewerrtc\v4 `
  scripts\project6_v4_final.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Python -m pytest -q `
  tests\test_v4_final_contract.py `
  tests\test_v4_no_dwf_network.py `
  tests\test_v4_parallel_runtime.py `
  tests\test_v4_window_kpis.py `
  tests\test_v4_candidate_generator.py `
  tests\test_v4_schedule_projection.py `
  tests\test_v4_action_authority.py `
  tests\test_v4_opportunity.py `
  tests\test_v4_opportunity_chain.py `
  tests\test_v4_peak_boundary.py `
  tests\test_v4_event_usage_ledger.py `
  tests\test_v4_checkpoint_catalog_lineage.py `
  tests\test_v4_pilot_plan.py `
  tests\test_v4_train1600_plan.py `
  tests\test_v4_dataset_manifest.py `
  tests\test_v4_event_split.py `
  tests\test_v4_reference_cache.py `
  tests\test_v4_active_learning.py `
  tests\test_v4_training.py `
  tests\test_v4_closed_loop.py `
  tests\test_v4_policy_lock.py `
  tests\test_v4_formal_blind.py `
  tests\test_v4_reporting.py `
  tests\test_v4_resume.py `
  tests\test_v4_final_labels.py `
  tests\test_v4_final_stage_registry.py `
  tests\test_v4_partial_audit.py `
  tests\test_v4_progressive_release.py `
  tests\test_v4_stratified_scheduler.py `
  tests\test_v4_candidate_budget.py `
  tests\test_v4_train_round_rotation.py `
  tests\test_v4_state_replenishment.py `
  tests\test_v4_event_replacement.py `
  tests\test_v4_preflight.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $v4Runner -Stage AuditContracts -Config $v4Config -DryRun
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

`AuditContracts` is a real static audit even when `-DryRun` is supplied. It
must report:

```text
network_variant=rainfall_only_no_dwf
active_dwf_flow_rows=0
network SHA match=true
physical network SHA match=true
exit_code=0
scope_complete=true
```

## Invocation contract

All stages accept:

```text
-Stage <name>
-Config <path>
-Workers 1..16
-Limit N
-Resume
-RetryFailed
-DryRun
```

Resume always reads the full plan, excludes valid completions, and only then
applies `Limit`. Without `-RetryFailed`, failed cases remain terminal evidence.
With `-RetryFailed`, only failed cases become pending again.

Use this fail-fast helper:

```powershell
function Invoke-V4Stage {
  param(
    [Parameter(Mandatory=$true)][string]$Stage,
    [int]$Workers = 16,
    [int]$Limit = 0,
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$DryRun
  )
  & $v4Runner `
    -Stage $Stage `
    -Config $v4Config `
    -Workers $Workers `
    -Limit $Limit `
    -Resume:$Resume `
    -RetryFailed:$RetryFailed `
    -DryRun:$DryRun
  if ($LASTEXITCODE -ne 0) {
    throw "Final V4 stage failed: $Stage exit=$LASTEXITCODE"
  }
}
```

## Future execution sequence

The following blocks are commands for future user-authorised execution. They
were not run during Final-pipeline implementation.

### Opportunity and Peak Boundary

Lineage status 2026-07-27: the Opportunity chain up to and including
`AuditOpportunityCoverage` has been re-run and passed (exit=0) after the
lineage recovery. Canonical artifacts now live only in:

```text
outputs\project6_dual_reference_v4\final_v4\opportunities\
    opportunity_pool.csv
    event_tier_catalog.csv
    standard_checkpoint_catalog.csv   (182 events, 910 rows)
    short_event_checkpoint_catalog.csv (62 short_2/short_3 events)
    opportunity_coverage_audit.json
outputs\project6_dual_reference_v4\final_v4\inventory\
    event_usage_ledger.csv            (244 events, all opportunity_scanned)
```

Execution resumes at `BuildPeakCandidateCatalog`. The commented Opportunity
stages below only need re-running if code/config SHAs change again.

```powershell
# Already re-run and passed on 2026-07-27 (repeat only after SHA changes):
# Invoke-V4Stage -Stage AuditContracts
# Invoke-V4Stage -Stage BuildEventInventory
# Invoke-V4Stage -Stage PlanOpportunityPool
# Invoke-V4Stage -Stage ScanOpportunityPool -Workers 16 -Resume
# Invoke-V4Stage -Stage BuildOpportunityPool
# Invoke-V4Stage -Stage AuditOpportunityCoverage

Invoke-V4Stage -Stage BuildPeakCandidateCatalog
Invoke-V4Stage -Stage PlanPeakBoundary
Invoke-V4Stage -Stage AuditPeakBoundaryPreflight

# Level 0: exactly one real case, then the incremental partial audit.
Invoke-V4Stage -Stage RunPeakBoundary -Workers 1 -Limit 1 -Resume
Invoke-V4Stage -Stage BuildPeakBoundaryPartial
Invoke-V4Stage -Stage AuditPeakBoundaryPartial

# Level 1: 16 more cases only after the single-case partial gate passed.
Invoke-V4Stage -Stage RunPeakBoundary -Workers 16 -Limit 16 -Resume
Invoke-V4Stage -Stage BuildPeakBoundaryPartial
Invoke-V4Stage -Stage AuditPeakBoundaryPartial

# Full scope only after the 16-case partial gate passed.
Invoke-V4Stage -Stage RunPeakBoundary -Workers 16 -Resume
Invoke-V4Stage -Stage BuildPeakBoundaryDataset
Invoke-V4Stage -Stage AuditPeakBoundary
```

A partial-gate pass is never a full-gate pass: `BuildPeakBoundaryDataset`
and `AuditPeakBoundary` still require the complete scope. Any hard
authenticity violation in a partial audit returns non-zero and stops all
further scale-up.

Stop unless the Peak audit proves at least 3 events, 6 checkpoints, 30-60
Peak-degraded actual-unique samples, at least 10 PFV-safe Peak hard negatives,
and at least 2 candidate families. If no degraded samples exist after the
frozen search, preserve `peak_constraint_binding_audit.json`; never delete the
Peak constraint automatically.

### Pilot400

```powershell
Invoke-V4Stage -Stage ClassifyExistingGate5R
Invoke-V4Stage -Stage PlanPilot400
Invoke-V4Stage -Stage AuditPilotPlan
Invoke-V4Stage -Stage AuditPilotPreflight

# Level 0: one real case.
Invoke-V4Stage -Stage RunPilot400 -Workers 1 -Limit 1 -Resume
Invoke-V4Stage -Stage BuildPilotPartial
Invoke-V4Stage -Stage AuditPilotPartial

# Level 1: up to 16 cumulative cases.
Invoke-V4Stage -Stage RunPilot400 -Workers 16 -Limit 15 -Resume
Invoke-V4Stage -Stage BuildPilotPartial
Invoke-V4Stage -Stage AuditPilotPartial

# Level 2: up to 40 cumulative cases; the stratified scheduler makes the
# first 40 span all 8 pilot events and the 40 state groups.
Invoke-V4Stage -Stage RunPilot400 -Workers 16 -Limit 24 -Resume
Invoke-V4Stage -Stage BuildPilotPartial
Invoke-V4Stage -Stage AuditPilotPartial

# Level 3: run the remaining primary cases only after the 40-case
# partial gate passed.
Invoke-V4Stage -Stage RunPilot400 -Workers 16 -Resume
Invoke-V4Stage -Stage BuildPilotDataset
Invoke-V4Stage -Stage AuditPilotDataset

# Reserve replenishment (only if a responsive state has fewer than 6
# accepted actual-unique samples): plan the reserve queue from
# audit_pilot_state_progress / plan_pilot_reserve, run the reserve cases
# with -Resume, then rebuild and re-audit the dataset.
# Reserve rows are actual-unique, never change the event split, and stop
# at max_candidate_budget_per_state=15; a state that still falls short is
# marked state_shortfall.

Invoke-V4Stage -Stage TrainPilotBaselines
Invoke-V4Stage -Stage EvaluatePilotGate
```

The pilot gate judges real accepted informative samples
(`min_accepted_informative_total: 300`), never planned row counts. Pilot
failure blocks Train1600.

### Train1600 active-learning rounds

`PlanTrain1600` fail-closes unless `pilot/evaluation/pilot_gate_verdict.json`
reports `scientific_pass=true` and `exit_code=0`. It writes the frozen split
(48 Train / 8 Calibration / 8 Locked Validation / 16 Reserve) into the event
usage ledger and emits:

```text
train1600\planning\train_checkpoint_catalog.csv   (64 events x 5 = 320 rows)
train1600\planning\reserve_checkpoint_catalog.csv (16 events x 5 = 80 rows)
train1600\planning\train1600_target_plan.csv      (1600 targets)
train1600\round0\plan.csv                         (400 Round0 cases)
```

If fewer than 80 usable standard_4plus events remain it blocks and writes
`train1600\planning\event_shortfall_report.json`; never pad with short
events, pilot events, or duplicated checkpoints.

```powershell
Invoke-V4Stage -Stage PlanTrain1600
Invoke-V4Stage -Stage AuditTrain1600Plan

# Round 0: preflight first, then one case, then partial gates before scale.
Invoke-V4Stage -Stage AuditTrainRound0Preflight
Invoke-V4Stage -Stage RunTrainRound0 -Workers 1 -Limit 1 -Resume
Invoke-V4Stage -Stage BuildTrainRound0Partial
Invoke-V4Stage -Stage AuditTrainRound0Partial
Invoke-V4Stage -Stage RunTrainRound0 -Workers 16 -Limit 15 -Resume
Invoke-V4Stage -Stage BuildTrainRound0Partial
Invoke-V4Stage -Stage AuditTrainRound0Partial
# Then repeat with -Limit 64 batches, running BuildTrainRound0Partial and
# AuditTrainRound0Partial after every 64 completed cases, until the round
# scope is complete:
Invoke-V4Stage -Stage RunTrainRound0 -Workers 16 -Limit 64 -Resume
Invoke-V4Stage -Stage BuildTrainRound0Partial
Invoke-V4Stage -Stage AuditTrainRound0Partial
Invoke-V4Stage -Stage AuditTrainRound0
Invoke-V4Stage -Stage TrainActiveLearner0

# Rounds 1-3 follow the identical pattern; the selector first filters out
# states that already reached 5 accepted actual-unique samples, exhausted
# budgets, duplicate actual candidates, completed cases, and stale-SHA plans.
Invoke-V4Stage -Stage SelectTrainRound1
Invoke-V4Stage -Stage AuditTrainRound1Preflight
Invoke-V4Stage -Stage RunTrainRound1 -Workers 1 -Limit 1 -Resume
Invoke-V4Stage -Stage BuildTrainRound1Partial
Invoke-V4Stage -Stage AuditTrainRound1Partial
Invoke-V4Stage -Stage RunTrainRound1 -Workers 16 -Limit 64 -Resume
Invoke-V4Stage -Stage BuildTrainRound1Partial
Invoke-V4Stage -Stage AuditTrainRound1Partial
Invoke-V4Stage -Stage AuditTrainRound1
Invoke-V4Stage -Stage TrainActiveLearner1

Invoke-V4Stage -Stage SelectTrainRound2
Invoke-V4Stage -Stage AuditTrainRound2Preflight
Invoke-V4Stage -Stage RunTrainRound2 -Workers 1 -Limit 1 -Resume
Invoke-V4Stage -Stage BuildTrainRound2Partial
Invoke-V4Stage -Stage AuditTrainRound2Partial
Invoke-V4Stage -Stage RunTrainRound2 -Workers 16 -Limit 64 -Resume
Invoke-V4Stage -Stage BuildTrainRound2Partial
Invoke-V4Stage -Stage AuditTrainRound2Partial
Invoke-V4Stage -Stage AuditTrainRound2
Invoke-V4Stage -Stage TrainActiveLearner2

Invoke-V4Stage -Stage SelectTrainRound3
Invoke-V4Stage -Stage AuditTrainRound3Preflight
Invoke-V4Stage -Stage RunTrainRound3 -Workers 1 -Limit 1 -Resume
Invoke-V4Stage -Stage BuildTrainRound3Partial
Invoke-V4Stage -Stage AuditTrainRound3Partial
Invoke-V4Stage -Stage RunTrainRound3 -Workers 16 -Limit 64 -Resume
Invoke-V4Stage -Stage BuildTrainRound3Partial
Invoke-V4Stage -Stage AuditTrainRound3Partial
Invoke-V4Stage -Stage AuditTrainRound3

Invoke-V4Stage -Stage BuildTrain1600Dataset
Invoke-V4Stage -Stage AuditTrain1600Dataset
```

Accepted targets and candidate budgets never mix: each state targets 5
accepted actual-unique samples with an initial budget of 6 and a maximum
of 10 candidates. The four rounds each target 400 accepted (320 basics +
a disjoint extra-80 rotation; 4 x 320 + 4 x 80 = 1600). Replenishment
after a rejection, no-op, or actual duplicate draws in the fixed order
state reserve candidate, new candidate family, boundary/uncertainty/
coverage-gap candidate, and never copies an actual schedule, counts a
reference or no-op, crosses states or splits, or lowers a KPI gate. A
state still short at maximum budget is marked `state_shortfall`, its
event becomes `event_shortfall`, the event's data stays auxiliary only,
and the formal 1600 table replaces the whole event (all 5 checkpoints)
with a same-split reserve event -- never a single checkpoint.

### Model, exact closed loop, and surrogate ablation

```powershell
Invoke-V4Stage -Stage TrainV4
Invoke-V4Stage -Stage CalibrateV4
Invoke-V4Stage -Stage EvaluateV4Locked

Invoke-V4Stage -Stage PlanExactClosedLoop
Invoke-V4Stage -Stage RunExactClosedLoop -Workers 16 -Resume
Invoke-V4Stage -Stage AuditExactClosedLoop

Invoke-V4Stage -Stage PlanSurrogateClosedLoop
Invoke-V4Stage -Stage RunSurrogateClosedLoop -Workers 16 -Resume
Invoke-V4Stage -Stage AuditSurrogateClosedLoop
```

The surrogate stages remain blocked unless the exact-SWMM closed-loop audit
passes.

### Policy Lock, Challenge, and Formal Blind

```powershell
Invoke-V4Stage -Stage LockPolicy
Invoke-V4Stage -Stage RunChallenge -Workers 16 -Resume
Invoke-V4Stage -Stage AuditChallenge

Invoke-V4Stage -Stage BuildFormalBlindInventory
Invoke-V4Stage -Stage RunFormalBlind -Workers 16 -Resume
Invoke-V4Stage -Stage AuditFormalBlind
```

Challenge events cannot be reused after a failed locked version. Formal Blind
must contain at least 24 newly frozen events and may not be filtered after
results are revealed.

### Paper evidence bundle

```powershell
Invoke-V4Stage -Stage BuildPaperResults
Invoke-V4Stage -Stage BuildPaperFigures
Invoke-V4Stage -Stage BuildPaperTables
Invoke-V4Stage -Stage BuildReproducibilityBundle
```

These stages export evidence, tables, and ordinary academic figures only. They
do not generate unsupported scientific conclusions.

## Dry-run inspection

To inspect any long stage without starting simulations or training:

```powershell
& $v4Runner `
  -Stage RunPeakBoundary `
  -Config $v4Config `
  -Workers 16 `
  -Limit 10 `
  -Resume `
  -DryRun
```

A long-stage DryRun must return non-zero incomplete status, with
`long_task_not_started=true` and `scope_complete=false`. A DryRun is never a
scientific pass.

## Failure evidence

On failure, retain:

```text
outputs/project6_dual_reference_v4/final_v4/audits/stage_status/<Stage>.json
outputs/project6_dual_reference_v4/final_v4/logs/
outputs/project6_dual_reference_v4/final_v4/heartbeats/
the case completion.json files
the failed/rejected manifest
the relevant plan and contract SHA files
```

Do not delete partial evidence, change thresholds, expand K, or reuse a cache
whose config/input/contract SHA does not match.
