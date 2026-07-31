# Project6 PFV-First Dual-Fallback 10 min V3 Contract

## Scope

This contract defines the V3 recovery line for `E:\RTC_sewer\Project6`. It
replaces the deleted v8/Core26/Residual10 output-dependent workflow with a
single-network, PFV-first, dual-fallback, same-state learned-effect workflow.

All formal evidence must be produced on:

`E:\RTC_sewer\Project6\data\wuhan_v8_storage_retrofit.inp`

## Control Timing

- `control_interval_min = 10`
- `prediction_horizon_min = 120`
- `prediction_horizon_steps = 12`
- `free_residual_action_min = 30`
- `free_residual_steps = 3`
- execute only `action_seq[0, :]`
- re-observe and re-optimize every control step

The controller manages all 36 facilities. A setting such as
`max_simultaneous_residual_overrides = 8` limits the number of facilities that
may deviate from the anchor in one control step; it does not reduce the managed
facility set to eight.

The 10 min control interval is not the SWMM hydraulic KPI integration step. KPI
integration must use the true timestamp differences from the simulated or
observed series.

## Baseline Roles

### Internal SWMM Rules

Internal SWMM rules are the primary PFV effectiveness benchmark. Candidate
actions must first satisfy:

`LCB(PFV_Internal - PFV_candidate) >= minimum_internal_improvement`

PFV-active online states must be defined from predicted Internal PFV, not from
near-zero No-control PFV.

### Selected Safe Fallback

The online fallback is selected independently before learned candidates are
scored. Candidate actions must also satisfy:

`LCB(PFV_selected_fallback - PFV_candidate) > 0`

and must not worsen TFV or peak relative to the selected fallback.

### No-Control

No-control is a diagnostic physical reference for explaining native-rule target
misalignment. It is not the primary PFV benchmark and must not be used as an
oracle fallback.

## Dual-Fallback Contract

Fallback candidates:

1. Executable Passive Fallback.
2. Internal SWMM Rules Fallback.

Executable Passive is a minimal per-facility intervention policy. It may change
zero, one, or a small coordinated set of facilities only when required for
storage interlock, pump dwell, upstream/downstream headroom, or safety.

Forbidden passive behavior:

- set all 36 facilities to zero
- set all 36 facilities fully open
- instant reset of all facilities
- ignoring pump dwell
- ignoring storage inlet/outlet interlock

Internal fallback must remove learned overrides and verify that original
`[CONTROLS]` can retake authority. It must not keep previous learned actions.

Fallback choice may use only:

- current sensor/GAT state
- current device feedback
- operational rainfall forecast
- frozen prediction model or physically valid online rollout

It must not use true future SWMM results.

Fallback priority when both are safe:

1. lower TFV conservative upper bound
2. lower peak conservative upper bound
3. lower sentinel/storage risk
4. lower PFV conservative upper bound
5. lower transition action cost

`selected_fallback_id` is frozen before learned candidate evaluation.

## Action Semantics

The model and labels must use `actual_executed_action_sequence`, not requested,
raw, projected, or target-only sequences.

Each decision must preserve these fields:

- native setting
- anchor setting
- requested residual
- projected setting
- target setting
- actual current setting
- actual executed setting
- override TTL
- release flag
- clipping flag
- rate-limit flag
- dwell status
- interlock status

The following counts are distinct and must be logged:

- native/anchor setting changes
- residual override count
- actual executed change count

## Same-State Contract

Same-state counterfactuals must restore:

- node/link/storage hydraulic state
- current facility setting
- pump on/off duration
- dwell remaining time
- override TTL
- previous action
- fallback mode
- continuation policy

State-clone equivalence tests must compare at least PFV, TFV, peak, selected
facility settings, storage levels, and key priority/sentinel depths under a
no-op continuation.

## Forecast Contract

Main Formal uses operational forecasts only. Perfect future rainfall may be run
only as an upper-bound diagnostic.

Each forecast record must include:

- forecast issue time
- forecast valid time
- forecast source
- forecast version
- forecast horizon
- forecast scenario id
- timezone
- rainfall units
- spatial mode
- update interval
- maximum forecast age
- minimum required horizon
- `truth_available_to_controller = false`

Required stress forecasts for design/calibration diagnostics:

- intensity -20%
- intensity +20%
- peak early
- peak late
- missed second peak
- false peak

Train, calibration, and formal forecast records must pass schema, unit,
timezone, horizon, and no-truth-to-controller consistency checks.

## Continuation Policy Contract

Every H30/H60/H90/H120 label must store a `continuation_policy_id`.

Allowed types:

- `one_step_action_advantage`
- `first_30min_plan_value`
- `fixed_anchor_continuation`
- `true_receding_horizon_closed_loop_value`

Fixed-branch labels must not be substituted for real rolling Shadow, Smoke, or
Formal validation.

## Information Coverage Contract

Data generation target is information coverage, not a fixed sample count. Each
case must serve at least one purpose:

- fill coverage gap
- increase independent event support
- cover facility/direction/phase/magnitude/duration
- calibrate PFV/TFV/peak boundary
- repair false-safe
- validate H30/H120 reversal
- validate fallback
- validate low-support action likely to be selected by MPC
- validate optimizer exploitation

Coverage states:

- `missing`
- `sufficient`
- `structural_infeasible`
- `low_response`

Pre-run filtering must remove:

- no-op
- illegal action
- near duplicate
- candidate that neither fills a gap nor was selected by active learning

Batches should contain 250-500 effective candidates.

Minimum support targets are defined in the V3 YAML config and must report
sample, checkpoint, event, and storm-family counts.

Coverage cells must include event, storm family, split, checkpoint, phase, state
risk cluster, anchor type, facility or hydraulic group, direction, magnitude,
duration, concurrency, interaction type, unique event support, feasibility
status, outcome class, and decision relevance. Global target lists alone are
not a completed coverage plan.

## OOD and Optimizer Exploitation

Candidate scoring must include:

- state distance
- action-sequence distance
- joint support or density
- ensemble disagreement
- OOD penalty
- OOD rejection

## Primary GAT Selection

The primary GAT candidate for Prompt 2 is user-confirmed as `sr0p15`.

This decision is recorded in:

`docs/contracts/gat_primary_selection_decision.json`

The selection rationale is a sensor-cost / reconstruction-quality compromise.
It must not be overridden by automatic ranking, even if `sr0p20` or `sr0p30`
has a higher full-network reconstruction NSE.

Before the selection is operational, the user must run `SelectPrimaryGAT` with
`-GATRegistryName "sr0p15"` and `-AcknowledgeSelection`. The stage must verify:

- checkpoint path and SHA256;
- strict load report;
- full node mapping;
- directed edge-set consistency;
- 134 mapped sensors;
- model class provenance;
- unchanged source-report hashes.

Successful selection creates only:

`outputs/project6_pfvfirst_dualfallback_10min_v3/gat/gat_primary_selection_lock.json`

It does not imply:

- GAT robustness passed;
- runtime state features were generated;
- Round 0 may run.

Prompt 2 completion still requires sr0p15 robustness diagnostics and at least
one real seven-frame runtime state build with shape, causality and missingness
audits.

Top-k SWMM shadow validation must report:

- predicted rank
- realized rank
- realized regret
- model-favorite but realized-worst actions
- optimizer exploitation examples

## KPI Contract

PFV, TFV, and peak_TFV_rate definitions are implemented by:

- `sewerrtc/evaluation/kpi_contract.py`
- `docs/contracts/kpi_contract.json`

Data generation, labeling, MPC, gates, and Formal scripts must call this shared
implementation rather than duplicating definitions.

PFV uses a frozen priority-node list and hash. TFV uses the full fixed INP node
set with non-flooding nodes filled as zero for every branch. Peak remains the
maximum over time of the full-network instantaneous flooding-rate sum.

## Formal and Statistics Contract

Formal events must be blind to:

- GAT training
- action data generation
- Calibration-A
- threshold setting
- manual design iteration

Formal provenance must scan Project4-Project6 and compare event id, rainfall
time-series hash, storm family, scaling relationship, and time-shifted near
duplicates.

Statistical unit is event. If storm-family dependence exists, cluster by storm
family. Candidate rows or checkpoints must not be bootstrapped as independent
samples.

Pre-register:

- primary estimand
- absolute and relative effect
- mean or median
- event-level paired bootstrap
- bootstrap count and seed
- TFV/peak non-inferiority tests
- zero-baseline handling
- secondary endpoint reporting rules

Formal leakage scans must compare Project2 through Project6 event ids, rainfall
time-series hashes, storm-family labels, scaling relationships, and time-shift
near duplicates. Renamed but hydrologically identical storms are not blind.

## Backup Reachability Contract

`backup_reachable_after_action = true` means that after executing the first
10 min action, at least one legal Passive or Internal fallback trajectory exists
over the remaining 120 min prediction horizon and over forecast stress
scenarios required by `forecast_contract.json`. PASS requires:

- facility bounds satisfied
- pump dwell satisfied
- storage interlock satisfied
- PFV/TFV/peak no worse than the selected fallback margins
- no terminal storage/sentinel/downstream recovery violation

If the forecast horizon is insufficient, learned candidates are disabled and
only safe fallback may be selected.

## Time Synchronization Contract

Every closed-loop row must record:

- measurement_time
- state_estimation_time
- forecast_issue_time
- decision_time
- command_time
- actuator_effective_time
- feedback_time

No data generated after `decision_time` may be used in that decision.

## SWMM Numerical Quality Gate

Each event must record:

- runoff continuity error
- flow-routing continuity error
- routing-step configuration
- hot-start restoration error
- repeated-run difference

Non-inferiority tolerances must not be below the numerical noise floor.
## Stage Status Repair Addendum

- Current runnable stages are limited to `Status`, `Audit`, and `InitCoverageSchema`.
- `InitCoverageSchema` replaces the previous ambiguous coverage planning scaffold. It may return success only as `status=scaffold_only`, must not create `_COMPLETED.json`, and must not unlock downstream stages.
- `BuildEventCatalog`, `BuildCheckpointCatalog`, and `RunInternalPFVOpportunityScan` are defined as required future stages but remain disabled until their provenance, state-clone and baseline-trajectory prerequisites are implemented.
- `BuildDataset`, `TrainPilot`, and `MinimalGate` are disabled until real same-state data generation, model training, and model gate implementations exist.
- All disabled stages must write execution status with `status=disabled`, `failure_reason=not_implemented`, `completion_marker=null`, return non-zero, and must not create `_COMPLETED.json`.
- A real completed stage may write `_COMPLETED.json` only after upstream marker checks, config hash checks, input hash checks, expected-output existence checks, and output hash calculation pass.
- Audit string matches in `[CONTROLS]` are only `preliminary_text_reference_count`; they are not native-rule behavior evidence.
- Exit code contract: `0` success, `2` disabled, `3` blocked upstream/stale marker, `4` runtime exception, `5` real gate failure, `6` config/contract/hash mismatch, `7` CLI or stage selection error.
- Sentinel contract is unresolved until independent provenance and thresholds are verified. Unresolved sentinel status blocks FatalAudit, AuditFallbacks, StateCloneTest, MPC and Smoke.
- Pump semantics for `add350.1`, `ADD301.2`, and `ADD301.3` remain unresolved until independent INP/source/engineering evidence confirms binary or variable-speed behavior and dwell constraints.
## V3 Step 1 GAT And Augmented-State Contract

Project4 sparse-sensor GAT checkpoints are read-only external assets. Their
presence and SHA256 hash do not imply compatibility. A GAT checkpoint may be
used for formal Project6 state reconstruction only after compatibility is
classified as `compatible_strict` or after an explicitly approved
`compatible_shared_base_graph_only` workflow.

Supported compatibility states are:

- `compatible_strict`: node set, node order, input features, normalization,
  graph signature, model structure, and checkpoint loading all match.
- `compatible_shared_base_graph_only`: the checkpoint covers a shared base graph
  but does not cover all retrofit nodes. Missing nodes must be supplied by
  direct SWMM state, sensor data, or independent physical features. GAT output
  must not be fabricated for missing nodes.
- `metadata_incomplete`: the checkpoint can be registered but lacks node order,
  sensor mask, normalization, graph signature, or other critical metadata.
- `incompatible`: node order, dimensions, normalization, graph, or critical
  node mapping cannot be reconciled.
- `load_failed`: the checkpoint cannot be safely loaded or the source file is
  absent.

Before human selection:

- `selected_primary_gat = null`;
- `selection_status = human_selection_required`;
- no stage may claim that the formal state pipeline is frozen.

The action-effect model receives a strictly causal 60 min state history with
seven frames: current, -10, -20, -30, -40, -50, and -60 min. Each frame must
record measurement time, source time, state-estimation time, decision time, data
age, missingness, quality flag, and the causal aggregation method. Observations
after the decision time, future interpolation, true future state gap filling, and
using forecast valid time as observation time are forbidden.

`add350.1` is user-confirmed as a variable-speed pump. This confirmation is
authoritative but does not by itself authorize learned residual control. Until
INP, pump curve, native rule, and engineering bounds are audited, residual
candidate generation for `add350.1` remains blocked. `ADD301.2` and `ADD301.3`
remain strict binary pumps with action set `{0, 1}`.
