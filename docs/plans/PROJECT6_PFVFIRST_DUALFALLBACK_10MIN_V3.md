# Project6 PFV-First Dual-Fallback 10 min V3 Plan

## Current State From File-System Audit

The Project6 `outputs` directory was effectively cleared. Therefore V3 must not
depend on prior output artifacts such as v8 templates, Residual10 models, old
effect datasets, or old empirical guards.

Reusable source assets still include:

- `data/wuhan_v8_storage_retrofit.inp`
- `data/project6_v8_storage_retrofit_control_enabled_ids.txt`
- `data/project6_v8_storage_retrofit_assets.csv`
- current `configs`, `scripts`, `sewerrtc`, and `tests`
- Project4 GAT checkpoints as read-only candidates, subject to compatibility
  verification

## Modules To Reuse

- INP parsing and PySWMM execution infrastructure in `sewerrtc/simulation`
- existing closed-loop CLI pattern in `scripts/08_run_closed_loop.py`
- existing temporal-joint candidate and action utilities where they preserve
  `[H,36]` action identity and timing
- existing actuator semantics work, after validating against
  `wuhan_v8_storage_retrofit.inp`

## Modules To Replace Or Extend

1. KPI definitions: centralize under `sewerrtc/evaluation/kpi_contract.py`.
2. GAT compatibility audit: add explicit hash/node/sensor/normalization checks.
3. Fallback selection: implement independent passive/internal fallback scoring.
4. Learned candidate scoring: compare candidate against both Internal and the
   selected fallback.
5. Same-state data generation: store actual executed action sequence and full
   controller memory.
6. Coverage planner: generate candidates by information gaps, not fixed sample
   counts.
7. OOD and shadow validation: reject unsupported candidate exploitation.

## V3 Stages

### Stage 0: Static Audit

Inputs:

- V3 config
- retrofit INP
- 36 facility list
- Project4 GAT candidate paths

Outputs:

- asset manifest
- GAT compatibility manifest
- actuator semantics manifest
- missing artifact report

Stop if:

- INP missing
- 36 facility ids not found
- GAT compatibility cannot be verified
- KPI contract cannot be loaded

### Stage 1: State Clone And Fallback Semantics

Inputs:

- retrofit INP
- selected events for clone tests
- fallback definitions

Outputs:

- state-clone equivalence report
- passive fallback legality report
- internal fallback control-release report

Stop if:

- same-state clone differs beyond numerical tolerance
- passive fallback changes all 36 facilities
- internal fallback leaves learned overrides active

### Stage 2: Coverage Planning

Inputs:

- actuator semantics
- priority nodes
- Internal predicted PFV-active events
- coverage targets

Outputs:

- batch candidate manifest
- coverage gap report
- duplicate/no-op/illegal action filter report

Stop if:

- effective batch has no PFV-active support
- no independent event support
- no legal fallback after candidate action

### Stage 3: Same-State Data Generation

Inputs:

- approved candidate manifest
- retrofit INP
- forecast contract

Outputs:

- same-state branch outputs
- actual-executed action sequences
- KPI labels from shared contract
- SWMM numerical quality report

Stop if:

- branch state is not equivalent
- actual action differs from logged action
- continuation policy missing
- numerical quality exceeds tolerance

### Stage 4: Model Fit And Gate

Inputs:

- same-state dataset
- GAT state features or perfect-state diagnostic features
- coverage audit

Outputs:

- effect model
- uncertainty model
- gate report
- OOD support report

Stop if:

- support targets missing
- unsafe recall or false-safe thresholds fail
- interval coverage out of bounds
- top-k true shadow validation fails

### Stage 5: Shadow And Smoke

Inputs:

- gated model
- frozen fallback thresholds
- smoke event set

Outputs:

- top-k shadow validation
- smoke closed-loop logs
- action legality report
- fallback reachability report

Stop if:

- backup unreachable after action
- PFV improvement over Internal absent in PFV-active events
- TFV/peak worse than selected fallback
- OOD rejection is bypassed

### Stage 6: Calibration-A

Use Calibration-A only to set interval, non-inferiority, OOD, and fallback
thresholds. Do not search many designs on this set.

### Stage 7: Locked Validation-B

Evaluate at most three frozen configurations and choose K in `{4,8,12}`. K=36
is pressure testing only.

### Stage 8: FormalBlind

FormalBlind is run only after all design, calibration, and validation decisions
are frozen.

## Required Ablations

All use the same retrofit INP:

- No-control diagnostic
- Internal
- Executable Passive
- Dual-fallback-only
- Internal + optimized passive/new facilities
- Full36 without learned effect model
- Full36 learned MPC
- perfect-state vs GAT-state
- K=4/8/12

## Known Risks

- Restored Project4 GAT checkpoints may be incompatible with Project6 node order.
- Passive fallback may be too weak unless facility semantics are correctly
  recovered.
- Learned candidates may exploit unsupported action combinations.
- Internal PFV-active event detection must be based on online predicted Internal
  PFV, not Formal realized Internal PFV.

## Prompt 2 Close-Out Plan

The user has selected `sr0p15` as the primary GAT. The close-out sequence is:

1. Refresh low-cost GAT registry/recovery/inspection/audit markers after config
   changes.
2. Run `SelectPrimaryGAT -GATRegistryName "sr0p15" -AcknowledgeSelection` to
   write the primary GAT lock.
3. Run sr0p15-only robustness diagnostics. Five-candidate comparison remains
   historical audit evidence, but expensive robustness work is focused on the
   selected model.
4. Keep robustness status as `pending` or `incomplete` unless validation
   provenance, unobserved-node, priority leave-out, sentinel leave-out,
   high-water, phase, sensor-failure, repeatability and latency diagnostics all
   pass.
5. Build runtime state features only from an explicit state input manifest. Do
   not default-scan old Project outputs.
6. Evaluate `project6_prompt2_completion_gate.json`. Prompt 2 passes only after
   the sr0p15 lock, robustness gate and real seven-frame state build all pass.

Round 0, effect-model training, MPC, Smoke, Calibration and Formal remain
blocked throughout Prompt 2 close-out.
## V3 Stage And Contract Addendum

### Stage -1: Fatal Audits

Confirm the single retrofit INP, 36 managed IDs, priority/sentinel/storage nodes, forecast contract, KPI contract, time synchronization fields, and SWMM numerical-quality requirements. This stage must fail if any required contract is missing.

### Static Audit Limits

File-presence audit is not compatibility audit. GAT assets are only `present_unverified` until metadata checks pass.

### Disabled Until Implemented

The following stages are intentionally fail-fast in the current runner: FatalAudit, AuditNativeRules, AuditFallbacks, RegisterGAT, AuditGAT, BuildStateFeatures, StateCloneTest, DryRunRound0, GenerateRound0, BuildDataset, TrainPilot, RunPolicyShiftAudit, PlanRound1, GenerateRound1, Round2, TrainFinal, MinimalGate, OptimizerExploitationAudit, DecisionShadowGate, BuildMPC, RunMPCDryRun, RunSmoke, Calibration-A, Locked Validation-B, Policy Lock and FormalBlind.

The only scaffold stages allowed to return success are Status, Audit and InitCoverageSchema. InitCoverageSchema returns `status=scaffold_only`, does not create `_COMPLETED.json`, and unlocks no downstream stage.

Before real Round 0 planning can be enabled, the implementation must add and complete: BuildEventCatalog, BuildCheckpointCatalog, StateCloneTest, and RunInternalPFVOpportunityScan. These are currently disabled and must return exit code 2 if called.

Status and Audit are read-only implemented stages. InitCoverageSchema is a non-destructive scaffold stage. All other stages are blocked or disabled until their contracts, upstream markers, hash checks and outputs are implemented.

They must not silently skip. Each needs a concrete script, marker, hash check and contract test before enabling.

### Minimum Implementation Notes

- KPI: use timestamp deltas from SWMM output; the 10 min control interval is not a hydraulic integration step.
- Coverage: report sample, checkpoint, event and storm-family support for each cell.
- Forecast: main Formal uses operational forecast only. Perfect forecast is an upper-bound diagnostic.
- Fallback: selected fallback must be chosen before any learned candidate is evaluated.
- Passive: minimum necessary intervention, not all-open, all-closed or full reset.
- Backup reachability: check at least one feasible recovery branch after the first executed action.
## Step 1 Implementation: GAT Connection And Augmented State

This step implements only the first paper component: sparse-sensor GAT asset
registration, compatibility auditing, causal 60 min state-history contracts,
facility-local hydraulic features, and state-clone preparation. It does not run
GAT inference, SWMM, PySWMM, action data generation, model training, Smoke,
Calibration, or FormalBlind.

Implementation scope:

1. Register the five Project4 sensor-sensitivity GAT checkpoints by path, hash,
   metadata, and declared sensor ratio. Files with the same checkpoint filename
   must remain separate assets because their parent paths and hashes differ.
2. Audit GAT compatibility with the Project6 retrofit network using layered
   states: `compatible_strict`, `compatible_shared_base_graph_only`,
   `metadata_incomplete`, `incompatible`, and `load_failed`.
3. Keep `selected_primary_gat = null` and
   `selection_status = human_selection_required` until a human selects a
   compatible candidate.
4. Build the augmented-state schema for seven causal history frames spanning the
   past 60 min at 10 min spacing.
5. Preserve unresolved sentinel status. Sentinel candidates may enter
   reconstruction diagnostics, but no sentinel safety PASS is allowed.
6. Encode `add350.1` as a variable-speed pump with pending numeric bounds and
   blocked residual candidate generation. Encode `ADD301.2` and `ADD301.3` as
   strict binary pumps.
7. Prepare state-clone schema and comparators only. The actual equivalence test
   remains blocked until real SWMM checkpoint and controller-memory artifacts
   exist.
