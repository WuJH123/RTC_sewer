# Project6 V3 Recovery Execution Plan

This plan restores truthful stage accounting before any additional model or
control work. It separates structural engineering readiness from runtime and
scientific evidence.

## Fixed Facts

- Active project: `E:\RTC_sewer\Project6`
- Active network: `E:\RTC_sewer\Project6\data\wuhan_v8_storage_retrofit.inp`
- Primary GAT decision: `sr0p15`
- Managed facilities: 36
- State sampling: 5 min
- Control interval: 10 min
- Prediction horizon: 120 min
- No-control role: diagnostic only
- Internal rules role: PFV effectiveness benchmark
- Dual fallback role: online safety and action-necessity benchmark
- `ADD301.2` and `ADD301.3`: binary pumps
- `add350.1`: variable-speed pump, not binary-toggle logic

## Recovery Stages

1. `AuditCurrentTruth`
   - Reads the existing locks, gates, baseline manifests, state manifests,
     checkpoint catalog, Round0 preview, and dry-run reports.
   - Writes `project6_current_truth_matrix.csv`,
     `project6_current_truth_report.json`, and `project6_recovery_gate.json`.
   - Does not run SWMM, GAT inference, training, data generation, MPC, or
     validation.

2. `EvaluatePrompt3AEngineeringGate`
   - May pass when static contracts, Prompt2 import, small baseline generation,
     state schema, coverage schema, and candidate preview evidence exist.
   - Does not mean runtime safety, scientific pass, or Round0 unlock.

3. `EvaluatePrompt3ARuntimeGate`
   - Must remain blocked until every baseline trajectory enters the state
     pipeline, actual features match frozen schemas, hot-start and controller
     memory are real, State Clone passes, hydraulic candidate dry-run is real,
     truth leakage is zero, engineering violations are zero, and the formal
     Round0 candidate target is met.

4. `EvaluatePrompt3ACompletion`
   - Aggregates engineering and runtime gates.
   - Passes only if both gates pass.

## Expected Current Result

The expected truthful result after the user runs the recovery audit is:

- Prompt2: `pass`
- Prompt3A engineering gate: `pass`
- Prompt3A runtime gate: `blocked`
- Baseline trajectory count: `6`
- Real state-processed trajectory count: reported from current files
- Hot-start equivalence: `not_run` or `missing`
- Hydraulic dry-run: `not_run`
- Effective Round0 candidates: current report value, expected around `9`

Do not modify thresholds or evidence to improve this result. The recovery stage
is a truth audit, not an optimization pass.

