# Project6 Agent Contract

This repository is in a recovery and freezing phase. Agents must read current
local files before deciding what is complete. Do not rely on old conversation
claims when local evidence disagrees.

## Fixed Project Scope

- Active project: `E:\RTC_sewer\Project6`
- Only physical network for active V4 evidence:
  `E:\RTC_sewer\Project6\data\wuhan_v8_storage_retrofit.inp`
- The active network is rainfall-only forced and contains no sanitary base
  inflow section or associated patterns. The preserved original is
  `data/wuhan_v8_storage_retrofit_original_with_base_inflow_7ea1e133.inp`;
  it is provenance-only and must never be selected by active configs.
- Primary GAT: user-selected `sr0p15`
- Managed facilities: all 36 facilities in
  `data/project6_v8_storage_retrofit_control_enabled_ids.txt`
- State sampling: 5 min
- Control interval: 10 min
- Prediction horizon: 120 min

## V4 Reference Roles

- No-control is the PFV safety reference.  It means all Engineering36 managed
  facilities are held at 1.0 after the checkpoint by time-gated priority-100
  SWMM rules. All branches retain the same native controls before checkpoint.
- Dynamic Internal SWMM rules are the TFV and peak performance reference.
- Candidate, No-control, Hold-previous and Dynamic Internal must have identical
  full 60-minute hydraulic prefix and checkpoint pre-action hashes.
- Hold-previous is an action-necessity diagnostic, not an independent PFV
  reference.
- Candidate-minus-reference is the only delta convention.
- Do not inherit V1-V3 controllers, thresholds, event splits, or paired
  datasets unless they pass the current V4 hash and same-state contracts.

## Facility Semantics

- `ADD301.2` and `ADD301.3` are strict binary pumps with action set `{0, 1}`.
- `add350.1` is a variable-speed pump. Do not route it through binary toggle
  logic and do not assume its numeric bounds until INP/rule evidence confirms
  them.
- Storage inlet/outlet interlocks, pump dwell, actual executed actions, override
  TTL, release, and fallback mode must be logged before learned control can be
  treated as runtime evidence.

## Truth and Completion Rules

- Do not claim a stage passed because a file or marker exists. Check the current
  evidence files and hashes.
- Schema-only, structural-only, `not_run`, blocked, failed, or stale stages must
  not be reported as runtime or scientific pass.
- Completion markers are invalid when outputs are empty, referenced files are
  missing, hot-start equivalence is `not_run`, hydraulic dry-run is `not_run`, or
  config/input/script hashes are stale.
- Prompt3A has separate gates:
  - Engineering gate: contracts, schemas, small baseline, and candidate preview.
  - Runtime gate: real state pipeline, hot-start, controller memory, State Clone,
    real hydraulic dry-run, leakage, engineering violations, and full Round0
    target support.

## User-Executed Commands

Unless the user explicitly permits execution in the current turn, agents must
write commands into the current V4 runbook or handover document.

The user runs commands manually and provides logs. Do not claim tests,
simulation, GAT inference, training, data generation, MPC, calibration, or
FormalBlind passed unless the user-provided evidence proves it.

## Required Evidence

Every code-change response must report:

- Modified files.
- Key design decisions.
- Commands for the user to run.
- Expected outputs and exit codes.
- Acceptance conditions.
- Logs needed if a step fails.
