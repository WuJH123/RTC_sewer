# Project6 V4.2 mainline integrity review

Canonical research chain:

`Phase R0 -> Step 1 Temporal Sparse GAT -> Step 2 four-reference hydraulic trajectory surrogate -> Step 3 PFV-first rolling MPC -> Step 4 closed-loop / Policy Lock / Challenge / Formal Blind`

This document distinguishes **implemented code**, **scientific evidence**, and **legacy/development utilities**. Passing unit tests is not equivalent to completing a paper stage.

## Phase R0

Canonical commands:

1. `audit_v42_existing_swmm_pool.py` — combined R0.1+R0.2 strict full-finite scan by default;
2. `audit_v42_case_alignment.py` — numeric Candidate/NC/DI/Hold same-prefix + same-H120-rainfall audit;
3. `build_v42_reusable_pool.py` — strict target-masked views; every formal/counterfactual role must resolve to finite evidence;
4. `build_v42_reuse_split_groups.py` — rainfall-series isolation groups.

Scientific rules:

- actual `setting:<Engineering36>` readback is the action authority;
- missing target is never zero;
- old revealed Calibration/Locked/Formal may be `consumed_development`; new V4.2 Challenge/Formal stays reserved;
- threaded optimization may not change hashes, case identities, classifications or output determinism.

## Step 1 — sparse-state reconstruction

Formal model: `TemporalSparseGATReconstructorV42`.

Required evidence before the mainline can proceed:

- new 13 x 5-min Project6 training, not legacy 7-frame/snapshot validation;
- rainfall-group-isolated validation;
- historical **actual readback** actions;
- uncertainty calibration;
- OOD calibration;
- frozen model SHA;
- no future hydraulic truth.

`state_input_manifest.py` contains historical/diagnostic paths and must not independently authorize the formal Step-1 mainline. Formal evidence must explicitly state `action_authority=actual_readback_setting`.

## Step 2 — four-reference hydraulic surrogate

Formal model: `MultiReferenceHydraulicSurrogate` + `HydraulicTrajectoryLoss`.

Canonical R0 bridge:

- `build_v42_r0_paper_dataset.py` reads the strict R0 manifests rather than rediscovering Train1600;
- history state is depth-only (`gat_compatible_causal_state`) plus historical actual readback actions;
- full-network flooding is a **prediction target**, not an online input feature;
- Candidate/NC/DI/Hold share one model;
- PFV is Candidate vs No-control; TFV/Peak are Candidate vs Dynamic Internal;
- KPI outputs are derived from the predicted flooding trajectory, not independent KPI heads;
- source physical IDs/detail paths are preserved for the raw Independent Oracle;
- `audit_v42_r0_independent_oracle.py` validates the exact R0-derived population at the formal H12 target timestamps.

Formal training admission has no fixed 1200/1600 quota. An explicit expected count is optional only for a deliberately frozen experiment.

## Step 3 — PFV-first rolling MPC

Canonical selector: `decide_pfvfirst_mpc`.

Required semantics:

1. hard PFV safety vs No-control;
2. hard Peak safety vs Dynamic Internal;
3. K <= 8 and bounds/rate/ramp/dwell/interlock/executability;
4. calibrated uncertainty and OOD gates;
5. only inside the safe set minimize TFV vs Dynamic Internal plus action/terminal/uncertainty costs;
6. execute only the first 10-min action and re-plan;
7. empty safe set / error -> frozen fail-closed fallback.

The selector consumes safety fields; formal Step-3 evidence must prove these fields and `changed_facilities` were derived from projected/written/readback execution, not asserted by a caller.

## Step 4 — hierarchical evidence

Required order:

1. true-state offline validation;
2. authoritative Exact-SWMM closed loop;
3. surrogate closed loop;
4. GAT-integrated closed loop;
5. Policy Lock;
6. Challenge;
7. Formal Blind (>=24 new rainfall SHA events).

Challenge/Formal must carry exactly the policy/model/fallback hashes frozen at Policy Lock. Formal must provide an explicit unique rainfall-SHA list and prove zero overlap with revealed development events. Post-reveal exclusion/retraining is forbidden.

## Mainline command

`python scripts/project6_v42_mainline.py`

This command is fail-closed and reports the first incomplete stage. FastTrack evidence remains development-only and cannot satisfy the formal mainline.

## Known execution gap after this review

The repository has the formal Step-1 architecture and the formal Step-2 model, but the **local data-dependent Step-1 training/calibration runner and authoritative Step-3 candidate/execution adapter must still be validated/finished locally**. The mainline gate intentionally blocks rather than allowing placeholder evidence.
