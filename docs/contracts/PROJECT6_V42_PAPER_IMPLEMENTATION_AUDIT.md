# Project6 V4.2 paper-workflow implementation audit

Audit date: 2026-07-31  
Authoritative workflow contract: `PROJECT6_V42_PAPER_WORKFLOW_V1`

This audit compares the repository implementation with the final four-step
paper method.  It intentionally distinguishes **code contract implemented** from
**scientific evidence already executed**.  A code path is not marked complete
merely because historical V3/V4.1 output exists.

## Executive status

| Paper step | Repository status after this PR | Formal scientific status |
|---|---|---|
| Step 1 sparse state reconstruction | Formal 13x5-min temporal/action/link-static GAT architecture and fail-closed state adapter implemented; legacy 7-frame truth-as-reconstruction path repaired | **Pending new formal GAT training, uncertainty calibration, OOD calibration and compatibility audit** |
| Step 2 four-reference hydraulic surrogate | Shared trajectory-first four-branch model, hydraulic trajectory losses, explicit raw target recorder and target-coverage gate implemented | **Blocked until raw Independent Oracle and complete hydraulic target coverage pass; legacy detail lacks explicit outfall-flow target** |
| Step 3 PFV-first rolling MPC | Canonical hard-safety selector implemented; TFV evaluated only inside safe set; frozen fallback on empty set/selection failure | **Needs closed-loop execution evidence with real candidate generator/readback/engineering guards** |
| Step 4 closed-loop + blind sequence | Independent V4.2 stage gate implemented in required order; V4.1/old Locked/development evidence cannot authorize Formal | **Exact/Surrogate/GAT-integrated closed-loop executions, Policy Lock, Challenge and Formal Blind still must be run** |

## Step 1 audit

### Errors found in merged `main`

1. `state_contract.py` declared seven 10-minute frames although the final
   trajectory contract uses thirteen 5-minute frames.
2. `state_input_manifest.py` copied authoritative SWMM truth depth into
   `reconstructed_depth`, effectively relabelling truth as a GAT estimate.
3. The same path used depth as `hydraulic_head` instead of invert elevation +
   depth.
4. Storage state was materialised with a zero-length storage axis.
5. GAT uncertainty and OOD were not present, yet Project6 baseline states could
   be labelled `full_project6_augmented_state_eligible=true`.
6. Historical `SparseGATReconstructor` is a current-snapshot depth model.  It
   does not consume 13-frame history, historical actions or link attributes and
   therefore cannot alone support the final Step-1 paper claim.

### Repairs

- Formal temporal offsets are now `t-60,t-55,...,t` (13 frames).
- SWMM truth is explicitly labelled true state and cannot populate
  `reconstructed_depth`.
- Physical head/filling/headroom are derived from INP metadata.
- Real storage nodes are materialised; unavailable storage volume/flow remains
  missing rather than zero-filled.
- Added `TemporalSparseGATReconstructorV42`, which consumes sparse depth/mask,
  rainfall, historical actions, topology, node static and link static inputs and
  returns depth mean + uncertainty.
- Added an adapter that can still reproduce legacy sr0p15 validation but marks
  it `legacy_single_snapshot` and never `formal_online_state_eligible`.

### Remaining evidence requirement

The new temporal GAT architecture has no inherited scientific score.  It must be
trained/validated on independent sparse-sensor data, uncertainty-calibrated and
locked before the GAT-integrated closed loop.  Historical sr0p15 NSE is useful
background evidence only unless the final paper method is changed back to its
actual single-snapshot input contract.

## Step 2 audit

### Errors found in merged `main`

1. The four branches existed at the action/depth level, but the formal trainer
   still predicted node depth only.
2. PFV/TFV/Peak were produced by independent KPI regression heads.  They were
   not deterministic derivatives of a predicted flooding-rate trajectory.
3. Storage volume, managed-facility flow, outfall flow and system
   flooding-rate sequence were not formal model outputs.
4. Historical actions were stored by the trajectory builder but not part of the
   old formal state encoder contract.
5. Legacy generic SWMM detail rows contain depth, flooding, storage volume and
   managed-facility flow, but do not contain an explicit
   `outfall_flow:<outfall_id>` target.

### Repairs

- Added a shared `MultiReferenceHydraulicSurrogate` for Candidate, No-control,
  Dynamic Internal and Hold Previous.
- Added outputs for node depth, node flooding rate, storage volume,
  managed-facility flow, outfall flow and system flooding-rate sequence.
- PFV and TFV deltas are integrated from **counterfactual rate differences** to
  avoid floating-point cancellation; Peak remains exactly
  `max(Candidate)-max(DynamicInternal)`.
- Added hydraulic trajectory losses.  KPI consistency is secondary and is only
  computed on already-derived KPIs.
- Added a formal SWMM recorder contract.  Outfall flow is recorded from the
  authoritative SWMM/PySWMM outfall node total inflow, not inferred from node
  flooding or a neighbouring link.
- Added raw-detail target coverage audit and a formal training-admission gate.
  Stored-oracle evidence or partial hydraulic target coverage cannot authorize
  formal training.

### Remaining evidence/data requirement

Existing local Train1600 detail files must be audited.  If they do not contain
explicit outfall-flow columns, the affected trajectories must be regenerated or
an authoritative SWMM binary-output extraction must be performed.  Do not
silently impute outfall flow.

## Step 3 audit

### Errors found in merged `main`

Legacy controllers mix older research semantics: some use Passive in the PFV
reference envelope, some treat TFV as a hard gate, and some retain weighted PFV
or Peak performance terms.  Those modules remain useful historical baselines
but do not implement the final paper controller.

### Repairs

The canonical paper selector is now
`sewerrtc.control.pfvfirst_mpc_v42.decide_pfvfirst_mpc`.

Safety admission occurs before the performance objective and contains:

- PFV safety vs No-control;
- Peak safety vs Dynamic Internal;
- K;
- bounds/rate/ramp/dwell/interlock status;
- uncertainty;
- OOD;
- executability.

Only admitted candidates are scored by

`DeltaTFV_DI + lambda_action J_action + lambda_terminal J_terminal + lambda_uncertainty J_uncertainty`.

TFV gain cannot compensate a safety failure.  Empty safe set or candidate
selection failure executes a frozen, hashed fallback.

### Remaining execution requirement

The selector receives audited engineering-status/readback fields from the
candidate generator.  Formal closed-loop evidence must demonstrate that these
flags are computed from the actual requested/projected/written/readback sequence
and not merely asserted by a caller.

## Step 4 audit

### Errors found in merged `main`

The older closed-loop pipeline is tied to a V4.1 compact-model gate, and its
Exact/Surrogate run handlers include deliberate blocked placeholders.  That
pipeline cannot be presented as completion of the final paper sequence.

### Repairs

Added an independent V4.2 workflow gate with the only admissible order:

1. true-state surrogate offline validation;
2. Exact SWMM closed loop;
3. surrogate closed loop;
4. GAT-integrated closed loop;
5. Policy Lock;
6. Challenge;
7. Formal Blind.

Every evidence file must carry the V4.2 paper contract/model lineage.  Legacy
V4.1, old Calibration/Locked or development evidence is rejected as a
substitute.  Formal Blind requires at least 24 events, new rainfall SHA only,
Policy Lock before reveal, no retraining and no post-reveal exclusions.

## Formal stop conditions

Do **not** claim the paper workflow complete until all of the following are true:

1. formal temporal GAT training/validation + uncertainty/OOD calibration passes;
2. raw Independent Oracle passes the admitted pool;
3. every formal hydraulic target group has audited physical coverage;
4. trajectory-first surrogate true-state offline validation passes;
5. Exact SWMM closed loop passes;
6. surrogate closed loop passes;
7. GAT-integrated closed loop passes;
8. Policy Lock hashes model/controller/fallback before reveal;
9. Challenge is executed without retraining;
10. Formal Blind is executed on unrevealed events and is the only final evidence.

Historical assets are retained for reproducibility and comparison, but they are
not allowed to lower these gates.
