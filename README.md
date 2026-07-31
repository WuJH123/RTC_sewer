# RTC_sewer

Real-time control (RTC) research codebase for the Wuhan large urban drainage system.

The current authoritative research line is **Project6 V4.2**. Its scientific objective is to use sparse sensing, a shared multi-reference hydraulic surrogate, and PFV-first rolling MPC to reduce total flooding and peak risk **without degrading priority flooding relative to the no-control reference**.

> **Important:** historical V3/V4/V4.1 assets are retained for reproducibility and comparison, but they do not authorize the final paper workflow. The final scientific contract is `PROJECT6_V42_PAPER_WORKFLOW_V1`.

## 1. Final paper workflow

### Step 1 — Sparse state reconstruction

Inputs:

- sparse sensor observations;
- sensor mask;
- rainfall history;
- network topology;
- historical executed actions;
- static node and link attributes.

Formal temporal semantics:

- state record interval: **5 min**;
- history window: **60 min**;
- history frames: **13**, i.e. `t-60, t-55, ..., t`;
- control interval: **10 min**.

Outputs include reconstructed full-network node depth, hydraulic head, filling degree, key Storage state, relevant flow/state summaries, GAT uncertainty and OOD information.

The state reconstructor is a **state estimator only**. It does not choose actions.

Formal implementation:

- `sewerrtc/models/temporal_sparse_gat_v42.py`
- `sewerrtc/state/v42_sparse_state.py`

Legacy `sr0p15` remains a historical single-snapshot depth reconstructor and cannot by itself authorize the final GAT-integrated closed loop.

### Step 2 — Shared multi-reference hydraulic surrogate

The formal surrogate implements

```text
Y_b = F_theta(X_t-60:t, R_t:t+120, U_b, G)
```

with **shared model parameters** and four hydraulic rollouts:

- Candidate;
- No-control (NC);
- Dynamic Internal rules (DI);
- Hold Previous.

The model should predict future hydraulic trajectories first:

- node depth;
- node flooding rate;
- Storage volume/state;
- managed-facility flow;
- Outfall flow;
- system flooding-rate sequence.

PFV, TFV and Peak are then derived from the predicted flooding-rate trajectories, not from authoritative free-standing KPI heads:

```text
Delta PFV_NC  = integral[(Candidate - NoControl) priority-node flooding rate]
Delta TFV_DI  = integral[(Candidate - DynamicInternal) system flooding rate]
Delta Peak_DI = max(Candidate system flooding rate) - max(DynamicInternal system flooding rate)
```

Formal implementation:

- `sewerrtc/v4/models_v42/hydraulic_multi_reference.py`
- `sewerrtc/v4/models_v42/hydraulic_trajectory_losses.py`
- `sewerrtc/v4/v42_paper_dataset.py`
- `sewerrtc/v4/v42_hydraulic_target_audit.py`
- `sewerrtc/simulation/v42_hydraulic_recorder.py`

Action authority for formal learning is the **actual/readback action** (`setting:<facility>`), not merely the requested/target action (`a:<facility>`).

### Step 3 — PFV-first rolling MPC

Candidate actions are admitted to the safe set only when all required hard constraints pass:

- PFV safety vs No-control;
- Peak safety vs Dynamic Internal;
- `K <= 8` changed facilities;
- bounds, rate, ramp, dwell and interlock;
- uncertainty and OOD gates;
- executability/readback consistency.

Only inside the safe set is performance optimized:

```text
DeltaTFV_DI
+ lambda_action * J_action
+ lambda_terminal * J_terminal
+ lambda_uncertainty * J_uncertainty
```

A TFV benefit is never allowed to compensate a PFV/Peak/engineering safety violation. If the safe set is empty, or candidate selection fails, the controller executes the frozen hashed fallback.

Formal implementation:

- `sewerrtc/control/pfvfirst_mpc_v42.py`

### Step 4 — Closed loop and blind evaluation

The only admissible final sequence is:

```text
True-state surrogate offline validation
    -> Exact SWMM closed loop
    -> Surrogate closed loop
    -> GAT-integrated closed loop
    -> Policy Lock
    -> Challenge
    -> Formal Blind
```

Development results, historical Calibration, old Locked results and V4.1 evidence cannot substitute for the final Formal Blind evaluation.

Formal workflow audit:

```powershell
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
& $Py .\scripts\project6_v42_paper.py
```

## 2. Highest-priority next task: reuse the complete existing SWMM data pool

**Do not start by generating another fixed 1200- or 1600-row dataset.**

The historical names `Train1200` / `Train1600` describe previous experiment plans; they are **not the scientific upper bound of the available physical data**. The next task is to inventory and audit **all existing real SWMM simulations that are physically compatible with the frozen V4.2 contract**, then reuse as much as possible.

The first question to answer is:

> How many existing real SWMM cases can be reused without rerunning SWMM for the new trajectory-first surrogate, and which hydraulic target groups are already available for each case/branch?

The audit must recursively scan the complete local Project6 output pool, including historical candidate/reference runs and later pilot/extension/oracle runs, rather than scanning only one `Train1600` directory or one manifest.

### Required pool-wide audit dimensions

For every discovered physical simulation/case, record at least:

- source root and run directory;
- event ID, checkpoint/state key and case ID;
- branch: Candidate / No-control / Dynamic Internal / Hold Previous;
- network/INP SHA and rainfall/event SHA;
- checkpoint and exact history/future timestamps;
- existence of `completion.json`, `detail.csv`, SWMM `.out`, `.rpt` or other authoritative raw output;
- readback action schedule `setting:<facility>`;
- node depth `h:<node>`;
- node flooding rate `flood:<node>`;
- Storage volume `storage_volume:<storage>`;
- managed-facility flow `flow:<facility>`;
- explicit Outfall flow `outfall_flow:<outfall>`;
- rainfall forcing;
- finite fraction, time coverage, branch alignment and duplicate/lineage hashes.

### Reuse classification

Each discovered case should be classified into one of the following categories instead of being immediately discarded:

1. **FULL_REUSE** — all four branches and all formal hydraulic targets already exist with correct timing/readback; no SWMM rerun is needed.
2. **REUSE_AFTER_EXTRACTION** — the original physical SWMM run is valid and authoritative `.out`/other raw output still exists, so missing targets (especially Outfall flow) can be extracted without rerunning the hydraulic simulation.
3. **PARTIAL_AUX_REUSE** — only part of the formal targets are present. These rows may be reused with explicit target-availability masks for compatible auxiliary/multi-task losses, but missing targets must never be imputed or treated as zero.
4. **RERUN_REQUIRED** — the physical run is compatible, but the missing target cannot be recovered from stored authoritative output, so only the minimum required branch/case should be rerun.
5. **INVALID_OR_INCOMPATIBLE** — wrong network variant, wrong DWF semantics, broken same-state/reference lineage, missing readback authority, corrupted timing, failed run or other contract violation.

The final audit must report **counts by physical case, state, event, branch and target group**, not only row counts.

### Critical reuse principle

The new trajectory-first dataset should be **pool-driven, not quota-driven**:

```text
all compatible physical SWMM evidence
    -> lineage + authenticity deduplication
    -> target-coverage audit
    -> recover missing targets from stored raw outputs where possible
    -> tiered reusable dataset
    -> train/calibration/validation split by independent event
```

There is no requirement to stop at exactly 1600 samples. If 2300, 4000 or more compatible, independent and hydraulically informative cases are available, they should be admitted according to the frozen scientific contract.

Likewise, do not duplicate or resample cases merely to reach a round number.

### Formal vs auxiliary supervision

To maximize reuse without weakening the paper contract:

- **core/formal supervision** must use physically observed targets with valid lineage and no imputation;
- **partial historical rows** may contribute only to target heads/loss terms for which authoritative labels exist;
- every target must have an availability mask and provenance;
- validation/Challenge/Formal Blind must remain event-isolated and cannot leak into training;
- any relaxation from the current all-target-per-row builder must be explicit, tested and documented rather than silently changing the contract.

## 3. Data-admission gates

The current fail-closed training admission combines:

1. raw Independent Oracle evidence;
2. hydraulic-target coverage evidence.

Existing entry point:

```powershell
& $Py .\scripts\audit_v42_paper_training_admission.py `
  --oracle-summary <RAW_ORACLE_SUMMARY.json> `
  --hydraulic-target-audit <HYDRAULIC_TARGET_AUDIT.json> `
  --expected-count <AUDITED_POOL_COUNT> `
  --output <TRAINING_ADMISSION.json>
```

`--expected-count` must reflect the **audited admitted pool**, not a hard-coded scientific target of 1200 or 1600.

Before formal surrogate training, the repository should provide a reproducible pool-wide inventory and reuse manifest, for example:

```text
outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse/
    physical_run_inventory.csv
    target_coverage_by_branch.csv
    target_coverage_by_case.csv
    duplicate_lineage_audit.csv
    recoverable_from_raw_output.csv
    rerun_required.csv
    invalid_or_incompatible.csv
    reusable_pool_manifest.parquet
    reusable_pool_summary.json
```

These filenames describe the intended evidence contract; if implementation names differ, preserve equivalent information and document the mapping.

## 4. Current implementation status

Code architecture and fail-closed contracts for the final paper workflow are implemented, but the scientific workflow is not yet complete.

Current remaining evidence sequence:

1. **pool-wide existing SWMM data reuse audit**;
2. recover authoritative missing hydraulic targets from existing raw outputs where possible;
3. run raw Independent Oracle on the admitted reusable pool;
4. build the complete/tiered trajectory-first dataset using readback actions;
5. train and validate the formal shared multi-reference hydraulic surrogate;
6. train/validate/calibrate the formal temporal sparse GAT;
7. true-state offline validation;
8. Exact SWMM closed loop;
9. surrogate closed loop;
10. GAT-integrated closed loop;
11. Policy Lock;
12. Challenge;
13. Formal Blind on unrevealed independent events.

Detailed scientific/code audit:

- `docs/contracts/PROJECT6_V42_PAPER_IMPLEMENTATION_AUDIT.md`
- `docs/contracts/PROJECT6_V42_PAPER_WORKFLOW_CONTRACT.json`

## 5. Repository and local environment

Repository:

```text
https://github.com/WuJH123/RTC_sewer
```

Typical local root:

```text
E:\RTC_sewer\Project6
```

Recommended Python interpreter:

```text
E:\RTC_sewer\Project6\.venv\Scripts\python.exe
```

Before running scientific stages, synchronize the local repository with `origin/main`, preserve untracked local SWMM outputs, verify the working tree and code SHA, and never delete/recreate historical physical output directories merely to match a new manifest.

## 6. Scientific safety rules

- SWMM is the authoritative physical truth source.
- No future hydraulic truth may enter online state reconstruction or MPC.
- Same-state counterfactual branches must diverge only after the checkpoint.
- Requested/projected/written/target/current/readback actions must remain distinguishable.
- Missing hydraulic labels are **missing**, not zero.
- Old development evidence cannot authorize Formal Blind.
- Challenge/Formal events cannot be used for retraining or post-reveal model selection.
- Reuse existing physical simulations before scheduling expensive SWMM reruns.
