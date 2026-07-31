# Project6 V4.2 FastTrack PoC

This is a **development-only** route for answering one question before spending days on exhaustive Phase R0:

> Can the current V4.2 architecture learn useful hydraulic/control structure and complete an end-to-end micro closed loop?

It does **not** replace the formal paper workflow and can never authorize Policy Lock, Challenge or Formal Blind.

## Why

The project-wide R0 pool contains thousands of historical `detail.csv` files. Full finite-value audit of all files is appropriate before final formal training, but it is unnecessarily expensive as the first test of scientific feasibility.

FastTrack therefore uses a two-speed strategy:

1. use the already available R0.1 metadata inventory to select a small representative core;
2. run expensive finite-value and same-state audits only on that core;
3. train/evaluate the **same formal Step-1 and Step-2 architectures** with short learning curves;
4. evaluate PFV-first MPC on held-out states;
5. run only a 3-event development micro closed loop;
6. expand evidence only if the learning curve/control diagnosis says data are the bottleneck.

## Default scale

- 16 independent rainfall/event groups;
- 3 four-reference cases per group;
- about 48 cases / about 192 branch trajectories before de-duplication;
- rainfall-group isolated train/validation/holdout split.

This is intentionally a feasibility core, not a final training sample-size claim.

## Step A — prepare the core

R0.1 metadata outputs must already exist under the normal data-reuse directory.

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
$Py = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe'

& $Py .\scripts\project6_v42_fasttrack.py prepare-core
```

The command:

- selects only target no-DWF development evidence;
- excludes reserved Challenge/Formal Blind evidence;
- requires metadata four-reference completeness and core hydraulic targets;
- performs full finite-value audit only for selected physical files;
- performs numeric Candidate/NC/DI/Hold same-state + same-H120-rainfall audit;
- builds normal V4.2 reusable manifests for this subset.

Outputs:

```text
outputs/project6_dual_reference_v4/final_v4/v42_fasttrack/core_pool/
  physical_run_inventory.parquet
  target_coverage_by_case.csv
  case_alignment_audit.csv
  reusable_pool_manifest.parquet
  reusable_case_manifest.parquet
  reusable_pool_summary.json
  evidence.json
```

## Step B — Step 1 learnability pilot

Use `TemporalSparseGATReconstructorV42` unchanged. Do not shorten the 13x5-min history or alter topology/action inputs.

Recommended development budget:

- train on rainfall-group-isolated subsets of 4, 8 and 12 groups;
- maximum 20 epochs for each learning-curve point;
- early stopping patience 5;
- keep a fixed holdout rainfall group set.

Write:

```text
v42_fasttrack/step1_gat/evidence.json
```

with at least:

```json
{
  "contract_id": "PROJECT6_V42_FASTTRACK_POC_V1",
  "stage": "step1_gat",
  "status": "pass",
  "development_only": true,
  "formal_authorization": false,
  "metrics": {
    "val_nse_median": 0.70,
    "priority_nse_median": 0.65
  },
  "learning_curve": [
    {"train_groups": 4, "train_score": 0.82, "val_score": 0.60},
    {"train_groups": 8, "train_score": 0.84, "val_score": 0.67},
    {"train_groups": 12, "train_score": 0.85, "val_score": 0.71}
  ]
}
```

If training and validation both plateau low, fix architecture/targets before adding thousands of files. If training remains high while validation improves with more groups, expand evidence selectively.

## Step C — Step 2 learnability pilot

Use `MultiReferenceHydraulicSurrogate` and `HydraulicTrajectoryLoss` unchanged. Preserve four branches, H120 and trajectory-derived PFV/TFV/Peak. Partial target masks are allowed only for this development pilot; they do not authorize formal training.

Required diagnostic metrics:

- PFV direction accuracy vs No-control;
- TFV direction accuracy vs Dynamic Internal;
- Peak direction accuracy vs Dynamic Internal;
- safe-candidate recall;
- false-safe rate;
- learning curve over 4/8/12 rainfall groups.

Write `v42_fasttrack/step2_surrogate/evidence.json`.

## Step D — Step 3 MPC pilot

Use the existing `pfvfirst_mpc_v42` selector unchanged.

Evaluate held-out same-state candidate sets and report:

- number of states evaluated;
- whether a SWMM-truth safe+useful candidate exists;
- safe-selection precision;
- good-candidate recall;
- candidate rejection reasons.

This separates three failure modes:

1. no useful physical candidate exists;
2. candidate exists but surrogate cannot rank it;
3. surrogate ranks it but MPC gating rejects/selects incorrectly.

Write `v42_fasttrack/step3_mpc/evidence.json`.

## Step E — Step 4 micro closed loop

Run only 3 representative development events, in this order:

1. Exact SWMM micro closed loop;
2. surrogate micro closed loop;
3. GAT-integrated micro closed loop.

Do not use Challenge/Formal Blind events. Do not claim statistical evidence.

Report PFV non-inferiority vs No-control, Peak non-inferiority vs Dynamic Internal, TFV improvement rate and fallback use. Write `v42_fasttrack/step4_micro_closed_loop/evidence.json`.

## Gate

```powershell
& $Py .\scripts\project6_v42_fasttrack.py gate
```

Interpretation:

- **complete / expand_evidence_for_formal_workflow**: the architecture is technically viable; now expand R0 only where formal coverage is missing.
- **data_limited_expand_targeted_evidence**: add only rainfall/state/action regions indicated by the learning curve/error analysis.
- **model_or_target_limited_do_not_bulk_expand**: stop bulk R0 work; repair targets/model semantics first.
- **targeted_repair_before_more_compute**: fix candidate generation/MPC/closed-loop issue before more data work.

## Formal boundary

FastTrack never changes or bypasses `PROJECT6_V42_PAPER_WORKFLOW_V1`. Before paper claims, return to the formal workflow and satisfy complete R0, raw Independent Oracle, full target coverage, true-state offline validation, the three formal closed loops, Policy Lock, Challenge and >=24-event Formal Blind.
