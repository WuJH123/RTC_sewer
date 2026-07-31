# V4.2 Existing SWMM Evidence Reuse

This workflow audits the **complete local Project6 physical SWMM evidence pool**.
It is intentionally not limited to Train1200/Train1600 and does not run SWMM.

## Why this exists

Historical Project6 results contain many valid hydraulic trajectories outside the
latest Train1600 manifest: Pilot, extensions, feasibility/oracle searches, Peak
Boundary, Gate5r, PFV-first rounds and consumed closed-loop evidence.  The final
trajectory-first surrogate should reuse every physically compatible piece of
real SWMM evidence without weakening the formal paper contract.

The local scan reported thousands of `completion.json` and `detail.csv` files.
Most historical detail files predate the explicit `outfall_flow:<id>` recorder,
so outfall supervision must be treated separately from the core hydraulic
trajectory targets.

## Reuse classes

The pool audit uses six classes:

1. `FULL_REUSE` — four references and all formal targets, including explicit
   outfall flow, are present.
2. `REUSE_AFTER_EXTRACTION` — authoritative stored evidence can recover a
   missing target without rerunning SWMM.  This class is not awarded merely
   because a neighbouring/incoming link exists; the recovery rule must be
   independently validated.
3. `PARTIAL_AUX_REUSE` — real partial hydraulic labels are available and may be
   used with explicit availability masks. Missing targets stay missing.
4. `SOURCE_DOMAIN_REUSE` — physically useful source-domain data such as DWF
   histories. These may support dynamics/action-effect pretraining but do not
   become fresh no-DWF validation evidence.
5. `RERUN_REQUIRED` — the physical scenario is potentially compatible but the
   required target cannot be recovered from stored evidence.
6. `INVALID_OR_INCOMPATIBLE` — failed/corrupt evidence, ambiguous physical
   semantics, or another hard contract failure.

Old calibration/locked/formal evidence that has already been revealed is
`consumed_development`, not new Formal Blind evidence.

## Discovery adapters

`sewerrtc/v4/v42_existing_pool_audit.py` discovers both:

- `completion.json`-managed runs; and
- orphan/legacy `detail.csv` or `*_detail.csv` files not represented by a
  completion record (including historical PFV-first and closed-loop layouts).

A manifest row is never assumed to equal one physical SWMM run.  Exact duplicate
physical evidence is collapsed in the canonical inventory while provenance is
retained in `duplicate_lineage_audit.csv`.

## Target rules

Core hydraulic reuse targets are:

- `h:<node>`: node depth, only after `h/head` semantics are checked against INP
  invert elevation;
- `flood:<node>`: node flooding rate;
- `storage_volume:<storage>`;
- `flow:<Engineering36 facility>`;
- `setting:<Engineering36 facility>`: actual/readback action authority;
- `rainfall_mm_h`;
- exact 13 x 5-min history and 12 x 10-min future time coverage.

Extended supervision adds:

- `outfall_flow:<outfall>`.

Missing labels are never filled with zero, copied from another variable, or
silently inferred.

## Outfall flow

`sewerrtc/v4/v42_outfall_recovery.py` separates two questions:

1. Do all physical incoming-link flow columns exist for an outfall?
2. Does their signed sum match a newly recorded explicit
   `outfall_flow:<outfall>` trajectory within frozen tolerance?

Only a new representative run containing **both** explicit outfall flow and all
incoming-link flows can validate the reconstruction rule. Historical CSVs cannot
validate themselves.  Until such a validation passes, historical rows lacking
explicit outfall flow remain `PARTIAL_AUX_REUSE` even when they are structural
reconstruction candidates.

## Commands

Use the local Project6 virtual environment:

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
$Py = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe'
```

Fast metadata/coverage pass:

```powershell
& $Py .\scripts\audit_v42_existing_swmm_pool.py
```

Full finite-value pass for the discovered physical evidence:

```powershell
& $Py .\scripts\audit_v42_existing_swmm_pool.py --full-finite-check
```

Build task-masked reusable manifests after the audit:

```powershell
& $Py .\scripts\build_v42_reusable_pool.py
```

Expected output root:

```text
outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse/
```

Key files:

```text
physical_run_inventory.csv
physical_run_inventory.parquet
target_coverage_by_branch.csv
target_coverage_by_case.csv
duplicate_lineage_audit.csv
source_reuse_summary.csv
partial_aux_reuse.csv
source_domain_reuse.csv
rerun_required.csv
invalid_or_incompatible.csv
data_reuse_audit.json
reusable_pool_manifest.parquet
reusable_case_manifest.parquet
reusable_pool_summary.json
```

## Masked reuse

`sewerrtc/v4/v42_reusable_pool.py` preserves the strict formal builder and adds a
separate reuse view.  A branch can contribute only to tasks for which it has
real evidence:

```text
L = m_depth    * L_depth
  + m_flood    * L_flood
  + m_storage  * L_storage
  + m_facility * L_facility
  + m_outfall  * L_outfall
```

The `m_*` masks come only from the evidence audit.  They are not model choices.

## Stop conditions

This phase does **not**:

- generate another fixed 1600 rows;
- rerun SWMM in bulk;
- delete historical cases;
- train the formal surrogate;
- enter MPC/closed loop/Policy Lock/Challenge/Formal Blind.

After the audit, report physical-run, case/state, event, source-domain and target
coverage counts.  Only the minimum irrecoverable subset should enter a future
rerun plan.
