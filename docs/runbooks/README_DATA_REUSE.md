# V4.2 Existing SWMM Evidence Reuse

This workflow audits the **complete local Project6 physical SWMM evidence pool**.
It is intentionally not limited to Train1200/Train1600 and does not run SWMM.

## Why this exists

Historical Project6 results contain many valid hydraulic trajectories outside the
latest Train1600 manifest: Pilot, extensions, feasibility/oracle searches, Peak
Boundary, Gate5r, PFV-first rounds and consumed closed-loop evidence. The final
trajectory-first surrogate should reuse every physically compatible piece of
real SWMM evidence without weakening the formal paper contract.

The local scan reported thousands of `completion.json` and `detail.csv` files.
Most historical detail files predate the explicit `outfall_flow:<id>` recorder,
so outfall supervision is treated separately from core hydraulic trajectory
supervision rather than causing the whole historical pool to be discarded.

## Reuse classes

The pool audit uses six classes:

1. `FULL_REUSE` — four references and all formal targets, including explicit
   outfall flow, are present.
2. `REUSE_AFTER_EXTRACTION` — authoritative stored evidence can recover a
   missing target without rerunning SWMM. This class is not awarded merely
   because an incoming link exists; the recovery rule must be independently
   validated first.
3. `PARTIAL_AUX_REUSE` — real partial hydraulic labels are available and may be
   used with explicit availability masks. Missing targets stay missing.
4. `SOURCE_DOMAIN_REUSE` — physically useful source-domain data such as DWF
   histories. These may support dynamics/action-effect pretraining but do not
   become fresh no-DWF validation evidence.
5. `RERUN_REQUIRED` — the physical scenario is potentially compatible but a
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

A manifest row is never assumed to equal one physical SWMM run. Exact duplicate
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
- a real 13 x 5-min history plus 12 x 10-min H120 future window.

Extended supervision adds:

- `outfall_flow:<outfall>`.

Missing labels are never filled with zero, copied from another variable, or
silently inferred. A metadata-only scan never authorizes training: the reusable
training view requires the full finite-value audit.

## Same-state counterfactual rule

Hashes are lineage evidence, not enough to authorize Candidate-vs-reference
learning. `v42_case_alignment_audit.py` reopens Candidate/NC/DI/Hold details and
compares the real 13-frame prefix numerically (node depth, Storage volume,
Engineering36 flow and readback setting) and verifies that H120 rainfall forcing
is identical. Counterfactual case supervision remains false until this audit
passes.

Legacy whole-event files with no formal checkpoint can still contribute to
single-branch dynamics/action-effect pretraining when their timestamps contain
at least one real 13x5-min + 12x10-min window; they do not automatically become
four-reference counterfactual cases.

## Outfall flow

`sewerrtc/v4/v42_outfall_recovery.py` separates two questions:

1. Do all physical incoming-link flow columns exist for an outfall?
2. Does their signed sum match a newly recorded explicit
   `outfall_flow:<outfall>` trajectory within frozen tolerance?

Only a new representative run containing **both** explicit outfall flow and all
incoming-link flows can validate the reconstruction rule. Historical CSVs cannot
validate themselves. Until such a validation passes, historical rows lacking
explicit outfall flow remain `PARTIAL_AUX_REUSE` even when they are structural
reconstruction candidates.

After validation passes, `v42_outfall_bulk_recovery.py` writes separate sidecar
Parquet files. It never edits historical `detail.csv` in place.

## Event/rainfall isolation

`v42_reuse_split.py` derives a `base_rainfall_fingerprint` from the recorded
elapsed-time/rainfall series. The same rainfall appearing in different historical
versions or DWF/no-DWF domains must stay in one future split group. Row-random
splitting is prohibited.

## Commands

Use the local Project6 virtual environment:

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
$Py = 'E:\RTC_sewer\Project6\.venv\Scripts\python.exe'
```

### 1. Fast project-wide discovery pass

```powershell
& $Py .\scripts\audit_v42_existing_swmm_pool.py
```

This is for inventory/reporting only. It must **not** authorize a training pool.

### 2. Full finite-value audit

```powershell
& $Py .\scripts\audit_v42_existing_swmm_pool.py --full-finite-check
```

This rewrites the audit inventory with finite-value evidence.

### 3. Numeric four-reference alignment audit

```powershell
& $Py .\scripts\audit_v42_case_alignment.py
```

Expected output: `case_alignment_audit.csv`.

### 4. Build target-masked reusable manifests

```powershell
& $Py .\scripts\build_v42_reusable_pool.py
```

By default this requires both the full finite audit and the case-alignment audit.
Use `--allow-missing-alignment-audit` only for diagnostic/generic-pretraining
views; in that mode counterfactual case eligibility stays false.

### 5. Build cross-version rainfall split groups

```powershell
& $Py .\scripts\build_v42_reuse_split_groups.py
```

Expected output: `split_group_manifest.parquet`.

### 6. Optional Outfall validation (requires explicit recorder evidence)

Do **not** run this against an old historical file lacking explicit outfall
columns. After a separately authorized representative run using the new recorder:

```powershell
& $Py .\scripts\validate_v42_outfall_reconstruction.py `
  --detail <NEW_DETAIL_WITH_EXPLICIT_OUTFALL.csv> `
  --output .\outputs\project6_dual_reference_v4\final_v4\v42_paper\data_reuse\outfall_reconstruction_validation.json
```

If and only if that report is `status=pass`, historical structural candidates
can be recovered into sidecars:

```powershell
& $Py .\scripts\recover_v42_outfall_sidecars.py `
  --validation-json .\outputs\project6_dual_reference_v4\final_v4\v42_paper\data_reuse\outfall_reconstruction_validation.json
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
case_alignment_audit.csv
reusable_pool_manifest.parquet
reusable_case_manifest.parquet
reusable_pool_summary.json
split_group_manifest.parquet
outfall_reconstruction_validation.json                 # only after validation run
recoverable_from_validated_links.csv                   # only after validation pass
outfall_sidecars/                                      # only after validation pass
```

## Masked reuse

`sewerrtc/v4/v42_reusable_pool.py` preserves the strict formal builder and adds a
separate reuse view. A branch can contribute only to tasks for which it has real
evidence:

```text
L = m_depth    * L_depth
  + m_flood    * L_flood
  + m_storage  * L_storage
  + m_facility * L_facility
  + m_outfall  * L_outfall
```

The `m_*` masks come only from the evidence audit. They are not model choices.
PFV/TFV/Peak counterfactual supervision still requires a valid same-state four-
reference case and real flooding-rate trajectories.

## Stop conditions

This phase does **not**:

- generate another fixed 1600 rows;
- rerun SWMM in bulk;
- delete historical cases;
- train the formal surrogate;
- enter MPC/closed loop/Policy Lock/Challenge/Formal Blind.

After the audit, report physical-run, case/state, event, source-domain and target
coverage counts. Only the minimum irrecoverable subset should enter a future
rerun plan.
