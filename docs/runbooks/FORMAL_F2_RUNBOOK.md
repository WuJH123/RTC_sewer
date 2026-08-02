# Project6 V4.2 Formal F2 runbook

This is the formal (paper-evidence) line. `fast_e2e_64plus` remains development-only and cannot authorize Formal.

## Scientific data roles

- **Formal Step1 physics:** all current-Wuhan, finite, causal, readback-compatible physical trajectories discovered through the F2 source registry; physical detail SHA dedupe; maximum 4 timeline-spread windows per physical run.
- **Formal Step2 core:** Train1600 current 14-gate rows + Pilot V3 training-eligible rows + raw re-admitted Peak Boundary + raw re-admitted revealed V4.1 Calibration/Locked. Candidate/NC/Dynamic-Internal/Hold-Previous are mandatory.
- **Auxiliary only:** V3/V4 Base/Rounds, Aug1, Gate5R/oracle search, Pilot V1/V2 superseded rows, augmented-state/unified/opportunity assets according to `configs/v42_formal_source_registry_f2.yaml`.
- **New F2 evaluation:** new rainfall SHA only: Calibration 12, Locked 16, Challenge 12, Formal Blind >=24 by default. Same rainfall SHA may never cross a role.

## 1. Formal Prepare

Run the VS Code task `RTC: Formal F2 Prepare`.

Required outputs under `.../v42_paper/formal_f2/`:

- `prepare/FORMAL_F2_EVENT_LEDGER.csv`
- `prepare/FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet`
- `prepare/FORMAL_F2_STEP1_POOL_AUDIT.json`
- `prepare/FORMAL_F2_STEP2_METADATA_POOL.parquet`
- `step2/FORMAL_F2_STEP2_RAW_MANIFEST.parquet`
- `step2/FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json`
- `evaluation_plan/FORMAL_F2_EVALUATION_PLAN_AUDIT.json`

Do not proceed unless:

- Step1 target train rainfall groups >=65;
- raw Step2 groups >=69 so the fixed Step2 train/validation/internal-calibration split can retain >=65 train groups;
- Calibration/Locked/Challenge/Blind group counts meet the frozen plan;
- every rainfall-group overlap is zero;
- raw Step2 admission proves same state, same forcing, frozen physical model, actual readback, H120 and current engineering semantics.

## 2. Formal Step1

Run `RTC: Formal Step1`.

Seeds: 17, 42, 73. Split seed and sensor-layout seed remain 42. The models share exactly the same rainfall split. Do not promote the internal model-calibration holdout to paper Calibration.

Step1 evidence is not written yet. New F2 Calibration is still required for uncertainty/OOD calibration.

## 3. Formal Step2

Run `RTC: Formal Step2`.

The primary Step1 seed 42 creates the causal Step2 state history. Each decision state uses thirteen real sparse-GAT calls at `t-60, ..., t`; each call has its own preceding 60-minute sensor/action/rain history, so the history source must cover `t-120..t`.

The three surrogate seeds 17/42/73 use the same rainfall split and train the shared four-reference trajectory model. Missing outfall/storage/facility-flow targets are not replaced with zero.

## 4. Generate NEW F2 Calibration SWMM cases locally

Use `evaluation_plan/calibration_plan.json`. This step must use the authoritative Wuhan SWMM runner and the frozen network/Engineering36 contract. It may use the preselected responsive checkpoints in the plan but must not use Proposed/baseline outcomes to change the selected rainfalls or checkpoints.

For each calibration candidate row write at least:

- `rainfall_sha256`
- `event_id`
- `checkpoint_min`
- `case_id`
- `history_detail_path` (same-event/state, full causal history through the checkpoint)

The corresponding `completion.json` must expose strict Candidate, No-control, Dynamic Internal and Hold Previous detail trajectories and authoritative engineering/readback evidence.

Save the manifest at:

`.../formal_f2/calibration/authoritative_cases/calibration_case_manifest.csv`

Then run `RTC: Formal Calibration Data Bridge`.

## 5. Formal Calibration

Run `RTC: Formal Calibration`.

It must pass on NEW F2 Calibration rainfall groups, disjoint from all model-development rainfalls:

- Step1 uncertainty scale + OOD uncertainty limit;
- Step2 3-model ensemble PFV/Peak one-sided conformal UCB;
- PFV false-safe, Peak false-safe and joint false-safe <= frozen alpha (default 0.05).

## 6. Compile Formal training evidence

Run `RTC: Formal Training Evidence`.

This writes `step1_gat/evidence.json` and `step2_surrogate/evidence.json` only if the real multi-seed + new-event calibration chain passes. It cannot consume fast/development artifacts.

## 7. Step3 engineering evidence

Before compiling Step3, perform an authoritative execution/readback audit on the calibrated controller and write:

`.../formal_f2/calibration/STEP3_AUTHORITATIVE_ENGINEERING_AUDIT.json`

It must prove actual execution-derived bounds/binary/rate/ramp/dwell/interlock/Adaptive-K, Engineering36, H12, K<=8, no future SWMM truth, and exact model hashes. Then run `RTC: Formal Step3 Evidence`.

## 8. Paper workflow

The existing fail-closed `paper_workflow_v42.py` remains the authority. Execute in order:

1. true-state offline validation;
2. authoritative exact SWMM closed loop;
3. surrogate closed loop;
4. GAT-integrated closed loop;
5. Policy Lock;
6. Challenge;
7. one-shot Locked Validation;
8. Formal Blind (>=24 new rainfall SHA).

Formal Blind must run authoritative SWMM for Proposed, EFD, Auto-RBC, All-close, No-control, Internal rules and Hold Previous using the same physical model, rainfall, initial condition and metric definitions. EFD/Auto-RBC proxy results from the fast line are never formal evidence.

## 9. Mainline audit

Run `RTC: V4.2 Mainline Audit` only after all preceding evidence files exist. The audit must remain fail-closed; never edit gate booleans simply to make it pass.

## Restart policy

Fix the first failing stage and resume from that stage. Reuse frozen manifests/model checkpoints where their input hashes still match. Do not return to exhaustive 30k-file Phase-0 scanning, regenerate existing Train1600, or retrain Step1 when only a downstream stage failed.
