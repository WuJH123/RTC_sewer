# Codex Handoff — Project6 V4.2 Fast E2E 64+ Rainfall Potential Test

You are taking over a long-running urban drainage real-time-control research
project. Do not treat this as a generic refactor. Your job in this session is to
run, debug and scientifically audit the **development-only 64+ rainfall end-to-end
potential test** that has already been implemented on GitHub.

## Repository and local environment

- GitHub: `https://github.com/WuJH123/RTC_sewer.git`
- local root: `E:\RTC_sewer\Project6`
- Python: `E:\RTC_sewer\Project6\.venv\Scripts\python.exe`
- output root: `E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\final_v4`
- implementation branch: `fix/v42-fast-e2e-64plus-baselines-r1`

Do not use `git reset --hard` or `git clean -fd`. Do not delete historical
outputs. If there are local uncommitted changes, report them and preserve them.
Do not regenerate historical SWMM data unless a specific missing physical branch
is proven to block this bounded development run.

## Scientific question for this run

Before spending days on formal multi-seed training, determine whether the actual
method has credible real-time-control potential:

`sparse sensors -> temporal GAT current-state reconstruction -> causal 13-frame
reconstructed history -> trajectory-first four-reference hydraulic surrogate ->
PFV-first safe candidate selection -> authoritative recorded SWMM outcome`

The run must compare the resulting Proposed strategy against:

1. No-control;
2. native / Dynamic Internal SWMM rules;
3. Hold-Previous / passive;
4. Equal Filling Degree (EFD) development baseline;
5. Auto-RBC development baseline;
6. All-close negative control.

This entire path is `development_only=true`. It must never create or copy a
passing formal `evidence.json`.

## Frozen scientific contracts

- network: `data/wuhan_v8_storage_retrofit.inp`;
- Engineering36 order comes from the repository contract, never a new handwritten
  ordering;
- Step1: 13 x 5-min causal history, fixed 10% sensor layout for this debug run;
- Step2 prediction: H=12 x 10 min = 120 min;
- Step3 control horizon: first 3 x 10-min steps may differ; execute only first
  10-min step; steps 4-12 are frozen tail in newly generated development baseline
  schedules;
- K<=8 for Proposed/EFD/Auto-RBC screening actions;
- ADD301.2 and ADD301.3 are binary pumps; do not make their executed setting
  fractional;
- PFV hard reference: No-control;
- Peak hard reference: Dynamic Internal;
- TFV is optimized only after PFV/Peak safety admission;
- future realised rainfall is forbidden as an online Proposed input.

## Data policy for this session

Do **not** spend this session trying to recover every historical file or perfect
all old provenance metadata. Use enough high-confidence strict data to test the
method.

Required sampling policy:

- default select 96 independent rainfall fingerprints;
- Step1 must see at least 64 independent auxiliary rainfall fingerprints;
- Step2 **training** must contain at least 65 independent rainfall groups (more
  than 64); the final audit enforces this;
- prefer historical V4/Train1600-like evidence where available;
- fill any shortfall only from existing strict finite/aligned
  `eligible_source_domain_counterfactual_aux` data;
- never relabel auxiliary/source data as formal target data;
- each selected Step2/Step3 state must retain at least 3 candidate cases;
- decision checkpoint must be >=120 min to support 13 genuine Step1
  reconstructions at t-60..t, each with its own preceding 60-min observation
  history.

## New files you must understand before running

Read these fully:

- `docs/runbooks/README_V42_FAST_E2E_64PLUS.md`
- `sewerrtc/v4/v42_fast_e2e.py`
- `sewerrtc/v4/v42_fast_e2e_warm.py`
- `scripts/run_v42_fast_e2e_64plus.py`
- `scripts/materialize_v42_fast_gat_history.py`
- `scripts/evaluate_v42_fast_baselines.py`
- `scripts/audit_v42_fast_e2e.py`
- `tests/test_v42_fast_e2e.py`

Also inspect the reused existing implementation:

- `scripts/train_v42_step1_streaming.py`
- `scripts/train_v42_step2_fast.py`
- `scripts/evaluate_v42_fast_policy_replay.py`
- `sewerrtc/models/temporal_sparse_gat_v42.py`
- `sewerrtc/v4/models_v42/hydraulic_multi_reference.py`
- `sewerrtc/control/pfvfirst_mpc_v42.py`

## Mandatory execution order

### 0. Git and code preflight

Run:

```powershell
cd E:\RTC_sewer\Project6
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$env:PYTHONPATH = "E:\RTC_sewer\Project6"

git status -sb
git fetch origin
git branch --show-current
git log -1 --oneline
```

Switch to/pull `fix/v42-fast-e2e-64plus-baselines-r1` only without overwriting
unrelated local edits.

Then run syntax/import/unit checks before touching the large data:

```powershell
& $Py -m pytest .\tests\test_v42_fast_feasibility.py .\tests\test_v42_fast_e2e.py -q
& $Py -m py_compile `
  .\sewerrtc\v4\v42_fast_e2e.py `
  .\sewerrtc\v4\v42_fast_e2e_warm.py `
  .\scripts\materialize_v42_fast_gat_history.py `
  .\scripts\evaluate_v42_fast_baselines.py `
  .\scripts\run_v42_fast_e2e_64plus.py `
  .\scripts\audit_v42_fast_e2e.py
```

If any check fails, diagnose the real root cause, make the smallest fix, add or
adjust a regression test, rerun the checks, commit to the same branch, and only
then continue.

### 1. Prepare-only data gate — no training yet

Run:

```powershell
& $Py .\scripts\run_v42_fast_e2e_64plus.py `
  --target-rainfall-groups 96 `
  --min-rainfall-groups 64 `
  --candidates-per-state 3 `
  --min-checkpoint-min 120 `
  --prepare-only
```

Read:

`outputs\project6_dual_reference_v4\final_v4\v42_paper\fast_e2e_64plus\FAST_E2E_PREPARE.json`

Do not train unless:

- `step1_selected_aux_groups >= 64`;
- `step2_selected_groups` is large enough that an 80/20 grouped split will leave
  **>=65 training groups**. With the current splitter this normally means use the
  default ~96 selected groups; do not accept a nominal 64 total groups as proof
  of 64+ training groups;
- `step2_selected_states == step2_selected_groups`;
- `step2_selected_cases >= 3 * step2_selected_states`;
- no reserved evaluation data is present;
- `minimum_checkpoint_min = 120`;
- Train1600-like preferred count and any non-preferred fill are explicitly
  reported.

If this gate fails, first increase the bounded reusable source pool or adjust the
selection while preserving rainfall isolation and candidate multiplicity. Do not
fall back to the old four-group LOEO line.

### 2. Run the bounded integrated chain

If prepare passes, run:

```powershell
& $Py .\scripts\run_v42_fast_e2e_64plus.py `
  --target-rainfall-groups 96 `
  --min-rainfall-groups 64 `
  --candidates-per-state 3 `
  --min-checkpoint-min 120 `
  --step1-epochs 6 `
  --step2-epochs 6 `
  --seed 42
```

Do not kill a healthy process just because an epoch is slow. Monitor the existing
heartbeat output. Stop only for traceback, OOM/dangerous memory growth, a
scientific contract failure, or no CPU/IO/progress for roughly 10 minutes.

### 3. Verify the actual Step1 -> Step2 bridge

Read:

`fast_e2e_64plus\step2_fast_e2e_gat_history_audit.json`

It must show:

- `state_source = gat_sparse_reconstruction`;
- `reconstructed_history_contract = 13_real_step1_calls_at_5min_spacing`;
- `current_frame_repetition_used = false`;
- `authoritative_swmm_history_used_as_online_input = false`;
- `realized_future_rainfall_used_online = false`;
- causal rainfall forecast authority;
- integrated rainfall groups >=64.

If any of these are false, stop. Do not interpret later controller metrics.

### 4. Verify Step2 before trusting controller replay

Read:

`fast_e2e_64plus\step2_model\fast_step2_report.json`

At minimum report:

- number of unique train/validation rainfall groups;
- `train_rainfall_groups >= 65`;
- depth NSE/RMSE;
- flooding MAE;
- PFV/TFV/Peak delta MAE and sign accuracy.

If training groups <=64, this run does not meet the task even if the model metrics
look good. Increase the selected total population and rerun from the cheapest
affected stage.

### 5. Verify PFV-first replay

Read:

`fast_e2e_64plus\policy_replay\fast_policy_replay_summary.json`

Report:

- number of replayed independent states/rain groups;
- fallback rate;
- actual safety rate;
- false-safe rate;
- Proposed PFV vs No-control;
- Proposed TFV vs Internal;
- Proposed Peak vs Internal;
- go signal.

A high depth NSE cannot compensate for a high false-safe rate.

### 6. Baseline comparison

Read:

`fast_e2e_64plus\baseline_comparison\FAST_E2E_BASELINE_COMPARISON.csv`

and JSON/row-level files.

Return one final table containing at least:

| Strategy | PFV | PFV reduction vs NC | TFV | TFV reduction vs Internal | Peak | Peak reduction vs Internal | SWMM-backed fraction | Notes |

for:

- Proposed;
- EFD;
- Auto-RBC;
- All-close;
- No-control;
- Internal rule;
- Hold-Previous.

Important authority rule: No-control/Internal/Hold and the selected Proposed
historical candidate are authoritative recorded SWMM outcomes. EFD/Auto-RBC/
All-close are authoritative only when a close historical action proxy exists;
otherwise their rows are surrogate screening estimates. Never hide this
 distinction.

### 7. Final fail-closed audit

Run:

```powershell
& $Py .\scripts\audit_v42_fast_e2e.py
```

Read:

- `FAST_E2E_AUDIT.json`
- `FAST_E2E_AUDIT.md`
- `FAST_E2E_VERDICT.json`

Do not claim this debug line is coherent unless `FAST_E2E_AUDIT.json` says
`passed=true`. The audit explicitly requires Step2 training on >64 independent
rainfall groups and verifies the causal/no-leak bridge and complete baseline
suite.

## How to react to failure without wasting compute

Fix only the first real blocker:

- prepare/data selection failure -> selection/manifests only, no training;
- Step1 failure -> Step1 only;
- causal GAT history failure -> history materializer/checkpoint selection only;
- Step2 prediction failure -> Step2 only, reuse frozen Step1/GAT histories;
- false-safe / poor selection -> candidate/Step2/selector diagnosis, do not
  retrain Step1 automatically;
- baseline proxy coverage poor -> this does not invalidate Proposed; mark it and
  move the affected baseline to the authoritative SWMM micro-test.

Never lower the safety gate to manufacture a GO result.

## Decision after the fast run

If `FAST_E2E_AUDIT passed=true` and `potential_go=true`, do **not** jump to Formal
Blind. The next authorized computation is one new authoritative rolling SWMM
micro closed-loop event with the exact same event/network/timing for:

- Proposed GAT-Surrogate-PFV-first MPC;
- EFD with a frozen Wuhan storage/control-zone mapping;
- Auto-RBC;
- All-close negative control;
- No-control;
- Native Internal rules.

Use actual write/readback and the same engineering projector. If that micro-test
also shows PFV/Peak safety and TFV benefit, then return to the formal V4.2
mainline and decide whether multi-seed Step1/Step2 training is worth the compute.

## Required final response to the user

Do not just say "completed". Return:

1. branch and final commit;
2. files modified during local debugging;
3. pytest/py_compile results;
4. prepare counts and actual Step2 training rainfall-group count;
5. Step1 metrics;
6. Step2 trajectory/KPI metrics;
7. PFV-first false-safe/fallback results;
8. complete baseline comparison table;
9. whether EFD/Auto-RBC rows were SWMM-backed or surrogate-screened;
10. `FAST_E2E_AUDIT` PASS/FAIL;
11. `potential_go` true/false;
12. the first authorized next computation and why.

Commit every scientifically meaningful fix with a regression test. Do not merge
into `main` until the bounded execution line has passed locally and its output
has been reviewed.
