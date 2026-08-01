# V4.2 64+ Rainfall Fast End-to-End Potential Test

This is the canonical **development-only** execution line for deciding whether
Project6 has enough real-time-control potential to justify expensive formal
training. It does not replace or weaken `README_V42_MAINLINE.md`.

## Why this line exists

The historical four-target-group LOEO development path is too small to answer a
generalisation/control-potential question. Conversely, running the complete
formal multi-seed/calibration/blind chain before the integrated controller works
is wasteful. This line therefore uses a bounded but diverse population:

- target selection: **96 independent rainfall fingerprints** by default;
- hard lower bound: **64 rainfall fingerprints**;
- historical V4/Train1600-like evidence is preferred when present;
- strict finite/aligned source-domain control-core evidence may fill the remaining
  slots without being promoted to formal target data;
- each Step2/Step3 state retains at least three candidate cases so MPC replay has
  a real action choice;
- checkpoints must be at least 120 min so the 13 reconstructed Step1 states can
  each be produced from their own 60-min causal observation history.

## Integrated causal chain

The run is deliberately more demanding than the original fast pilot:

1. Step1 sees >=64 independent auxiliary rainfall groups during representation
   pretraining, followed by the existing target-domain development fine-tuning.
2. For every Step2 decision state, thirteen actual Step1 calls are made at
   `t-60, t-55, ..., t`. The Step2 input is therefore a real reconstructed
   history, not SWMM truth and not `current_GAT_state x 13`.
3. Realised future rainfall is retained only as a diagnostic/label. The online
   forcing supplied to Step2 is a causal persistence/decay forecast built from
   rainfall observed through decision time.
4. Step2 trains the existing shared `MultiReferenceHydraulicSurrogate` on
   Candidate / No-control / Dynamic-Internal / Hold-Previous trajectories.
5. PFV-first replay selects candidates from predictions; the selected historical
   candidate is then scored using its recorded authoritative SWMM trajectory.

## Baselines

The fast comparison always includes:

- Proposed GAT–Surrogate–PFV-first MPC;
- No-control;
- Native/Dynamic Internal rules;
- Hold-Previous / passive reference;
- Equal Filling Degree (EFD) development baseline;
- Auto-RBC development baseline;
- All-close negative control.

EFD follows the standard equal-filling principle: use normalized filling degree
and coordinate releases relative to the system filling level. In this fast line,
the current graph-local normalized filling signal is used as a screening proxy,
and EFD/Auto-RBC pass the same H12/H3 and K<=8 perturbation budget. If a close
historically simulated candidate exists, its SWMM result is used. Otherwise the
fast surrogate estimates the baseline outcome and the row is explicitly labelled
`surrogate_screen_no_close_recorded_action`.

This mixed authority is acceptable only for **development screening**. A positive
result authorizes one new authoritative SWMM micro closed-loop where Proposed,
EFD, Auto-RBC, All-close, No-control and Internal are all run on the same event.

## Cheap first command

Always run preparation first. It reads manifests and materialises a bounded
subset but does not train a model.

```powershell
cd E:\RTC_sewer\Project6
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$env:PYTHONPATH = "E:\RTC_sewer\Project6"

& $Py .\scripts\run_v42_fast_e2e_64plus.py `
  --target-rainfall-groups 96 `
  --min-rainfall-groups 64 `
  --candidates-per-state 3 `
  --min-checkpoint-min 120 `
  --prepare-only
```

Read:

`outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus/FAST_E2E_PREPARE.json`

Do **not** start training unless all of the following hold:

- Step1 selected auxiliary groups >=64;
- Step2 selected/materialized rainfall groups >=64;
- each selected state has >=3 candidate cases;
- no reserved evaluation data was admitted;
- the preferred Train1600-like count and any non-preferred fill are reported.

## Full development run

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

If a compatible Step1 checkpoint already exists, avoid retraining it:

```powershell
& $Py .\scripts\run_v42_fast_e2e_64plus.py `
  --target-rainfall-groups 96 `
  --min-rainfall-groups 64 `
  --candidates-per-state 3 `
  --reuse-step1-model-dir "<compatible TemporalSparseGAT model directory>" `
  --step2-epochs 6
```

## Required outputs

The final decision files are:

- `FAST_E2E_VERDICT.json`
- `policy_replay/fast_policy_replay_summary.json`
- `baseline_comparison/FAST_E2E_BASELINE_COMPARISON.csv`
- `baseline_comparison/FAST_E2E_BASELINE_COMPARISON.json`
- `baseline_comparison/FAST_E2E_BASELINE_ROWS.csv`

The comparison table reports PFV, TFV and peak flooding rate plus reduction
relative to the scientifically relevant references. Every EFD/Auto-RBC/All-close
row also reports whether its outcome is authoritative SWMM or surrogate-only.

## GO / NO-GO interpretation

`potential_go=true` is only a development signal. It means the integrated chain
has shown enough promise to justify **one authoritative SWMM rolling micro-test**
with the complete baseline suite.

`potential_go=false` means do not spend compute on multi-seed formal training.
Use the saved artifacts to isolate whether the bottleneck is Step1 reconstruction,
Step2 trajectory/safety prediction, candidate diversity, or PFV-first selection.

## Literature basis for the screening EFD

The fast EFD implementation follows the established qualitative principle used
in urban drainage RTC: filling degree is a normalized depth/volume of storage,
and control coordinates releases around the system average filling degree. See:

- Rimer et al. (2023), *pystorms: A simulation sandbox for the development and
  evaluation of stormwater control algorithms*, Environmental Modelling &
  Software 162, 105635.
- Kroll et al., work on Equal Filling Degree storage control and integrated RTC.

Before any paper claim, replace the graph-local screening mapping by a frozen
Wuhan storage/control-zone-to-actuator mapping and rerun every baseline through
one authoritative SWMM execution engine.
