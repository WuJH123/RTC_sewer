# V4.2 Step-1 streaming training runbook

This runbook is the authoritative local execution path for the large Step-1
window manifest.  Do **not** use the eager dataset/trainer for the ~627k-window
population.

## Why streaming is required

The window manifest contains metadata for many overlapping 13x5-minute windows.
Materialising every `[13, 932]` depth/mask tensor and keeping thousands of source
CSV DataFrames resident can consume tens of GB of RAM.  The streaming dataset
keeps only manifest metadata in memory and reads one physical detail file at a
time.

The CSV projection is name-authoritative:

1. read the file header;
2. verify every required column exists;
3. `pd.read_csv(..., usecols=<column names>)`;
4. reorder explicitly to the canonical required-column list;
5. fail closed on missing/non-finite hydraulic inputs.

## Files

- `sewerrtc/v4/v42_step1_streaming.py` — bounded-memory iterable dataset,
  deterministic target split utilities and balanced auxiliary sampling.
- `scripts/train_v42_step1_streaming.py` — streaming trainer, streaming NSE/RMSE
  metrics, independent model/sensor/split seeds and RMSE-based checkpointing.
- `tests/test_v42_step1_streaming.py` — reordered-column and split regressions.

## Local sync

```powershell
cd E:\RTC_sewer\Project6
git fetch origin
git switch fix/v42-step1-streaming-formal-r3
git pull --ff-only origin fix/v42-step1-streaming-formal-r3

$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$env:PYTHONPATH = "E:\RTC_sewer\Project6"

& $Py -m pytest -q tests/test_v42_step1_streaming.py tests/test_v42_step1_formal_integrity.py --tb=short
& $Py -m pytest -q tests -k "v42 or step1" --tb=short
```

## First development run: target-only A0

Before priority weighting, uncertainty NLL or auxiliary pretraining, establish a
clean fixed-sensor target-only baseline.

List the four formal target rainfall groups from the existing manifest, then run
one leave-one-event-out fold at a time:

```powershell
& $Py .\scripts\train_v42_step1_streaming.py `
  --model-seed 42 `
  --sensor-layout-seed 42 `
  --split-seed 42 `
  --sensor-ratio 0.10 `
  --validation-group <ONE_TARGET_GROUP> `
  --no-reserve-calibration `
  --priority-weight 0 `
  --wet-priority-weight 0 `
  --nll-weight 0 `
  --epochs 20 `
  --patience 6 `
  --batch-size 16 `
  --num-workers 0 `
  --output-dir <FOLD_OUTPUT_DIR>
```

Repeat for all four target groups.  The trainer prints and records the exact
`expected_windows` and `actual_windows_seen`; these must match.  A count such as
millions of "windows" for a ~1200-window train split is an evidence-integrity
bug and must not be accepted.

## Priority and uncertainty ablations

Only after the A0 leave-one-event-out baseline is understood:

1. compare `--priority-weight 0/2/3/5` with everything else frozen;
2. add uncertainty using `--nll-weight 0.25 --nll-warmup-epochs 5
   --nll-ramp-epochs 5`;
3. keep checkpoint selection on validation overall-unobserved RMSE while NLL
   weight changes;
4. do not claim calibrated uncertainty yet.

## Auxiliary pretraining

Auxiliary pretraining is **off by default**.  It can only be enabled with an
explicit rainfall-group allow-list produced by a provenance/domain compatibility
audit:

```powershell
& $Py .\scripts\train_v42_step1_streaming.py `
  ... `
  --aux-pretrain `
  --aux-allowlist .\outputs\...\compatible_aux_groups.json `
  --aux-max-windows-per-group 16 `
  --aux-max-windows-per-run 4
```

Unknown/source-domain windows must never enter formal target validation or
calibration.

## Seed contract

Formal multi-seed experiments later vary only `--model-seed` (0..4).  Keep
`sensor-layout-seed`, `split-seed` and `aux-sampling-seed` frozen so model-seed
variation is not confounded with sensor placement or event split changes.

## Formal stopping rule

This streaming runner writes a training report and checkpoints, but deliberately
sets `formal_evidence_ready=false`.  Do not create a passing Step-1
`evidence.json` until target-event diversity, multi-model-seed robustness,
uncertainty calibration and OOD calibration have all passed.
