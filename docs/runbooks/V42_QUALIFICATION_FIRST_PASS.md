# V4.2 qualification-first execution

## Purpose

Use a small, diverse, already admitted development population to exercise the
complete Project6 V4.2 software chain before the expensive Formal production
run.  Qualification is a wiring/runtime/scientific-potential pass only.  It is
always `development_only=true`, writes to a separate output root, never consumes
untouched Formal evaluation rainfalls, and can never authorize Formal evidence.

The Formal F2 line and all of its thresholds remain unchanged.

## Why this is faster

The current Formal assets contain 82 Step1 train rainfall groups and 81 admitted
Step2 rainfall groups.  The qualification profile selects:

- 69 Step1 target-train rainfall groups, enough for the trainer to reserve an
  internal holdout and retain 65 train groups;
- at most 64 temporally spread windows per rainfall and two windows per physical
  run;
- 69 Step2 rainfall groups;
- one state and three distinct actual Candidate schedules per rainfall;
- revealed development rainfalls for micro Calibration/Challenge/Locked/Blind
  qualification, never the untouched Formal plans.

This keeps independent-event diversity while reducing repeated windows and
candidate rows by more than an order of magnitude.

## Current qualification core

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_v42_qualification_first_pass.py --stage core
```

The command is restartable and performs:

1. reuse and validate Formal units 01-05;
2. build isolated qualification Step1/Step2 manifests;
3. run Step1 seeds 17/42/73 for one epoch each;
4. materialise the 13-frame causal GAT history using pre-action state semantics;
5. run Step2 seeds 17/42/73 for one epoch each;
6. write `QUALIFICATION_28_STAGE_STATUS.json`.

All outputs live under:

```text
outputs/project6_dual_reference_v4/final_v4/v42_paper/qualification_first_pass/
```

Formal outputs under `formal_f2/` are read-only inputs and are never overwritten.

## Causal history correction

The qualification GAT bridge excludes the checkpoint action from its state
signature.  Candidate details may already contain the new action at `t`, while a
whole-event history source still contains the pre-action setting.  Same-state
matching therefore uses:

- checkpoint depth;
- rainfall history through `t`;
- actual actions through `t-5`;
- exact history coverage `t-120..t`.

It still performs thirteen real Step1 calls at `t-60,...,t`.  Current-frame
repetition, authoritative full-state history as online input, and realised
future rainfall remain forbidden.

## Deferred Formal promotion work

The qualification Step2 model uses the already complete core hydraulic targets
(depth/flooding/storage/facility-flow availability is checked upstream) but does
not claim explicit outfall-flow supervision.  Formal production remains blocked
until real `outfall_flow:<outfall>` targets are recovered or generated and the
Formal trainer actually supervises them.

Qualification results must never be copied into:

- `v42_paper/step1_gat/evidence.json`;
- `v42_paper/step2_surrogate/evidence.json`;
- Formal Policy Lock, Locked Validation, or Formal Blind evidence.

## Completing all 28 units quickly

After the qualification core (01-12) passes, implement/run micro authoritative
qualification stages 13-28 under the same qualification root:

- 2 revealed development Calibration events;
- 2 Challenge events;
- 2 Locked-like events;
- 4 pseudo-Blind development events;
- all seven strategies on the four pseudo-Blind events using authoritative SWMM.

These events are already revealed development rainfalls and therefore do not
consume the untouched Formal plans.  Each event-strategy result is cached in a
small execution ledger and reused when input/model/policy hashes match.

The purpose is to expose missing runners, schema mismatches, engineering issues,
and runtime bottlenecks before the full 12/16/12/24 Formal production campaign.

## Promotion sequence

Only after the qualification 28-stage pass succeeds:

1. recover/generate explicit outfall-flow targets for at least the full Formal
   training population;
2. run full 3-seed Formal Step1 and Step2 production training;
3. generate the new untouched Calibration cases;
4. calibrate uncertainty/OOD and PFV/Peak safety;
5. run the mandatory Formal closed-loop, Policy Lock, Challenge, Locked
   Validation, and >=24-event seven-strategy Formal Blind campaign;
6. run the fail-closed V4.2 mainline audit.

A qualification PASS demonstrates executable software, not a scientific Formal
PASS.
