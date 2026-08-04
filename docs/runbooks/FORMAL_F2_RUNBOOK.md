# Project6 V4.2 Formal F2 runbook

This is the **paper-evidence** line. Qualification/fast artifacts remain development-only and can never authorize Formal evidence.

## Frozen scientific design

- Network: `data/wuhan_v8_storage_retrofit.inp`, rainfall-only/no-DWF, Engineering36.
- State record: 5 min. Control decision: 10 min. Causal history: `t-60,t-55,...,t` (13 frames). Prediction horizon: `t+10,...,t+120` (12 steps). Only the first 30 min of a candidate schedule may deviate from its frozen anchor; only the first 10-min action is executed before replanning.
- Split authority: rainfall SHA/fingerprint. The current generation requires zero rainfall-group overlap between model development and Calibration/Challenge/Locked/final held-out roles. Historical labels are lineage metadata only and are **not** a permanent split gate.
- Step1: sparse-sensor Temporal GAT state reconstruction only. No future hydraulic truth.
- Step2: shared four-reference hydraulic trajectory surrogate: Candidate / No-control / Dynamic Internal / Hold Previous. No-control means all Engineering36 settings are exactly 1.0 over H120. Real branch equivalence is allowed and audited.
- Default Step2 target contract is `CONTROL_CORE`: node depth + node flooding rate + storage volume + managed-facility flow. `outfall_flow` is optional. `FULL_HYDRAULIC` is an explicit extension and additionally requires real outfall-flow supervision. Missing targets are never zero-filled.
- Step3 canonical selector: `sewerrtc.control.pfvfirst_mpc_v42.decide_pfvfirst_mpc`.
  - hard PFV budget: `PFV_delta_UCB <= 100 m3 + 0.05*PFV_no_control`;
  - hard PFV-Core8 priority-depth UCB limit from the raw frozen INP;
  - hard K<=8/bounds/binary/rate/ramp/dwell/interlock/uncertainty/OOD/executability;
  - primary performance objective: minimize TFV relative to Dynamic Internal;
  - positive Peak excess relative to Dynamic Internal is penalized, **not** a zero-deterioration hard gate;
  - empty safe set or selection error executes the frozen legal fallback.

## Current-generation data roles

Formal Step1 may reuse physically compatible historical Wuhan trajectories assigned by the **current frozen ledger** to model development. Formal Step2 may reuse rows that pass current Raw Readmission and the selected target contract. Prior `historically_revealed` / `historically_reserved` labels are retained for provenance but do not by themselves ban a rainfall from this generation.

The frozen evaluation plan is:

- Calibration: exactly 12 rainfall groups;
- Challenge: at least 12;
- Locked Validation: at least 16, one-shot after Policy Lock;
- final held-out (`formal_blind` legacy path name): at least 24, after Policy Lock.

All four sets must be rainfall-SHA-disjoint from current model development and from each other. No post-evaluation exclusion or weight retraining is allowed for Challenge/Locked/final.

## 01-05 — Formal Prepare and precompute audit

Run from a shell (VS Code tasks are optional conveniences, not a requirement):

```powershell
$Py = '.\.venv\Scripts\python.exe'
& $Py -u scripts\run_v42_formal_f2.py --stage prepare --split-seed 42
& $Py -u scripts\audit_v42_formal_precompute_readiness.py --step2-target-contract CONTROL_CORE
```

Do not start GPU training unless the current artifacts prove: Step1 train diversity >=65; Raw Step2 >=69 groups; each selected state has >=3 distinct actual Candidate schedules; causal history source coverage >=69 groups with checkpoint>=120; CONTROL_CORE coverage >=69 groups; zero current split overlap; Calibration12/Locked16/Challenge12/final>=24.

## 06-08 — Formal Step1

```powershell
& $Py -u scripts\run_v42_formal_f2.py --stage step1 --seeds 17 42 73 --split-seed 42 --sensor-layout-seed 42
```

All seeds use the same rainfall split and sensor layout. Internal model-selection/calibration groups are not paper Calibration.

## 09-12 — Causal GAT bridge and Formal Step2

```powershell
& $Py -u scripts\run_v42_formal_f2.py --stage step2 --seeds 17 42 73 --primary-step1-seed 42 --step2-target-contract CONTROL_CORE
```

The history source is separate from Candidate outcome detail and must cover `checkpoint-120 ... checkpoint`. Thirteen real sparse-GAT calls reconstruct the states at `t-60 ... t`; the current frame is never repeated as fake history.

## 13-17 — Authoritative Calibration and training evidence

Generate the 12 frozen Calibration rainfall groups with the authoritative Wuhan SWMM runner, then build the calibration bridge. Both Step1 and Step2 calibration reports must contain **exactly the same 12 rainfall SHA values as the frozen ledger**. A partial 8/12 subset cannot authorize Formal.

Calibration safety quantities are PFV budget and priority-node depth. Peak is retained as a performance-error diagnostic/penalty term, not a hard safety label.

```powershell
& $Py -u scripts\prepare_v42_formal_calibration_data_f2.py ...
& $Py -u scripts\run_v42_formal_calibration12_f2.py --seeds 17 42 73 --primary-step1-seed 42
& $Py -u scripts\compile_v42_formal_training_evidence_strict_f2.py --seeds 17 42 73 --primary-seed 42
```

The underlying calibration modules may retain smaller diagnostic minima for non-paper diagnostics, but the Formal entrypoint and evidence compiler both fail closed unless Step1 and Step2 calibration equal the frozen ledger **12/12 exactly**.

## 18-19 — Step3 engineering evidence

The authoritative execution audit must derive Engineering36 legality, K, target writes, actual/current readback, cross-decision dwell, bounds/binary/rate/ramp/interlock, PFV budget and priority-depth safety from actual execution. It must use the frozen GAT/surrogate/calibration hashes. The production runner performs this execution audit before compiling Step3 evidence.

## 20-28 — Formal closed-loop campaign

Use **`scripts/run_v42_formal_production_f2.py`**. `scripts/run_v42_formal_paper_f2.py` is the internal restart/orchestration implementation; the production entrypoint injects the rule-free-plant/native-Internal-shadow safety runtime, the executed surrogate-only stage22, and expanded Policy-Lock hashing. Never run the development `run_v42_qualification_micro.py` as Formal evidence.

Before starting, create the local deployment manifests under:

`outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/evaluation_inputs/`

with one row for every already-frozen event in `calibration/challenge/locked_validation/formal_blind`. Each row must contain `event_id`, `rainfall_sha256`, `inp_path`, `rain_duration_min`, and `simulation_duration_min`. These manifests only resolve frozen rainfalls to local authoritative INP files; they are **not allowed to select or replace evaluation rainfalls**.

Run the zero-SWMM preflight first:

```powershell
& $Py -u scripts\run_v42_formal_production_f2.py --stage preflight
```

Then run stage by stage or as one supervised campaign:

```powershell
& $Py -u scripts\run_v42_formal_production_f2.py --stage all --device cuda --max-candidate-sequences 64
```

The ordered stages are:

1. Step3 authoritative execution/readback audit and evidence compile;
2. true-state offline diagnostic validation;
3. authoritative SWMM exact closed loop with true-state diagnostic state source and authoritative No-control/Internal/Hold references;
4. actual surrogate-state-feedback rolling closed loop initialized from the same authoritative prefix;
5. full sparse-GAT-integrated authoritative SWMM closed loop;
6. Policy Lock;
7. Challenge;
8. one-shot Locked Validation;
9. final held-out >=24 rainfall groups × seven strategies;
10. strict/mainline audit.

Final strategies are `Proposed`, `EFD`, `Auto-RBC`, `All-close`, `No-control`, `Internal`, and `Hold`. Every final strategy/event result must be authoritative SWMM. Formal No-control is physically all-open; Formal All-close is physically all-zero. `Internal` alone retains the frozen native SWMM `[CONTROLS]` rules. Proposed/EFD/Auto-RBC/No-control/All-close/Hold run on a physical-SHA-equivalent runtime clone with native `[CONTROLS]` disabled so the evaluated policy actually owns Engineering36. Proposed simultaneously advances a separate native-rule Internal shadow only to the **current** time; only current readback is exposed, and no future shadow state/action is used online. Before the first 120-min decision, the rule-free Proposed plant causally replays current Internal readback to reproduce a deterministic common control prefix.

The production runner is restartable and hash-ledgered. Reuse is permitted only when input/model/policy hashes still match. Long local runs should be launched by Codex/shell with persistent stdout/stderr/PID/exit markers; a chat-turn timeout must not restart a valid running process.

## Strict closure

After all evidence exists:

```powershell
& $Py -u scripts\audit_v42_formal_strict_f2.py
& $Py -u scripts\project6_v42_mainline.py
```

The strict audit independently rechecks exact Calibration12, Step2 target contract/supervision, all-open No-control, PFV-budgeted/priority-depth Step3 semantics, Policy-Lock lineage and the complete held-out workflow. Do not edit evidence booleans to make an audit pass.

## Restart policy

Fix the **first failed stage** and resume from there. Reuse manifests/checkpoints only when their input hashes match. Do not return to exhaustive legacy Phase-0 scanning, regenerate Train1600, retrain Step1 for a downstream-only failure, substitute future SWMM truth online, or weaken a safety/data gate to obtain a pass.
