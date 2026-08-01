# V4.2 Fast Feasibility Pilot

This path is **development-only**. It answers an early go/no-go question before
spending days on formal Step-1 multi-seed/calibration, formal target-domain
Step-2 training, and Formal Blind.

It never writes a passing paper `evidence.json` and cannot authorize the formal
mainline.

## What it does

1. Reuses the strict R0 evidence pool; it never rescans 30k logical files.
2. Selects at most 64 `auxiliary_pretrain` rainfall groups for cheap Step-1
   representation pretraining. These groups remain auxiliary; no provenance is
   upgraded.
3. Uses at most 96 `eligible_source_domain_counterfactual_aux` four-reference
   cases to train the current `MultiReferenceHydraulicSurrogate` architecture on
   depth + flooding trajectories and trajectory-derived PFV/TFV/Peak deltas.
4. Replays PFV-first candidate selection on held-out rainfall groups. The model
   chooses using predictions; the selected action is then scored with the
   already-recorded authoritative SWMM trajectory.
5. If the replay is promising, the next required experiment is one new
   authoritative SWMM rolling closed-loop event with frozen pilot models.

## Why auxiliary Step-1 pretraining exists

The formal target population is intentionally small and strictly defined.
Auxiliary runs expose the temporal GAT to many more rainfall/hydraulic regimes,
so it can learn transferable graph and temporal representations before target
fine tuning. Auxiliary runs are never used for formal validation/calibration.
Negative transfer remains possible, so the formal workflow still requires an
explicit compatibility audit and target-only evidence.

## Fast commands

```powershell
cd E:\RTC_sewer\Project6
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$env:PYTHONPATH = "E:\RTC_sewer\Project6"

# Prepare only: cheap audit/materialisation.
& $Py .\scripts\run_v42_fast_feasibility.py --prepare-only

# Train quick Step-2 core model and run SWMM-backed policy replay.
& $Py .\scripts\run_v42_fast_feasibility.py --step2-epochs 6
```

For Step 1, first reuse a completed good LOEO A0 checkpoint if available. If a
fresh quick Step-1 model is needed, use the generated allow-list:

```powershell
$Fast = ".\outputs\project6_dual_reference_v4\final_v4\v42_paper\fast_feasibility"
$S1 = ".\outputs\project6_dual_reference_v4\final_v4\v42_paper\step1_gat"

& $Py .\scripts\train_v42_step1_streaming.py `
  --model-seed 42 `
  --sensor-layout-seed 42 `
  --split-seed 42 `
  --aux-sampling-seed 42 `
  --sensor-ratio 0.10 `
  --aux-pretrain `
  --aux-allowlist "$Fast\step1_fast_aux_allowlist.json" `
  --aux-epochs 1 `
  --aux-max-windows-per-group 8 `
  --aux-max-windows-per-run 2 `
  --epochs 6 `
  --patience 3 `
  --priority-weight 0 `
  --wet-priority-weight 0 `
  --nll-weight 0 `
  --output-dir "$Fast\step1_model"
```

## Interpretation

A positive offline replay is only a screening signal. It proves neither the
formal target-domain surrogate nor the integrated online chain. The next step is
one authoritative SWMM rolling event using causal Step-1 reconstructions,
future rainfall forecast, the quick Step-2 surrogate, and the canonical
PFV-first selector. Only after that micro experiment is positive should the
project spend compute on the full formal evidence chain.
