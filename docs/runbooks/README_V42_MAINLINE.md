# Project6 V4.2 canonical scientific mainline

This runbook is the only final-paper execution order. Historical V3/V4.1/V4.2
assets remain reproducibility/baseline code and cannot authorize a later stage.

## Canonical chain

1. **Phase R0 – authoritative evidence pool**
   - CLI: `scripts/audit_v42_existing_swmm_pool.py`
   - strict post-processing: `sewerrtc/v4/v42_r0_strict.py`
   - discovery/cache refresh: `sewerrtc/v4/v42_r0_refresh.py`
   - numeric four-reference alignment: `scripts/audit_v42_case_alignment.py`
   - strict reusable pool: `scripts/build_v42_reusable_pool.py`
   - rainfall isolation: `scripts/build_v42_reuse_split_groups.py`
   - PFV-first continuation files (`candidate_then_internal/passive`) are auxiliary
     until authoritative provenance proves canonical DI/Hold semantics.
   - stale scan caches must use `--refresh-scan-cache`; `--resume-scan-cache`
     is valid only when the discovery fingerprint matches exactly.

2. **Step 1 – sparse-state reconstruction**
   - architecture: `sewerrtc/models/temporal_sparse_gat_v42.py`
   - R0 temporal windows: `scripts/build_v42_step1_windows.py`
   - online adapter: `sewerrtc/state/v42_sparse_state.py`
   - causal Step1->Step2 history bridge:
     `sewerrtc/state/v42_reconstructed_history.py`
   - formal contract: 13 x 5-min causal history, sparse depth/mask, rainfall,
     actual Engineering36 readback actions, topology, node/link static features.
   - each 5-min online Step-1 reconstruction is appended to the causal history
     buffer. Step 2 is enabled only after 13 real reconstructed frames exist.
     Current-frame repetition and SWMM-truth history substitution are forbidden.
   - formal evidence must prove new temporal-GAT training, rainfall-group
     isolation, uncertainty calibration, OOD calibration and model SHA.
   - historical single-snapshot GAT scores are background only.

3. **Step 2 – four-reference hydraulic trajectory surrogate**
   - model: `sewerrtc/v4/models_v42/hydraulic_multi_reference.py`
   - loss: `sewerrtc/v4/models_v42/hydraulic_trajectory_losses.py`
   - R0 bridge: `scripts/build_v42_r0_paper_dataset.py`
   - raw population target audit: `scripts/audit_v42_r0_hydraulic_targets.py`
   - raw Independent Oracle: `scripts/audit_v42_r0_independent_oracle.py`
   - training admission: `scripts/audit_v42_paper_training_admission.py`
   - input history is GAT-compatible causal **depth history** + actual readback;
     full-network flooding history is never an online input.
   - Candidate / No-control / Dynamic Internal / Hold Previous share one model.
   - PFV/TFV/Peak are deterministic derivatives of predicted flooding-rate
     trajectories, never free KPI heads.
   - formal counterfactual training is target no-DWF only; source-domain data is
     auxiliary.
   - Oracle and target audit must carry the same `sample_lineage_sha256`.

   **Not a formal paper trainer:** `sewerrtc/v4/v42_trainer.py` contains the
   historical depth/KPI-head training line. Keep it for reproducibility or
   ablation only. It must not create Step-2 formal evidence or Policy-Lock
   weights. A formal trainer must instantiate `MultiReferenceHydraulicSurrogate`
   and `HydraulicTrajectoryLoss` on the R0-derived manifest above.

4. **Step 3 – PFV-first rolling MPC**
   - selector: `sewerrtc/control/pfvfirst_mpc_v42.py`
   - authoritative execution adapter:
     `sewerrtc/control/pfvfirst_mpc_v42_authoritative.py`
   - formal candidate builder:
     `build_calibrated_authoritative_mpc_candidate`.
   - PFV hard safety: Candidate vs No-control.
   - Peak hard safety: Candidate vs Dynamic Internal.
   - PFV/Peak UCBs are computed as calibrated prediction mean + frozen `z*std`;
     uncertainty/OOD pass is derived from calibrated scores and thresholds.
     Caller-supplied UCB/pass flags are development-only and cannot authorize
     formal safety evidence.
   - TFV: minimize vs Dynamic Internal only inside the safe set.
   - H=12, Engineering36, K<=8, bounds/rate/ramp/dwell/interlock,
     uncertainty/OOD/executability, frozen hashed fallback.
   - execute only the first 10-min action; written/readback actions must match.

5. **Step 4 – closed-loop and blind evidence**
   - stage gate: `sewerrtc/v4/paper_workflow_v42.py`
   - full-chain gate: `scripts/project6_v42_mainline.py`
   - order: true-state offline -> Exact SWMM -> surrogate closed loop ->
     GAT-integrated closed loop -> Policy Lock -> Challenge -> Formal Blind.
   - Policy Lock binds policy, surrogate, temporal GAT and fallback hashes.
   - Challenge/Formal must reuse the exact locked hashes.
   - Formal Blind: >=24 independent new rainfall SHAs, no retraining, no
     post-reveal exclusion.

## Stop rule

Run `scripts/project6_v42_mainline.py` after every stage. The first failing stage
is the only authorized next task. Do not skip forward by reusing legacy evidence.

## Current implementation boundary

The formal temporal GAT architecture, formal four-reference surrogate/loss,
strict R0 bridge, causal reconstructed-history bridge, MPC selector/adapters and
evidence gates exist. This does **not** mean formal training/closed-loop evidence
has been executed. In particular, the new temporal GAT still needs a formal
trainer/calibration run, and the final Step-2 trainer must be wired to the
R0-derived dataset using `MultiReferenceHydraulicSurrogate + HydraulicTrajectoryLoss`
rather than the historical `v42_trainer.py` line.
