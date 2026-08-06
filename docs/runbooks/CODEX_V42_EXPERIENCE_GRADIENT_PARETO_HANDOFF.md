# Project6 V4.2 — Experience-guided differentiable MPC + PFV–TFV Pareto handoff

## Scientific flow

`sparse sensing -> causal full-state reconstruction -> differentiable action-effect surrogate -> authoritative experience guidance -> state-adaptive continuous gradient search -> Engineering36 projection/dedup -> rolling PFV-UCB hard admission -> minimum TFV -> execute first 10 min -> write/readback -> replan`

The online hydraulic contract remains `PFV_candidate_total <= 1.05 * PFV_no_control_total + 100 m3`.  The PFV-relaxation/Pareto analysis is sensitivity analysis only and must not silently change the online safety contract.

## New code

- `sewerrtc/control/experience_bank_v42.py`
  - causal state signatures;
  - nearest-state authoritative warm-start retrieval;
  - no future truth online.
- `sewerrtc/control/differentiable_hybrid_search_v42.py`
  - exact binary outer modes;
  - state-adaptive facility selection by Step2 gradient;
  - coarse H3-constant spatial optimisation;
  - bounded H3 temporal refinement;
  - search barrier is proposal guidance only, not safety authority.
- `sewerrtc/v4/v42_experience_gradient_runtime.py`
  - coverage + experience + gradient candidates;
  - delegates projection/PFV-UCB/min-TFV to the corrected selector.
- `scripts/build_v42_authoritative_experience_bank.py`
  - recomputes all historical labels from authoritative detail.csv;
  - legacy stored KPI labels are ignored.
- `sewerrtc/control/pfv_tfv_tradeoff_v42.py` + `scripts/audit_v42_pfv_tfv_tradeoff.py`
  - state-wise PFV–TFV Pareto frontier;
  - contract grid;
  - marginal TFV benefit per extra PFV m3;
  - descriptive knee points.
- `scripts/run_v42_experience_gradient_production.py`
  - opt-in production-compatible entrypoint;
  - experience bank hash is included in policy lineage.

## Local validation order

1. Checkout `feat/v42-experience-gradient-pareto-r1`. Do not reset/clean unrelated user work.
2. Run `py_compile` on every new Python file.
3. Run:
   - `tests/test_v42_experience_bank.py`
   - `tests/test_v42_differentiable_hybrid_search.py`
   - `tests/test_v42_pfv_tfv_tradeoff.py`
   - existing rolling-PFV/candidate-selector focused tests.
4. Build the canonical historical bank from the Round2 combined candidate manifest and the frozen Step2 state manifest.  The expected output path used automatically by the runtime is:
   `outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/diagnostics/experience_gradient/AUTHORITATIVE_EXPERIENCE_BANK.parquet`
5. Require zero or fully explained bank-build failures before using the bank online.  Check state/action dedup and candidate/detail lineage hashes.
6. Run `audit_v42_pfv_tfv_tradeoff.py` on that bank.  Do not run SWMM for this audit.
7. Reconcile old Round2 stored-label reports against the canonical bank; from this point use only shared authoritative detail-derived metrics.
8. Before any closed loop, run a bounded gradient-recovery diagnostic on the eight frozen FAST-direct states.  The key target is whether the hybrid search can rediscover high-value regions such as the T15_D105 candidate-space gap without reading direct-SWMM answers online.
9. Compare local Step2 gradient direction against authoritative finite perturbations.  If gradient direction/ranking is unreliable, stop and improve Step2 before closed-loop use.
10. Recompute independent PFV calibration for any new Step2 checkpoint.  Never shrink the conformal margin merely to make the safe set non-empty.
11. Run only 3–5 development events first.  Require causal state source, non-empty PFV-admitted set where physically available, target write/readback, rolling PFV ledger consistency and positive TFV direction.
12. Run an independent pre-Policy-Lock PFV safety audit on untouched rainfall groups.
13. Policy Lock must freeze Step1, Step2, PFV calibration, experience bank, gradient optimiser, projection, selector and runtime hashes.
14. Only after lock: Challenge -> Locked -> Final/Blind.

## PFV–TFV trade-off interpretation

For each candidate define:

- PFV cost: `PFV_candidate - PFV_no_control`;
- TFV benefit: `TFV_internal - TFV_candidate`.

A candidate is Pareto-efficient if no other candidate has both lower/equal PFV cost and higher/equal TFV benefit with at least one strict improvement.

The contract grid scans relative margins `0–15%` and absolute margins `0–2000 m3`.  Report both admitted-state summaries and all-state summaries with unavailable states contributing zero controllable benefit.  This avoids changing medians merely because the population of admitted states changed.

The online paper contract remains 5% + 100 m3 unless a separate scientific decision explicitly changes and re-freezes it.

## Hard stops

Do not:

- use Challenge/Locked/Final events in the experience bank or active learning;
- feed future SWMM states or realised future rain into retrieval/gradient search;
- let the gradient barrier replace the final PFV-UCB gate;
- drop mandatory global Engineering36 coverage;
- claim a mathematical global optimum from the gradient solution;
- interpret the PFV–TFV Pareto sensitivity scan as permission to relax safety online;
- run Final before local compile/tests, canonical bank, gradient validation and independent PFV calibration pass.
