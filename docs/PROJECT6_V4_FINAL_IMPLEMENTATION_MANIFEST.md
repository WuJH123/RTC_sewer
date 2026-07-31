# Project6 V4 Final implementation manifest

Date: 2026-07-27

This manifest records the code-only Final-V4 delivery. No long-running
Opportunity, Peak Boundary, Pilot400, Train1600, closed-loop, Challenge, or
Formal Blind task was executed.

## Unified entrypoints

- `scripts/project6_v4_final.py`
- `scripts/project6_runs/RUN_PROJECT6_V4_FINAL.ps1`
- `configs/wuhan_project6_v4_final.yaml`
- `docs/contracts/PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json`

## Final-V4 modules

- `sewerrtc/v4/contracts.py`
- `sewerrtc/v4/inventory.py`
- `sewerrtc/v4/opportunity.py`
- `sewerrtc/v4/candidates.py`
- `sewerrtc/v4/simulation.py`
- `sewerrtc/v4/manifests.py`
- `sewerrtc/v4/labels.py`
- `sewerrtc/v4/peak_boundary.py`
- `sewerrtc/v4/pilot.py`
- `sewerrtc/v4/active_learning.py`
- `sewerrtc/v4/training.py`
- `sewerrtc/v4/closed_loop.py`
- `sewerrtc/v4/evaluation.py`
- `sewerrtc/v4/reporting.py`
- `sewerrtc/v4/runtime.py`
- `sewerrtc/v4/pipeline.py`

`sewerrtc/v4/simulation.py` is a fail-closed facade over
`sewerrtc/simulation/pyswmm_runner.py`; it does not copy the SWMM runner.

## Existing modules repaired

- `sewerrtc/simulation/kpi_metrics.py`: H120 uses
  `checkpoint < elapsed_min <= checkpoint + 120`.
- `sewerrtc/control/v4_candidate_generator.py`: common projection includes
  no-reversal evidence and the Final-V4 family wrapper supplies the complete
  family registry.

## Tests added or extended

- `tests/test_v4_final_contract.py`
- `tests/test_v4_no_dwf_network.py`
- `tests/test_v4_parallel_runtime.py`
- `tests/test_v4_window_kpis.py`
- `tests/test_v4_candidate_generator.py`
- `tests/test_v4_schedule_projection.py`
- `tests/test_v4_action_authority.py`
- `tests/test_v4_opportunity.py`
- `tests/test_v4_opportunity_chain.py`
- `tests/test_v4_peak_boundary.py`
- `tests/test_v4_pilot_plan.py`
- `tests/test_v4_dataset_manifest.py`
- `tests/test_v4_event_split.py`
- `tests/test_v4_active_learning.py`
- `tests/test_v4_training.py`
- `tests/test_v4_closed_loop.py`
- `tests/test_v4_policy_lock.py`
- `tests/test_v4_formal_blind.py`
- `tests/test_v4_reporting.py`
- `tests/test_v4_resume.py`
- `tests/test_v4_final_labels.py`
- `tests/test_v4_final_stage_registry.py`
- `tests/test_oracle_pareto_v4.py`: corrected the test import path to the
  authoritative script.

## Verification completed

- Python compilation: passed.
- Final stage registry: 56 stages.
- All Project6 V4-related tests discovered by filename: 189 passed.
- Static `AuditContracts` through the PowerShell runner: exit 0.
- Active DWF FLOW rows: 0.
- Canonical Engineering36 count and uniqueness: 36.
- Network, physical-network, action-order, and facility-semantics SHA checks:
  passed.

Runtime validation was limited to the tiny four-row process-pool echo fixture
and the existing 30-minute synthetic tiny-network SWMM integration fixture.
No Wuhan-network or event-library simulation was started.

## Long tasks intentionally not run

- `ScanOpportunityPool`
- `RunPeakBoundary`
- `RunPilot400`
- all `RunTrainRound*` stages
- `TrainV4`, `CalibrateV4`, and locked evaluation
- exact and surrogate closed-loop stages
- Challenge
- Formal Blind
- paper evidence generation

Use `docs/runbooks/PROJECT6_V4_FINAL_RUNBOOK.md` for the fail-fast future
execution sequence. Each downstream stage is prerequisite-gated and any
non-zero result blocks continuation.
