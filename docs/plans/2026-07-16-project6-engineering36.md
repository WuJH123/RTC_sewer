# Project6 Engineering36 Implementation Plan

**Goal:** Build a clean 36-facility hierarchical RTC engineering branch that preserves PFV event noninferiority while reducing TFV and peak risk under sparse sensing.

**Architecture:** The branch freezes experiment definitions first, then advances through gated stages. Core26 candidates are executable Tier 1 actions; Residual10 is a rejectable Tier 2 increment; No-control remains the final fallback. PFV is handled as an event-level uncertainty-bounded budget, not a per-window improvement requirement.

**Tech Stack:** Python, pandas, PyYAML, PySWMM/SWMM project artifacts, existing GAT checkpoints, PowerShell run orchestration, pytest.

---

### Task 1: Freeze Engineering Contract

**Files:**
- Create: `configs/wuhan_project6_engineering36.yaml`
- Create: `scripts/124_init_project6_engineering36.py`
- Create: `scripts/project6_runs/RUN_PROJECT6_ENGINEERING36.ps1`

**Steps:**
1. Generate `facilities_36_semantics.csv` from the audited INP actuator table and the 36-facility mask.
2. Assert Core26/Residual10 counts, residual ID membership, and binary pump semantics.
3. Freeze priority core, sentinel, event split, and sparse-sensor GAT model registry.
4. Write `contract_manifest.json` with hashes.

### Task 2: Enforce Event PFV Budget

**Files:**
- Create: `sewerrtc/control/event_pfv_budget.py`
- Test: `tests/test_project6_engineering36_contract.py`

**Steps:**
1. Implement `EventPfvBudget` with `max(200 m3, 0.02 * predicted_event_no_control_PFV)`.
2. Debit realized and in-flight conservative PFV costs cumulatively.
3. Reject candidates whose future PFV UCB exceeds remaining budget.

### Task 3: Rebuild Same-State Data Generator

**Files:**
- Create next: `scripts/125_plan_engineering36_same_state_cases.py`
- Create next: `scripts/126_generate_engineering36_same_state_cases.py`

**Requirements:**
- No-control vs Core26 and Core26 vs Core26+Residual10.
- Exact frozen 36-action hash.
- Event-disjoint fit/calibration/smoke/FormalBlind split.
- PFV budget boundary, TFV/peak contrast, H30/H120 reversal, double-peak release, downstream pump-risk strata.

### Task 4: Train and Gate

**Files:**
- Modify next: action-effect training entrypoint to consume Engineering36 dataset semantics.
- Create next: Engineering36 gate report.

**Requirements:**
- H30/H60/H90/H120 PFV, TFV, peak, terminal hydraulic outputs, uncertainty intervals, sentinel risk, reversal probability.
- Gate must fail closed when PFV/peak/sentinel/reliability conditions fail.

### Task 5: Smoke, Calibration, FormalBlind

**Files:**
- Extend next: `RUN_PROJECT6_ENGINEERING36.ps1`

**Requirements:**
- Smoke before calibration.
- Calibration acceptance thresholds from `configs/wuhan_project6_engineering36.yaml`.
- FormalBlind lock before blind execution; no rerun after unblinding.
