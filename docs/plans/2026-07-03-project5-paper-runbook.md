# Project5 Paper-Structured Runbook

**Goal:** Provide a step-by-step Project5 execution route aligned with the manuscript structure, while avoiding rerunning completed or stale-but-diagnostic stages.

**Architecture:** Project5 uses the Project4 SWMM/PySWMM, GAT, surrogate, residual counterfactual, and NativeShield pipeline, but the active paper target is now an 8-node PFV-core priority zone with 2 depth/surcharge sentinel nodes. PFV labels, residual labels, risk stratification, and formal gates must use the Project5 PFV-core configuration.

**Tech Stack:** Python, PySWMM/SWMM, pandas, PyTorch/PyTorch Geometric artifacts, residual action-value model, empirical action guard, Project5 scripts under `E:\RTC_sewer\Project5\scripts`.

---

## Common Shell Setup

Run these once in PowerShell:

```powershell
cd E:\RTC_sewer\Project5
$Py = "E:\RTC_sewer\Project\env\Scripts\python.exe"
$Cfg = "configs\wuhan.yaml"
```

Use run tags to avoid overwriting expensive outputs. Reuse `--skip-existing` or `--resume` where supported.

## Current State Snapshot

Current verified state as of 2026-07-03:

| Paper block | Current state | Evidence | Rerun? |
| --- | --- | --- | --- |
| Network audit | Done | `outputs/audit/node_table.csv` has 932 rows; `link_table.csv` has 1276 rows; `actuator_table.csv` has 109 rows | No, unless INP changes |
| Priority-zone design | Done and current | `outputs/design/priority_nodes.txt` has 8 PFV-core nodes; `priority_sentinel_nodes.txt` has 2 sentinel nodes | No, unless priority definition changes |
| Rainfall library | Done | `outputs/rainfall_library/rainfall_event_table.csv` has 100 events | No, unless rainfall config changes |
| GAT sparse reconstruction | Reused, not Project5-retrained | `outputs/models_paired_no_controls/gat_sr0p10.pt`; `gat_reconstruction_report_project4_reused.csv` | Skip for current analysis; rerun only for Project5-specific sensor claims |
| Graph/horizon surrogate | Reused/available | `graph_surrogate_best.pt`, `horizon_ridge_surrogate.npz`, `horizon_residual_quantile_uncertainty.npz` | Skip unless retraining from Project5 data |
| Influence-domain candidate audit | Not executed after 8-node refactor | No `outputs/network/` influence files | Run before relying on influence-domain candidate generation |
| Residual relabel | Done and current | `project5_residual_relabel_report.json`: 5044 rows, 20 events, 8 PFV-core nodes | No, unless residual bank or priority changes |
| Residual action-value model | Done and current | `residual_action_value.pt`; training report refreshed 2026-07-03 | No, unless new residual data is appended |
| Empirical guard | Done and current | guard table has 91 rows, 5 allowed rows | No, unless residual data/model changes |
| Project5 closed-loop under new 8-node priority | Not done | existing single-event runs predate the 8-node design update; treat as stale diagnostics | Run smoke/top20/full below |
| Recalculated formal metrics | Done as diagnostic only | `project5_priority_recalc_metadata.json` points to Project4 formal run, not new Project5 closed loop | Use for diagnosis, not final proof |
| Formal gate | Done and currently fails | `outputs/evaluation_project5_priority_zone/project5_formal_gate.json`: PFV mean reduction -0.074%, TFV worse 0.350, peak worse 0.275 | Failure is expected until new Project5 closed-loop/on-policy data improve controller |
| Project5 sensor sensitivity | Not done | no `outputs/sensor_sensitivity/` | Run later only after controller is stable |
| Manuscript tables/figures | Not done | no `outputs/manuscript_tables/`; no `outputs/figures_paired_no_controls/` | Run after formal gate is meaningful |

Important stale-output note: `outputs/closed_loop_paired_no_controls/formal/project5_single_T75_D210_*` was generated before the new 8-node priority design. Do not cite those single-event closed-loop results as current Project5 evidence.

## Stage 1: Methods - Network, Controls, Priority Zone, Sensors

Purpose: establish the Wuhan SWMM network, internal rules, actuator set, 8-node PFV-core priority zone, 2 sentinel nodes, and sparse sensors.

Current state: executed and current.

Skip if these files exist and match the active config:

```text
outputs/audit/node_table.csv
outputs/audit/link_table.csv
outputs/audit/actuator_table.csv
outputs/design/priority_nodes.txt
outputs/design/priority_sentinel_nodes.txt
outputs/design/sensor_nodes.csv
```

Run only if the INP, priority nodes, sentinel nodes, or sensor policy changes:

```powershell
& $Py scripts\00_audit_inp.py --config $Cfg
& $Py scripts\02_select_priority_and_sensors.py --config $Cfg
```

Expected current design:

```text
PFV core: MSLBZW001, HS1316314, YS2530050, HS2529198, MH0200773, HS1330349, HS2529139, HS2529052
Sentinels: MH0200770, HS1355904
```

## Stage 2: Experimental Design - Rainfall Event Library

Purpose: define design storms and temporal rainfall patterns used by trajectory generation and closed-loop evaluation.

Current state: executed. The active rainfall table has 100 events.

Skip if this file exists and rainfall config has not changed:

```text
outputs/rainfall_library/rainfall_event_table.csv
```

Run only if `configs/wuhan.yaml` rainfall settings change:

```powershell
& $Py scripts\01_generate_rainfall_library.py --config $Cfg --mode train
```

Use `--mode debug` only for small code checks. Use `--mode formal` only if you intentionally want the config's formal rainfall subset rather than the current 100-event training-style table.

## Stage 3: Methods - Sparse-Sensing GAT Reconstruction

Purpose: support the paper claim that sparse sensors can reconstruct full-network/priority-zone hydraulic states.

Current state: GAT model and report are reused from Project4. This is acceptable for code continuity, but not enough for a final Project5-specific sensor-sensitivity claim under the 8-node PFV-core definition.

Skip for controller debugging because these artifacts already exist:

```text
outputs/models_paired_no_controls/gat_sr0p10.pt
outputs/diagnostics_paired_no_controls/gat_reconstruction_report_project4_reused.csv
```

Run this block only if you need Project5-specific GAT training or new sensor-sensitivity results:

```powershell
& $Py scripts\03_generate_generic_trajectories.py --config $Cfg --mode full --resume --workers 8
& $Py scripts\04_build_tensor_cache.py --config $Cfg --time-stride 1 --horizon-steps 6
& $Py scripts\05_train_gat.py --config $Cfg --epochs 120 --device cuda
```

If CUDA is unavailable:

```powershell
& $Py scripts\05_train_gat.py --config $Cfg --epochs 120 --device cpu
```

Do not run this block just to update residual action-value or formal gate. It is expensive and not currently the bottleneck.

## Stage 4: Methods - Temporal Graph Surrogate, Horizon Scorer, Uncertainty

Purpose: support horizon candidate ranking and safety screening.

Current state: reused/available artifacts exist:

```text
outputs/models_paired_no_controls/graph_surrogate_best.pt
data/surrogate/horizon_mpc_dataset.parquet
outputs/models_paired_no_controls/horizon_ridge_surrogate.npz
outputs/models_paired_no_controls/horizon_residual_quantile_uncertainty.npz
outputs/models_paired_no_controls/action_applicability_model_card.json
```

Skip unless you regenerate trajectory data or change horizon features/objective.

If retraining is needed:

```powershell
& $Py scripts\06_train_surrogate.py --config $Cfg --epochs 120 --device cuda --batch-size 128 --num-workers 0
& $Py scripts\42_build_horizon_surrogate_dataset.py --config $Cfg --workers 8 --resume
& $Py scripts\43_train_horizon_surrogate.py --config $Cfg
& $Py scripts\44_validate_horizon_surrogate.py --config $Cfg
& $Py scripts\45_train_uncertainty_heads.py --config $Cfg
& $Py scripts\46_validate_uncertainty_gate.py --config $Cfg
& $Py scripts\47_train_action_applicability_model.py --config $Cfg
& $Py scripts\48_validate_action_applicability_model.py --config $Cfg
```

Run the influence-domain audit before depending on influence-domain candidate generation under the new 8-node PFV core:

```powershell
& $Py scripts\49_build_influence_domains.py --config $Cfg --khop 3
& $Py scripts\50_audit_influence_candidate_generation.py --config $Cfg --max-delta 0.08
```

## Stage 5: Methods - Residual Action-Value and Safety Shield

Purpose: relabel residual counterfactuals for the 8-node PFV core, train residual action-value, and build empirical guard.

Current state: executed and current, but still weak. Formal gate reports residual PFV direction accuracy `0.570`, safe precision `0.780`, and guard coverage `3` events.

Skip if these files are newer than the last priority/residual data change:

```text
outputs/closed_loop_paired_no_controls/internal_residual_counterfactuals/residual_counterfactual_results.csv
outputs/closed_loop_paired_no_controls/internal_residual_counterfactuals/project5_residual_relabel_report.json
outputs/models_paired_no_controls/residual_action_value.pt
outputs/diagnostics_paired_no_controls/residual_action_value_training_report.csv
outputs/diagnostics_paired_no_controls/action_template_outcomes/action_template_empirical_guard_table.csv
```

Rerun this block after adding any new residual counterfactual rows:

```powershell
& $Py scripts\62_relabel_residual_counterfactuals_project5_priority.py --config $Cfg --workers 8
& $Py scripts\12_train_residual_action_value.py --config $Cfg --device cpu --epochs 120
& $Py scripts\18_audit_action_template_outcomes.py `
  --config $Cfg `
  --mode formal `
  --run-tag project5_priority_relabel_guard `
  --min-samples 20 `
  --min-events 3 `
  --min-pfv-improve-safe-frac 0.30 `
  --max-pfv-worse-frac 0.45 `
  --min-safe-guarded-frac 0.45 `
  --max-peak-worse-frac 0.45

$GuardSrc = "outputs\diagnostics_paired_no_controls\formal\project5_priority_relabel_guard\action_template_outcomes\action_template_empirical_guard_table.csv"
$GuardDstDir = "outputs\diagnostics_paired_no_controls\action_template_outcomes"
New-Item -ItemType Directory -Force -Path $GuardDstDir | Out-Null
Copy-Item -LiteralPath $GuardSrc -Destination (Join-Path $GuardDstDir "action_template_empirical_guard_table.csv") -Force
```

## Stage 6: Results - Project5 Closed-Loop Smoke Under the New PFV Core

Purpose: verify the current controller actually runs with the new 8-node PFV-core design. This has not been done yet.

Start with the top-5 high-risk PFV events from the current diagnostic event table:

```text
T75_D75_chicago_center,T75_D75_chicago_late,T75_D150_chicago_late,T75_D105_chicago_center,T75_D75_block
```

Run the smoke closed loop:

```powershell
$Top5 = "T75_D75_chicago_center,T75_D75_chicago_late,T75_D150_chicago_late,T75_D105_chicago_center,T75_D75_block"

& $Py scripts\08_run_closed_loop.py `
  --config $Cfg `
  --mode formal `
  --run-tag project5_pfvcore_top5_smoke_v1 `
  --event-ids $Top5 `
  --workers 6 `
  --baseline-policies internal_rules,auto_rbc,no_control `
  --proposed-base native `
  --skip-existing

& $Py scripts\61_recalculate_project2_priority_zone_metrics.py `
  --config $Cfg `
  --run-dir outputs\closed_loop_paired_no_controls\formal\project5_pfvcore_top5_smoke_v1 `
  --out-dir outputs\evaluation_project5_pfvcore_top5_smoke_v1
```

Do not use formal gate on top-5 because gate requires at least 20 high-risk paired events. Inspect:

```text
outputs/evaluation_project5_pfvcore_top5_smoke_v1/project5_priority_policy_summary_main.csv
outputs/closed_loop_paired_no_controls/formal/project5_pfvcore_top5_smoke_v1/proposed/
```

## Stage 7: Methods/Results - Project5 On-Policy Residual Counterfactuals

Purpose: fix the current weak residual action-value model by generating Project5-specific on-policy residual samples.

Do this only after Stage 6 produces a valid source closed-loop run with controller histories.

Recommended top-20 event set:

```text
T75_D75_chicago_center,T75_D75_chicago_late,T75_D150_chicago_late,T75_D105_chicago_center,T75_D75_block,T75_D105_block,T75_D105_chicago_late,T75_D150_chicago_center,T75_D75_chicago_early,T75_D75_double_peak,T75_D105_double_peak,T75_D105_chicago_early,T75_D210_chicago_late,T75_D150_block,T75_D210_chicago_center,T75_D150_chicago_early,T75_D150_double_peak,T75_D210_block,T75_D210_chicago_early,T75_D210_double_peak
```

First create a top-20 source run:

```powershell
$Top20 = "T75_D75_chicago_center,T75_D75_chicago_late,T75_D150_chicago_late,T75_D105_chicago_center,T75_D75_block,T75_D105_block,T75_D105_chicago_late,T75_D150_chicago_center,T75_D75_chicago_early,T75_D75_double_peak,T75_D105_double_peak,T75_D105_chicago_early,T75_D210_chicago_late,T75_D150_block,T75_D210_chicago_center,T75_D150_chicago_early,T75_D150_double_peak,T75_D210_block,T75_D210_chicago_early,T75_D210_double_peak"

& $Py scripts\08_run_closed_loop.py `
  --config $Cfg `
  --mode formal `
  --run-tag project5_pfvcore_top20_source_v1 `
  --event-ids $Top20 `
  --workers 8 `
  --baseline-policies internal_rules,auto_rbc,no_control `
  --proposed-base native `
  --skip-existing
```

Dry-run the on-policy plan:

```powershell
& $Py scripts\17_generate_on_policy_residual_counterfactuals.py `
  --config $Cfg `
  --mode formal `
  --source-run-tag project5_pfvcore_top20_source_v1 `
  --max-events 20 `
  --samples-per-phase 4 `
  --max-cases 200 `
  --workers 8 `
  --max-delta 0.08 `
  --dry-run-plan
```

If the dry-run reports useful jobs, execute:

```powershell
& $Py scripts\17_generate_on_policy_residual_counterfactuals.py `
  --config $Cfg `
  --mode formal `
  --source-run-tag project5_pfvcore_top20_source_v1 `
  --max-events 20 `
  --samples-per-phase 4 `
  --max-cases 200 `
  --workers 8 `
  --max-delta 0.08 `
  --resume
```

Then rerun Stage 5 to retrain residual action-value and guard.

## Stage 8: Results - Project5 Formal Closed-Loop Evaluation

Purpose: generate the actual Project5 closed-loop evidence for the paper. This is not completed yet.

Do this only after:

1. Stage 6 top-5 smoke runs without runtime failures.
2. Stage 7 either improves residual/guard diagnostics or establishes that the current data remain insufficient.

Run main paper policy set first:

```powershell
& $Py scripts\08_run_closed_loop.py `
  --config $Cfg `
  --mode formal `
  --run-tag project5_pfvcore_full_formal_v1 `
  --workers 8 `
  --baseline-policies internal_rules,auto_rbc,no_control `
  --proposed-base native `
  --skip-existing
```

Recalculate Project5 PFV-core metrics:

```powershell
& $Py scripts\61_recalculate_project2_priority_zone_metrics.py `
  --config $Cfg `
  --run-dir outputs\closed_loop_paired_no_controls\formal\project5_pfvcore_full_formal_v1 `
  --out-dir outputs\evaluation_project5_pfvcore_full_formal_v1
```

Run formal gate:

```powershell
& $Py scripts\63_project5_formal_gate.py `
  --config $Cfg `
  --paired-metrics outputs\evaluation_project5_pfvcore_full_formal_v1\project5_priority_paired_metrics_main.csv `
  --out-dir outputs\evaluation_project5_pfvcore_full_formal_v1
```

Only if the main policy set is acceptable, run supplementary diagnostics:

```powershell
& $Py scripts\08_run_closed_loop.py `
  --config $Cfg `
  --mode formal `
  --run-tag project5_pfvcore_full_formal_supplement_v1 `
  --workers 8 `
  --baseline-policies internal_rules,auto_rbc,no_control,all_open,random_safe,efd_static,efd_storage_priority `
  --proposed-base native `
  --skip-existing
```

## Stage 9: Results - Risk-Stratified Water Research Tables

Purpose: build the risk-stratified event table and Water Research-style summaries for the actual Project5 run.

Do not run this for final claims until Stage 8 has a current Project5 run tag.

For the future full formal run:

```powershell
& $Py scripts\40_build_risk_stratified_event_table.py `
  --config $Cfg `
  --mode formal `
  --run-tag project5_pfvcore_full_formal_v1 `
  --baseline-policy internal_rules `
  --out-dir outputs\evaluation_project5_pfvcore_full_formal_v1

& $Py scripts\41_evaluate_risk_stratified_results.py `
  --config $Cfg `
  --mode formal `
  --run-tag project5_pfvcore_full_formal_v1 `
  --baseline-policy internal_rules `
  --event-table outputs\evaluation_project5_pfvcore_full_formal_v1\risk_stratified_event_table.csv `
  --out-dir outputs\evaluation_project5_pfvcore_full_formal_v1
```

## Stage 10: Results/Discussion - Sensor Sensitivity

Purpose: support the sparse-sensing claim under Project5's 8-node PFV core.

Current state: not executed in Project5. There is no `outputs/sensor_sensitivity/`.

Do not start this until controller behavior is stable enough to justify a sensor-sensitivity experiment.

Recommended future structure:

```text
outputs/sensor_sensitivity/sr0p05/
outputs/sensor_sensitivity/sr0p10/
outputs/sensor_sensitivity/sr0p15/
outputs/sensor_sensitivity/sr0p20/
outputs/sensor_sensitivity/sr0p30/
```

Because Project5 does not yet have a single wrapper script for this refactored 8-node setup, create or adapt a small runner only after Stage 8 is stable. Do not manually rerun all core training for each ratio until the controller passes a top-20 smoke.

## Stage 11: Manuscript Tables and Figures

Purpose: create final paper tables and figures.

Current state: not executed in Project5. There is no `outputs/manuscript_tables/` and no `outputs/figures_paired_no_controls/`.

Only run after Stage 8/9 produces acceptable results.

Existing scripts that can be used or adapted:

```powershell
& $Py scripts\10_significance_analysis.py --config $Cfg --mode formal --run-tag project5_pfvcore_full_formal_v1
& $Py scripts\14_wr_diagnostics_report.py --config $Cfg --mode formal --run-tag project5_pfvcore_full_formal_v1
& $Py scripts\60_select_representative_native_shield_scenarios.py --config $Cfg --mode formal --run-tag project5_pfvcore_full_formal_v1
```

`scripts\11_make_figures.py` currently expects older diagnostics files and should be audited before use for the Project5 PFV-core paper figures.

## Recommended Next Execution Order

Do not rerun from the beginning. The next useful sequence is:

1. Run Stage 6 top-5 smoke under the new 8-node priority.
2. Inspect `outputs\evaluation_project5_pfvcore_top5_smoke_v1\project5_priority_policy_summary_main.csv`.
3. If runtime is clean, run Stage 7 top-20 source run and on-policy residual dry-run.
4. If dry-run gives enough jobs, execute on-policy residual generation.
5. Rerun Stage 5 residual training and guard.
6. Rerun Stage 6 or Stage 8 depending on whether residual/gate diagnostics improve.

Current formal gate fails because the controller and residual model are weak under the 8-node PFV core; it does not fail because the new priority-zone files are missing.
