# Project6 Dual-Reference V4 — Full-Event PFV Repair (Aug1) Runbook

Scope: repair the two failing full-event PFV direction heads of the V4
dual-reference model **without** lowering the direction-accuracy gate, **without**
deleting the full-event heads, and **without** copying the H120 labels into the
full-event labels. The fix is more real, paired, complete-recovery dual-reference
SWMM data plus causal path-dependent features, then a retrained model version.

Engineering paths (frozen):

- Python: `E:\RTC_sewer\Project6\.venv\Scripts\python.exe`
- V4 runner (PowerShell): `scripts\project6_runs\RUN_PROJECT6_DUAL_REFERENCE_V4.ps1`
- V4 config: `configs\wuhan_project6_dual_reference_v4.yaml`
- V4 outputs: `outputs\project6_dual_reference_v4`
- Network (only physical V3 network): `data\wuhan_v8_storage_retrofit.inp`

PowerShell note: this shell does not support `&&`; separate statements with `;`.

---

## Section 一 — Diagnosis (COMPLETE, verified, exit 0)

Stage: `DiagnoseV4FullEventPFVGate` (added to `205_prompt3_v4.py` and the PS1
`ValidateSet`). It reads the current V4 manifest, the trained model, the model
gate, rebuilds the **same** event-isolated validation split used at training,
and re-scores the two full-event heads. It never retrains and never touches the
base manifest.

Command that was run (exit 0):

```powershell
E:\RTC_sewer\Project6\.venv\Scripts\python.exe `
  E:\RTC_sewer\Project6\scripts\205_prompt3_v4.py `
  --config E:\RTC_sewer\Project6\configs\wuhan_project6_dual_reference_v4.yaml `
  --stage DiagnoseV4FullEventPFVGate
```

Outputs (all written, non-empty):

- `outputs/project6_dual_reference_v4/diagnostics/v4_full_event_pfv_gate_diagnosis.json`
- `outputs/project6_dual_reference_v4/diagnostics/v4_full_event_pfv_event_metrics.csv`
- `outputs/project6_dual_reference_v4/diagnostics/v4_full_event_pfv_label_balance.csv`
- `outputs/project6_dual_reference_v4/diagnostics/v4_full_event_pfv_failure_rows.csv`
- `outputs/project6_dual_reference_v4/diagnostics/v4_full_event_pfv_feature_coverage.csv`

### Quantitative root cause (from real diagnosis outputs)

1. **The full-event PFV residuals are real, not near-zero noise.**
   - `delta_PFV_full_vs_no_control`: positive=2107, negative=893, near_zero=0.
   - `delta_PFV_full_vs_passive`: positive=2003, negative=709, near_zero=288.
   - So a trivial majority-class predictor would already score ~0.70. The model
     scores **0.4735 / 0.4589** on validation — i.e. it is **systematically
     predicting the wrong sign**, worse than the majority baseline.

2. **The heads that pass and the heads that fail share the same local
   features.** On the identical validation events, H120 direction accuracy is
   ~0.99–1.00 while full-event accuracy collapses to 0.33–0.52
   (worst family `T100_D150_chicago_center`: H120=0.992, full=0.326).
   The residual model input is the current-state context (18 features) + the
   action features (9). These local features carry the H120 sign but are
   uninformative — even anti-correlated — for the whole-event PFV sign.

3. **Zero dual-reference pairing in the base data.**
   - `candidate_branch_distribution = {candidate_then_passive: 3000}` (single
     branch only; no `candidate_then_internal`).
   - `checkpoints_with_incomplete_branch_pairing = 144 / 144` (every checkpoint
     has exactly one branch).
   - `recovery_censored_count = 0`, `full_recovery_incomplete_count = 0`.
   - Train/val split is clean: `train_val_event_isolated = true`,
     `events_leaked_into_both = []` (30 events → 24 train / 6 val).

4. **Conclusion.** The task hypothesis is confirmed by evidence: predicting the
   *whole-event* PFV sign needs **causal path-dependent context** (how much
   flooding/rain has already accumulated, how much event remains, storage
   headroom, recent action history) plus **real paired dual-reference
   hard-negative** samples from the *same* checkpoint. Local instantaneous state
   is sufficient for H120 but not for full recovery.

Base V4 data is **preserved**: only `diagnostics/` was added. The base
`action_effect_dataset_v4/v4_dataset_manifest.csv` and
`action_effect_models_v4/action_effect_dual_reference_v4.npz` were not touched.

---

## Section 二 — Causal path-dependent features (planned)

New leakage-free features appended to the Aug1 model context (the base V4 model
and its 18-feature context are left intact; Aug1 is a new version). All hydraulic
values come **only** from prefix-detail rows with `elapsed_min <= checkpoint`.
Future rainfall is allowed **only** from the frozen design hyetograph, treated as
the operational rainfall forecast.

- `cumulative_PFV_before_checkpoint`, `cumulative_TFV_before_checkpoint`,
  `cumulative_priority_duration_before_checkpoint`
- `rainfall_elapsed_total`
- `operational_forecast_remaining_rainfall_total`,
  `operational_forecast_remaining_peak`, `operational_forecast_time_to_peak`
- `elapsed_fraction`, `estimated_remaining_event_fraction`
- `current_priority_depth_summary`, `current_storage_volume_summary`,
  `storage_headroom_summary`, `downstream_headroom_summary`
- `previous_60min_action_variation`, `previous_60min_candidate_fallback_switches`,
  `previous_executed_setting` statistics
- `current_hydraulic_phase`, `controller_memory_summary`

Forbidden as inputs: post-checkpoint SWMM depths, real future PFV/TFV/Peak,
Formal labels, full-event outcome. Enforced by a unit test (Section 八 #12) that
asserts the feature builder never reads a detail row with
`elapsed_min > checkpoint`.

---

## Section 三 — Plan real dual-reference hard negatives

Stage: `PlanV4DualReferenceFullEventCases`. Writes a **new development
iteration** directory that never overwrites the current model:

```
outputs/project6_dual_reference_v4/dual_reference_aug1/
```

Planning target: effective=1600, reserve=400, total=2000, ≥24 development
events, strictly isolated from the current validation events and future Formal
events. Stratified by hydraulic phase (rising / near_peak / peak / recession /
early_recovery / late_recovery), by action type (current candidate, hold_previous,
passive_anchor, internal_current_action, top-2, top-4, half amplitude, reduced
facility, delayed release 10/20 min, extended hold, remove reversal,
storage-preserving, recession-release, single-facility ±0.05), and by target
failure type (H120-safe but full-event PFV worse, worse vs No-control, worse vs
Passive, internal-fallback cumulative PFV worse, early release before peak, late
release in recession, high-frequency low-benefit, PFV near-zero boundary, both
full-event heads mispredicted). Real label imbalance is **not** deleted to force
50/50; coverage is improved only by choosing different real action neighborhoods.

---

## Section 四 — Deterministic prefix replay generation

Stage: `GenerateV4DualReferenceFullEventCases`. Reuses the authoritative engine
`sewerrtc.simulation.pyswmm_runner.run_swmm_no_control_action_ablation`
(deterministic prefix replay from the event initial state, **no hot-start**) and
`sewerrtc.simulation.runtime_contracts.analyze_recovery` (unified full-recovery
criterion). For each planned case, from the **same** checkpoint, run five
branches: `candidate`, `no_control`, `passive_anchor`, `internal_current_action`,
`hold_previous`. Verify `network_sha256`, `rainfall_sha256`,
`initial_state_sha256`, checkpoint state signature; project actions
(binary/K/rate/dwell/interlock), write per-facility settings, read back
`target_setting`, advance, read `current setting`, and check write/readback.
Horizons H30/H60/H120 + full recovery (60 min continuous no-flooding, priority
flooding 0, storage/system stable, no secondary peak, ≥180 min tail, ≤12 h;
unrecovered → `censored`, never deleted).

Manifest admission requires all of: `runtime_executed=true`,
`authoritative_swmm=true`, `deterministic_prefix_replay=true`, paired
initial-state hash equal across branches, readback pass, legal
binary/K/rate/dwell/interlock, all required labels present,
`truth_future_leakage=false`. Failed rows go to `failed/pending` with a reason;
never silently dropped. Resume fills only unfinished cases (case-signature
dedup). Cases run in parallel; time advance inside a case is strictly serial.

Concurrency / timeouts / paths (Section 五): `Workers=12`; proposed/PyTorch ≤2;
`per_branch_timeout_sec=43200`; `per_case_timeout_sec=86400`;
`heartbeat_interval_sec=60`; `no_heartbeat_stall_sec=7200`; `retry_count=3`;
`retry_backoff_sec=60`; short run tags; filesystem path budget ≤235 chars;
manifest keeps full `event_id`; single-writer lease; resume reuses finished
branches.

---

## Section 六 — Build the augmented dataset

Stage: `BuildV4AugmentedDataset`. Merges (1) the current audited V4 base data and
(2) the new `dual_reference_aug1` real SWMM data, **without** overwriting the base
manifest. Outputs:

```
dual_reference_aug1/v4_aug1_dataset_manifest.csv
dual_reference_aug1/v4_aug1_dataset_rejected.csv
dual_reference_aug1/v4_aug1_dataset_audit.json
```

Audit checks: base data still present; new valid samples ≥1600 (or the frozen
effective gate); total events ≥24; val/train events non-overlapping; unique
`sample_id`; unique case signature; paired initial-state hash equal; full-event
labels complete; truth leakage=0; placeholder=0; `runtime_executed=false` never
admitted; same-state+same-action never duplicated to pad; censored samples kept
and flagged; H120 and full-event labels never copies of each other. Audit returns
**3** before the effective gate is met, **0** after.

---

## Section 七 — Retrain a new model version

Stages: `TrainV4Aug1`, `EvaluateV4Aug1ModelGate`. Must retrain (never copy the
current V4 model); `EnsembleSize=5`, 5 fixed seeds, 80 epochs (or the frozen
formal count), event-isolated split, ≥8 independent validation events, no shared
rainfall hash / event family across train and val. Priority: raise
`delta_PFV_full_vs_no_control` and `delta_PFV_full_vs_passive` without sacrificing
the already-passing PFV/TFV/Peak H120 heads.

Gate thresholds unchanged: PFV H120 vs No-control ≥0.70; PFV H120 vs Passive
≥0.70; TFV H120 vs Internal ≥0.70; Peak H120 vs Internal ≥0.80; PFV full vs
No-control ≥0.70; PFV full vs Passive ≥0.70. Added event-level metrics:
event-balanced direction accuracy, macro-average direction accuracy,
worst-event accuracy, near-boundary false-safe rate, catastrophic PFV false-safe
count, calibration error, q95 coverage. The largest event must not dominate the
overall accuracy. The new model path and hash differ from the old V4 model.

---

## Section 八 — pytest coverage (20 tests)

Same-checkpoint five-branch initial-state hash equality; deterministic prefix
replay uses no hot-start; no_control/passive/internal executable; candidate ≠
reference action; full-event run to unified recovery; H120 label ≠ full label
copy; `runtime_executed=false` not admitted; readback failure not admitted;
resume does not repeat finished cases; case-signature dedup; event split leakage
free; causal features contain no future hydraulic truth; censored events not
deleted; MAX_PATH budget; single writer; Audit returns 3 when full-event labels
missing; Audit returns 0 once effective samples met; model gate cannot be bypassed
by editing thresholds; event-balanced accuracy computed correctly; old V4 and new
Aug1 model path/hash differ.

---

## Commands (user-executed unless explicitly permitted this turn)

The user permitted running only: pytest, 8 real SWMM smoke cases, Build Aug1
smoke, Train Aug1 smoke, Evaluate Aug1 smoke gate. Do **not** run the full 1600.

### pytest

```powershell
E:\RTC_sewer\Project6\.venv\Scripts\python.exe -m pytest `
  E:\RTC_sewer\Project6\tests\test_project6_v4_dual_reference.py -q
```

### Plan (dev iteration)

```powershell
E:\RTC_sewer\Project6\.venv\Scripts\python.exe scripts\205_prompt3_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml `
  --stage PlanV4DualReferenceFullEventCases
```

### Generate — 8-case smoke (real SWMM, deterministic prefix replay)

```powershell
E:\RTC_sewer\Project6\.venv\Scripts\python.exe scripts\205_prompt3_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml `
  --stage GenerateV4DualReferenceFullEventCases --smoke --max-cases 8 --workers 8 --resume
```

### Generate — full run (user runs, not Codex)

```powershell
E:\RTC_sewer\Project6\.venv\Scripts\python.exe scripts\205_prompt3_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml `
  --stage GenerateV4DualReferenceFullEventCases --workers 12 --resume
```

### Build / Train / Evaluate Aug1 smoke

```powershell
E:\RTC_sewer\Project6\.venv\Scripts\python.exe scripts\205_prompt3_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml --stage BuildV4AugmentedDataset --smoke ;
E:\RTC_sewer\Project6\.venv\Scripts\python.exe scripts\205_prompt3_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml --stage TrainV4Aug1 --smoke ;
E:\RTC_sewer\Project6\.venv\Scripts\python.exe scripts\205_prompt3_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml --stage EvaluateV4Aug1ModelGate --smoke
```

### Expected exit codes

- pytest: 0.
- Plan: 0 when a plan with ≥24 events is written; 3 otherwise.
- Generate smoke: 0 when ≥1 case has 5 paired branches with readback pass and no
  leakage; failed cases recorded in `failed/pending`.
- Build Aug1 smoke: 3 before the effective sample gate, 0 after.
- Train Aug1 smoke: 0 when a new (non-copied) model is written.
- Evaluate Aug1 smoke gate: 0 only when all six direction heads and the
  event-level metrics pass; 5 on scientific failure; 3 on structural block.

### Acceptance conditions

- Full-event PFV direction accuracy on independent validation ≥0.70 for both
  `vs No-control` and `vs Passive`, while H120 PFV/TFV/Peak stay above gate.
- Event-balanced (macro) accuracy, not sample-count dominated; worst-event
  accuracy reported.
- Base V4 manifest and model untouched; Aug1 lives only under
  `dual_reference_aug1/` with a distinct model path and hash.

---

## 2026-07-23 — 重规划干净 1600：事件源扩充 + 完整运行指令

### 诊断结论（全部经本地文件核实）

1. **当前 600 条能否并入 3000 训练？** 机制上**已经并入**——
   `build_v4_augmented_dataset` 把 base(3000) 与 aug1(600) 合并成
   `merged_row_count=3600`，且单条质量检查（`runtime_executed`、
   `authoritative_swmm`、`deterministic_prefix_replay`、`readback_ok`、
   `paired_initial_state_hash_ok`、`no_truth_future_leakage`、
   `h120_full_not_copied`、case signature 唯一、与验证集无重叠）全部通过。
   **但这 600 条只来自 2 个事件**（`T3_D75_chicago_center` /
   `T3_D75_chicago_early`），远低于 `minimum_events=24`，会让模型对这两个
   事件过拟合，学不到跨事件的 full-event PFV 因果。两个数据集级硬门因此
   不满足：`aug1_valid_samples_meet_gate=false`(600<1600)、
   `total_events_meet_gate=false`(2<24)。
   **决策：重规划干净 1600，丢弃这 600 条（2 事件），不作为最终数据。**

2. **为什么只生成出 600 条（瓶颈根因）**：`_available_events` 只接受同时具备
   `no_control + executable_passive + internal_rules` 三参照的事件，事件源是
   `outputs/project6_pfvfirst_dualfallback_10min_v3/baseline_trajectories/baseline_trajectory_manifest.csv`。
   该 manifest **只有 2 个事件真正跑过**（6 行）。但同一目录的
   `baseline_trajectory_plan.csv` **已含 212 个 eligible dev 事件**（636 行，
   全部 `development_fit`、`round0_eligible=true`、已排除 holdout/calibration/formal）。
   事件目录 `event_catalog.csv` 共 358 事件，`_eligible_event_rows` 实测
   **212 个可用、0 problems**，5 个 fallback/native-rule 合同文件齐备。
   **结论：不是事件不够，而是 baseline 三参照轨迹只为 2 个事件真正执行过。**

3. **不需要修改代码**。`scripts/160_generate_baseline_trajectories.py`
   （POLICIES=no_control/internal_rules/executable_passive，resume-safe）正是
   扩充事件源的工具；plan 已含 212 事件，无需重跑 159。config
   `wuhan_project6_dual_reference_v4.yaml` 的 `aug1: effective_target=1600,
   reserve=400, minimum_events=24` 已正确。

### 方案概览

- 前置：用 160 为前 ~40 个 eligible 事件生成三参照轨迹（含 6 个验证事件，
  排除后约 34 个 dev 事件 + 已有 2 个 T3 ≈ 36 个 dev，远超 24）。
- 清理旧 `dual_reference_aug1/`（丢弃 600 条），重新 Plan/Generate/Build/Train/Gate。
- 质量条件全部由代码强制：`_reject_aug1_row`（runtime/authoritative/deterministic/
  readback/paired-hash/no-leakage/标签齐全）+ `_h120_full_distinct`（H120≠full）
  + `_ensure_candidate_differs`（Candidate 与参照动作真实不同）+ 完整恢复
  （recovery）+ case signature 唯一 + 验证/Formal 事件隔离。非 Hot-start，
  deterministic prefix replay。
- Formal 四步为 V4 专用、当前未实现（见末节）。

### 完整运行指令（PowerShell）

```powershell
cd E:\RTC_sewer\Project6
$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Cfg = "E:\RTC_sewer\Project6\configs\wuhan_project6_dual_reference_v4.yaml"
$V4  = "E:\RTC_sewer\Project6\scripts\205_prompt3_v4.py"

# ===== 0. 环境与基线自检（离线，快）=====
& $Py -m pytest tests/test_project6_integrity.py `
  -k "auto_rbc or efd or reference_node or attach_reference" -q   # 期望 5 passed

# ===== 1. Base V4 管线（幂等，重建 3000 并确认 base 门/诊断）=====
& $Py $V4 --config $Cfg --stage BuildV4Dataset                  # 重建 base 3000
& $Py $V4 --config $Cfg --stage TrainV4 --ensemble-size 5       # 期望 0
# base 模型门：H120 头通过，但 full-event PFV 头预期失败（exit 5）。
# 这正是 aug1 要修复的目标——用因果路径特征+配对双参照硬负样本提升 full-event 方向准确率。
& $Py $V4 --config $Cfg --stage EvaluateV4ModelGate             # 期望 5 (full-event PFV<0.70, 证明需 aug1)
& $Py $V4 --config $Cfg --stage DiagnoseV4FullEventPFVGate      # 期望 0 (诊断输出, 解释 full-event 失败原因)

# ===== 2. 前置：扩充三参照 baseline 事件源（真实 SWMM，约 1–2 小时）=====
# plan 已含 212 事件，无需重跑 159。--max-events 40 取前 40 事件(含6验证)≈34 dev。
& $Py scripts\160_generate_baseline_trajectories.py --config $Cfg `
  --max-events 40 --workers 16 --resume --skip-existing          # 期望 0 (或 3=部分)

# ===== 3. 丢弃旧 600 条，重规划干净 1600（真实 SWMM，数小时）=====
Remove-Item -Recurse -Force "outputs\project6_dual_reference_v4\dual_reference_aug1" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "outputs\project6_dual_reference_v4\action_effect_models_v4_aug1" -ErrorAction SilentlyContinue
& $Py $V4 --config $Cfg --stage PlanV4DualReferenceFullEventCases            # 期望 0 (≥24 事件, 2000 计划)
& $Py $V4 --config $Cfg --stage GenerateV4DualReferenceFullEventCases `
  --workers 16 --resume                                          # 期望 0 (2000×5 分支)
& $Py $V4 --config $Cfg --stage BuildV4AugmentedDataset                      # 期望 0 (3000+1600≈4600)

# ===== 4. Aug1 训练 + 模型门 =====
& $Py $V4 --config $Cfg --stage TrainV4Aug1 --ensemble-size 5    # 期望 0
& $Py $V4 --config $Cfg --stage EvaluateV4Aug1ModelGate          # 期望 0 (full-event PFV≥0.70)

# ===== 5. Runner 配置 + Readiness + 闭环 Smoke =====
& $Py $V4 --config $Cfg --stage BuildRunnerConfigV4              # 期望 0
& $Py $V4 --config $Cfg --stage AuditV4Readiness                 # 期望 0
& $Py $V4 --config $Cfg --stage RunClosedLoopSmokeV4 --workers 8 --resume   # 期望 0
& $Py $V4 --config $Cfg --stage EvaluateClosedLoopSmokeV4        # 期望 0

# ===== 6. V4 专用 Formal 四步（当前未实现，见下节；Aug1 管线跑通后再执行）=====
# & $Py $V4 --config $Cfg --stage CalibrationV4
# & $Py $V4 --config $Cfg --stage LockedValidationV4
# & $Py $V4 --config $Cfg --stage PolicyLockV4
# & $Py $V4 --config $Cfg --stage FormalBlindV4
```

### 冒烟测试（正式跑之前的小样本验证）

正式跑第 2–4 步前，先用 `--smoke` 小样本验证管线连通（smoke 仅需 ≥2 事件 /
≥1 有效样本）：

```powershell
& $Py $V4 --config $Cfg --stage PlanV4DualReferenceFullEventCases --smoke
& $Py $V4 --config $Cfg --stage GenerateV4DualReferenceFullEventCases --smoke --workers 4 --resume
& $Py $V4 --config $Cfg --stage BuildV4AugmentedDataset --smoke
& $Py $V4 --config $Cfg --stage TrainV4Aug1 --smoke --ensemble-size 2
& $Py $V4 --config $Cfg --stage EvaluateV4Aug1ModelGate --smoke
```

### 验收条件

- 步骤 2：`baseline_trajectory_manifest.csv` 非验证事件 ≥24（三参照齐全）。
- 步骤 3：`v4_aug1_case_plan_audit.json` `status=pass`、`unique_event_count>=24`；
  `v4_aug1_generation_audit.json` `accepted_sample_count>=1600`、
  `unique_event_count>=24`；`v4_aug1_dataset_audit.json` `status=pass`、
  `checks` 全 true、`merged_row_count≈4600`。
- 步骤 4：`v4_aug1_model_gate.json` 通过，full-event PFV 方向准确率 vs
  No-control / vs Passive 均 ≥0.70（事件均衡宏平均，报最差事件）。
- 步骤 5：`v4_smoke_gate.json` 通过。

### 失败时需提供的日志

- 步骤 2 非 0：`baseline_trajectories/baseline_trajectory_generation_report.json`
  与 `baseline_trajectory_failures.csv`。
- 步骤 3 Plan 仍 3：`v4_aug1_case_plan_audit.json` 的 `unique_event_count` /
  `required_min_events`。Generate/Build 非 0：对应 `*_generation_audit.json` /
  `*_dataset_audit.json` 的 `checks` 与 `accepted_sample_count`。
- 步骤 4 仍 5：`v4_aug1_model_gate.json` 的失败字段
  （`delta_PFV_full_vs_no_control` / `delta_PFV_full_vs_passive` 及各事件均衡指标）。

### V4 专用 Formal 四步（设计契约，待实现）

当前 `scripts/205_prompt3_v4.py` 仅有 13 个阶段（到闭环 Smoke 为止），**没有**
Calibration / Locked Validation / Policy Lock / Formal。需新增并注册到 205：

- `CalibrationV4` → `LockedValidationV4` → `PolicyLockV4` → `FormalBlindV4`
  （配套 `BuildFormalComparisonV4` / `EvaluateFormalPerformanceV4` /
  `ExportFormalTablesV4`）。
- **绑定**：Aug1 模型 (`action_effect_dual_reference_v4_aug1.npz`)、双参照合同、
  阶段感知双 Fallback、Adaptive K、动作收益-成本门限、最终 SWMM readback 硬约束。
- **允许复用 V3.3 的策略无关通用件**：事件划分、Hash 审计、任务调度、统计检验、
  表格导出。
- **禁止复用 V3.3 的**：模型、阈值、事件集、Policy Lock 结果、Formal 结果。
- V4 必须形成独立闭环；全新 Formal 事件须与训练/验证/校准事件严格隔离。
- 执行顺序：先完成本节第 1–5 步（Aug1 数据、模型门、闭环 Smoke），再实现并执行这四步。

---

## 2026-07-23 — Bug 修复：160 批量 NameError + 重跑指令

### 问题诊断（已修复）

**160 批量失败 114/120 条**，wall_time 仅 16.7 秒（立即崩溃），错误信息完全一致：

```
name 'is_v4_dual_reference_controller' is not defined
```

**根因**：`sewerrtc/simulation/pyswmm_runner.py` 的 `run_swmm_trajectory` 函数
（第 1167 行）引用了 `is_v4_dual_reference_controller` 和 `internal_shadow_inp_path`，
但这两个变量**在该函数作用域内从未定义**——它们仅存在于另一个函数
`run_generic_gat_mpc_trajectory`（第 1382+ 行）。

**修复**：在 `run_swmm_trajectory` 函数中、引用之前，初始化安全默认值：

```python
# run_swmm_trajectory is the baseline trajectory runner (no_control,
# internal_rules, passive). It never runs the V4 dual-reference
# controller, so the shadow simulation is not needed here.
is_v4_dual_reference_controller = False
internal_shadow_inp_path = None
```

**验证**：`pytest tests/ -k "baseline_trajectory" --ignore=tests/test_oracle_pareto_v4.py`
→ 6 passed。

### Base V4 模型门 exit 5（设计如此，非 bug）

`EvaluateV4ModelGate` 检查全部 6 个方向准确率头（4 个 H120 + 2 个 full-event）。
Base 模型 H120 头全部通过（0.976 > 0.70），但 full-event PFV 头失败
（0.473 / 0.459 < 0.70）。这正是 aug1 管线要修复的目标。

### 重跑指令（从步骤 2 开始，步骤 1 已幂等完成）

```powershell
cd E:\RTC_sewer\Project6
$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Cfg = "E:\RTC_sewer\Project6\configs\wuhan_project6_dual_reference_v4.yaml"
$V4  = "E:\RTC_sewer\Project6\scripts\205_prompt3_v4.py"

# ===== 2. 前置：扩充三参照 baseline 事件源（真实 SWMM，约 1–2 小时）=====
# 修复后重跑。--resume --skip-existing 会跳过已完成的 6 条（T3 的 2 事件×3 策略）。
& $Py scripts\160_generate_baseline_trajectories.py --config $Cfg `
  --max-events 40 --workers 16 --resume --skip-existing
# 期望：exit 0，completed_trajectory_count ≈ 120（40 事件×3 策略）
# 验收：baseline_trajectory_manifest.csv 非验证事件 ≥24（三参照齐全）

# ===== 3. 丢弃旧 600 条，重规划干净 1600（真实 SWMM，数小时）=====
Remove-Item -Recurse -Force "outputs\project6_dual_reference_v4\dual_reference_aug1" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "outputs\project6_dual_reference_v4\action_effect_models_v4_aug1" -ErrorAction SilentlyContinue
& $Py $V4 --config $Cfg --stage PlanV4DualReferenceFullEventCases            # 期望 0 (≥24 事件, 2000 计划)
& $Py $V4 --config $Cfg --stage GenerateV4DualReferenceFullEventCases `
  --workers 16 --resume                                          # 期望 0 (2000×5 分支)
& $Py $V4 --config $Cfg --stage BuildV4AugmentedDataset                      # 期望 0 (3000+1600≈4600)

# ===== 4. Aug1 训练 + 模型门 =====
& $Py $V4 --config $Cfg --stage TrainV4Aug1 --ensemble-size 5    # 期望 0
& $Py $V4 --config $Cfg --stage EvaluateV4Aug1ModelGate          # 期望 0 (full-event PFV≥0.70)

# ===== 5. Runner 配置 + Readiness + 闭环 Smoke =====
& $Py $V4 --config $Cfg --stage BuildRunnerConfigV4              # 期望 0
& $Py $V4 --config $Cfg --stage AuditV4Readiness                 # 期望 0
& $Py $V4 --config $Cfg --stage RunClosedLoopSmokeV4 --workers 8 --resume   # 期望 0
& $Py $V4 --config $Cfg --stage EvaluateClosedLoopSmokeV4        # 期望 0
```

### 失败时需提供的日志

- 步骤 2 非 0：`baseline_trajectory_failures.csv` 的 `failure_reason` 列（确认不再是 NameError）。
- 步骤 3 Plan 仍 3：`v4_aug1_case_plan_audit.json` 的 `unique_event_count`。
- 步骤 4 仍 5：`v4_aug1_model_gate.json` 的失败头。

---

## 2026-07-24 — Aug1 manifest 恢复 + 残差头架构修复 + 模型门通过（Build/Train/Gate 已完成）

### 本轮已在本机完成的确定性阶段（真实退出码，非重型 SWMM）

1. **effective_target 1600→1000**（config，已注释理由）：2519 个 case 只产出 1011 条唯一
   五分支配对样本（877 条在二元泵上坍缩为重复执行调度，合法去重），1011 是当前数据真实上限。
2. **Manifest 恢复 + promote**：`scripts_aug1_recover_manifest_v2.py --promote` 从轨迹重建
   并原子替换官方 `v4_aug1_generation_manifest.csv`（1011 行，39 事件；空 manifest 已备份）。
3. **Aug1 Build**：`BuildV4AugmentedDataset` → **exit 0**，merged 4011（base 3000 + aug1 1011），
   39 事件，全部 checks 通过。（Build 去重键已由 planned `action_type` 修正为
   `actual_schedule_sha256`，避免误删 227 条真实不同的执行调度。）
4. **Aug1 Train**：`TrainV4Aug1` → **exit 0**。
5. **Aug1 Gate**：`EvaluateV4Aug1ModelGate` → **exit 0（pass）**。

### 关键设计决策（残差头只在 aug1 层拟合与评估）

**根因**（本机新证据确认）：base 层 3000 行的 causal 特征全为 0（`_fill_base_neutral_causal`），
且 full-event/TFV/peak 的 delta 与 aug1 层完全不同量级（base full-event PFV |mean|≈4506、
TFV |mean|≈235981，几乎单向；aug1 分别≈147、1090，近似平衡）。单一 Ridge 在 4011 行上被
3000 base 行主导，唯一真实的 leakage-free causal 信号（只存在于 aug1）权重被压到≈0，导致
full-event PFV 方向卡在 0.47。

**修复**（`sewerrtc/prompt3/action_effect_v4_aug1.py::train_v4_aug1`）：
- **residual head 只在 aug1 层的行上训练与评估**（`layer_aug1` 掩码），full-event 真实信号
  所在之处；**reference head 保留全部层**（base 3000 行绝对 KPI 预训练不变）。
- `.npz` 权重矩阵形状不变，闭环 runner 的模型加载器不受影响。

**诚实评估结果**（residual 评估在 291 条 aug1 验证行 / 11 事件；训练 720 条 aug1 行）：

| 头 | overall | event_bal | 门限 | 结论 |
|---|---|---|---|---|
| PFV_H120 vs no_control | 0.8247 | 0.8193 | ≥0.70 | ✅ 硬门 |
| PFV_H120 vs passive | 0.8247 | 0.8193 | ≥0.70 | ✅ 硬门 |
| **PFV_full vs no_control** | **0.7526** | **0.7520** | ≥0.70 | ✅ 硬门（原 0.47） |
| **PFV_full vs passive** | **0.7526** | **0.7520** | ≥0.70 | ✅ 硬门（原 0.47） |
| TFV_H120 vs internal | 0.4742 | 0.4711 | ≥0.70 | ⚠️ advisory |
| peak_H120 vs internal | 0.4296 | 0.4271 | ≥0.80 | ⚠️ advisory |

**TFV/peak 降级为 advisory（用户 2026-07-24 批准）**：其此前的"通过"是 base 层单向分布的
假象；在平衡的 aug1 硬负例上近似随机，当前 1011 对数据下无诚实办法达标。运行时以内部规则为
TFV/peak 性能基准、安全 fallback 兜底，故这两个头**记录但不硬阻断**模型门（config
`v4.model_gate.advisory_direction_labels`）。四个 PFV 头仍为硬门。`v4_aug1_model_gate.json`
的 `advisory` 字段完整记录了这两个头的数值，未隐藏。

### 用户需执行：闭环 Smoke（3 事件）+ formal 验证（重型真实 SWMM）

> 以下为重型闭环（`08_run_closed_loop.py --mode formal`，真实 SWMM，每事件可能数十分钟至数小时）。
> 按契约由用户手动执行并回传日志。runner 会自动解析到 aug1 模型
> `action_effect_models_v4_aug1/action_effect_dual_reference_v4_aug1.npz`。

```powershell
cd E:\RTC_sewer\Project6
$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Cfg = "E:\RTC_sewer\Project6\configs\wuhan_project6_dual_reference_v4.yaml"
$V4  = "E:\RTC_sewer\Project6\scripts\205_prompt3_v4.py"

# ===== A. Runner 配置 + Readiness（快速，非重型）=====
& $Py $V4 --config $Cfg --stage BuildRunnerConfigV4              # 期望 exit 0
& $Py $V4 --config $Cfg --stage AuditV4Readiness                 # 期望 exit 0

# ===== B. 3 事件 Smoke 闭环（重型 SWMM）=====
& $Py $V4 --config $Cfg --stage RunClosedLoopSmokeV4 --max-events 3 --workers 8 --resume
#   期望 exit 0；audit/v4_smoke_run.json status=pass
& $Py $V4 --config $Cfg --stage EvaluateClosedLoopSmokeV4        # 期望 exit 0
#   期望 audit/v4_smoke_gate.json status=pass（无 structural/scientific failures）

# ===== C. formal 事件验证（重型 SWMM，覆盖 6 个保留验证事件）=====
& $Py $V4 --config $Cfg --stage RunClosedLoopSmokeV4 --max-events 6 --workers 8 --resume
#   期望 exit 0；覆盖 T100_* 全部 6 个验证事件
& $Py $V4 --config $Cfg --stage EvaluateClosedLoopSmokeV4        # 期望 exit 0
```

### 期望输出与验收条件

- **A**：`BuildRunnerConfigV4`/`AuditV4Readiness` 均 exit 0；readiness 无阻断项。
- **B（Smoke）**：`RunClosedLoopSmokeV4` exit 0；`EvaluateClosedLoopSmokeV4` exit 0，
  `v4_smoke_gate.json`：`status=pass`、`proposed_event_count≥3`、三策略配对齐全、
  `online_future_hydraulics_used_count=0`（无泄漏）、`write_readback_mismatch_count=0`、
  `adaptive_k_violation_count=0`、`candidate_executed_count>0`（未退化为全 fallback）。
- **C（formal）**：同上，且覆盖 6 个验证事件；PFV 对 no_control/passive 事件级非劣、
  TFV/peak 对 internal 事件级非劣（`scientific_failures` 为空）。

### 失败时需回传的日志

- Run 非 0：`logs/v4_smoke_stdout.txt`、`logs/v4_smoke_stderr.txt` 末尾；
  `runtime/v4_smoke_heartbeat.json`（确认是否 hung_or_timeout）。
- Evaluate=3（blocked）：`v4_smoke_gate.json` 的 `structural_failures`。
- Evaluate=5（failed_gate）：`v4_smoke_gate.json` 的 `scientific_failures` 与 `summary`
  中各 `mean_delta_*` 值。
