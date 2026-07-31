# Project6 V4.1 Compact Rescue 交接文档（2026-07-30）

> 用途：开新对话继续推进 V4.1 救援管线。本文档所有"已完成"结论均以
> **当日磁盘证据**（stage_status JSON、artifact、pytest 输出）核验，不是凭
> 旧对话声称。新对话的 agent 必须先读本文件与 `AGENTS.md`，再读
> `docs/runbooks/PROJECT6_PFVFIRST_DUALFALLBACK_V3_RUNBOOK.md` §21–§22。

---

## 0. 一句话现状

**V4.1 Phase-1（消融+选择+确定性五种子训练）与全新评价计划、Calibration
四分支 SWMM 生成（100/100）已在最终代码 SHA `3e1f3d47…` 下全部通过；
精确断点 = `BuildV4CompactCalibrationV1`（下一条要跑的命令）。
一次性 Locked 尚未消费（intent/result 均不存在）。
下游闭环 Run stage 仍是 fail-closed 占位，且 3 个 Stage 未注册。**

---

## 1. 已完成工作总结（磁盘证据核验，2026-07-30 07:00–07:51 完成）

### 1.1 关键 SHA（全部一致，证据链有效）

| 项 | 值 |
|---|---|
| 当前 working code SHA | `3e1f3d472f08d5167a4da60aa9d65c751531c677ac4f7cb37b97118cac06c36a` |
| config SHA | `b0b7dbda2eeffbf2b51f31a9ce72fd1adb7c7f0954f312bd2e1f15e800b1eba9` |
| V4.1 模型 SHA | `5c0160bf46b18d03cfdaa714f03b694b33b2e0c20672b076b97f2ce49d660a09` |
| Calibration run input SHA | `c26479a70e26c261b15b229b5460fc3247760169673330392f53f5268c9fc426` |

配置：`configs\wuhan_project6_v4_final.yaml`；
输出根：`outputs\project6_dual_reference_v4\final_v4`；
驱动：`scripts\project6_v4_final.py`。

**已验证：当前 code SHA 与所有隔夜 stage 证据的 `code_git_sha` 一致。**
只要不再改任何被 `working_code_sha` 覆盖的文件（`sewerrtc/v4/*.py`、
`scripts/project6_v4_final.py`、`pyswmm_runner.py`、`kpi_metrics.py`、
`v4_candidate_generator.py`、RUN ps1），链条继续有效。

### 1.2 Stage 状态（读自 `audits/stage_status/*.json`）

| Stage | 状态 |
|---|---|
| FreezeV4OfflineV0Evidence … TrainV4CompactTrueStateV1（Phase-1 全部 10 个） | **pass, exit=0** |
| PlanV4CompactCalibrationLockedV1 / AuditV4CompactEvaluationPlanV1 | **pass** |
| RunV4CompactCalibrationV1 | **pass**，completed=100, failed=0, scope_complete=true |
| BuildV4CompactCalibrationV1 及之后所有 stage | **ABSENT（未运行）** |

- Calibration runs 目录含 100 个 `completion.json`（四分支：candidate /
  no_control / dynamic_internal_rules / hold_previous，authoritative SWMM，
  `wuhan_v8_storage_retrofit.inp`）。
- Locked 目录只有 `plan.csv`，**无 runs**——Locked SWMM 从未启动。
- `models/v4_compact_v1/` 中 **不存在** `v4_compact_v1_calibration.json`、
  `v4_compact_v1_locked_intent.json`、`v4_compact_v1_locked_evaluation.json`、
  `v4_predictive_generalization_gate.json` → **一次性 Locked 机会完好**。

### 1.3 V4.1 模型选择结果（仅 Train-grouped 证据，未读任何 Locked）

- 架构：紧凑 B（per-head：pfv=B, tfv=B, peak=A），peak 用 consistency 形式。
- 特征组合：`candidate_minus_di_only`；per-head 特征块见
  `v4_compact_v1_selection.json` 的 `head_feature_blocks`。
- 冻结契约：5 seeds、pfv hurdle、full-event heads 禁用、online K∈{1,2} 禁用、
  per-head standard scaler、seed-ensemble std 不确定度。
- n_train=1200；`reads_old_locked=false`、`reads_new_*=false`。

### 1.4 全新评价计划（§12，已冻结）

- v4.1_calibration：4 事件 / 100 cases；v4.1_locked：8 事件 / 200 cases；
  accrual reserve：4 事件 / 100 cases。全部来自 Reserve，冻结于任何新标签
  之前；旧 V4.0 Calibration/Locked 已标记 `eligible_for_v1_official_evaluation=false`。
- Plan handler 已материaliz可执行 plan：`v4_compact_eval/{calibration,locked}/plan.csv`
  + `planning/schedules/**` + `v4_compact_branch_plan.csv`（freeze 内含
  `executable_plan_sha256` / `branch_plan_sha256`）。

### 1.5 测试（2026-07-30 本机实测，非声称）

```
tests\test_v4_compact_phase1.py + test_v4_compact_phase2.py +
test_v4_closed_loop.py + test_v4_closed_loop_pipeline.py +
test_v4_compact_online_adapter.py + test_v4_policy_lock.py +
test_v4_formal_blind.py + test_v4_reporting.py + test_v4_final_stage_registry.py
→ 41 passed, 136 warnings (174s)
```
（警告只来自 ElasticNet 诊断拟合未收敛，属已知无害。）

### 1.6 诚实预警：Gate 可能不通过

Train-grouped CV（`completion.json`）显示连续头 R² 仍为负
（pfv −0.148 / tfv −0.221 / peak −0.140），top-5 feasible recall=1.0、
fallback_rate=0。**预测泛化门很可能给出 `scientific_fail`（exit=5）。**
这是合法科学结果：必须冻结负结果，禁止用 Locked 调参；如继续研究开 V4.2。
不要把 exit=5 当运行错误"修复"。

---

## 2. 技术难点与踩坑（隔夜运行实录）

1. **重复后台进程争抢同一输出根**：前台命令被终端时间片中断后重复启动，
   曾产生 8 个重复学习曲线进程。解决：只用一个
   `Start-Process ... -PassThru` 后台实例 + 每 ≤60s 轮询 PID；启动新长任务前
   先 `Get-Process python` 检查孤儿进程（只允许停掉能被 Project6 PID/日志
   证明的孤儿进程）。
2. **PowerShell 不支持 `&&`**：一律用 `;` 或 `if ($LASTEXITCODE -ne 0) { exit }`。
3. **code SHA 极敏感**：任何 `sewerrtc/v4/*.py` 改动都会改变
   `working_code_sha`，使全部 stage 证据 stale。这是本项目最贵的坑——
   见 §3 的执行顺序决策。
4. **`CompactHeadSpecificModel` 只有 `to_bytes()` 没有 `from_bytes`**：
   加载用 `pickle.loads(path.read_bytes())`（Phase-2 handler 已如此实现）。
5. **一次性 Locked 保护顺序（V4.1 改进版）**：先校验全部输入存在
   （缺输入返回 exit=3、不写 intent、不烧机会）→ 写 immutable intent →
   才读 Locked 数据。禁止删 intent/result 强行重跑。
6. **长任务监控**：不用固定长 Sleep；resource_blocked ≠ 科学失败，
   scientific_fail ≠ 运行错误。

---

## 3. 后续工作（按顺序）与关键决策点

### 决策点 D1（新对话第一件事）：先补下游代码还是先消费 Locked？

规格（pasted-text V4.1 下游提示词）明确要求 **"先实现全部下游代码 →
最终 SHA 重盖章 → 最后才消费一次性 Locked"**，因为 Locked 之后再改代码
会破坏证据链。隔夜 agent 重盖章时下游 Run stage 仍是 fail-closed 占位。
因此推荐路线 A：

**路线 A（规格忠实，推荐）**
1. 补齐 §4.3 所列全部下游缺口（代码+测试），全程不碰冻结数据；
2. 代码 SHA 必然变化 → 重跑重盖章序列（Phase-1 确定性重训 ~50 min +
   Plan/Audit + Calibration SWMM ~9 min@16 workers，全部可复现）；
3. 再走 Build/Audit Calibration → Calibrate → Locked SWMM → 一次性
   Evaluate → Gate。

**路线 B（快速出 Gate 结论）**：当前 SHA 下直接续跑断点直至 Gate。
代价：Gate 之后实现下游会改 SHA，闭环阶段将带着"上游证据 stale"的
审计负担；若 Gate=scientific_fail（§1.6 预警）则路线 B 反而更快拿到
负结果结论，下游实现留给 V4.2。**若首要目标是尽快知道 V4.1 是否
generalize，可选 B 并接受上述代价。**

### 3.1 立即可跑的断点续跑命令（路线 B / 或路线 A 第 3 步）

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
$Py='E:\RTC_sewer\Project6\.venv\Scripts\python.exe'
$V4='scripts\project6_v4_final.py'; $Cfg='configs\wuhan_project6_v4_final.yaml'

& $Py $V4 --stage BuildV4CompactCalibrationV1 --config $Cfg
& $Py $V4 --stage AuditV4CompactCalibrationV1 --config $Cfg
# Locked SWMM：先 Canary 再放量（exit 3 = 未完成待续，允许）
& $Py $V4 --stage RunV4CompactLockedV1 --config $Cfg --workers 1 --limit 1 --resume
& $Py $V4 --stage RunV4CompactLockedV1 --config $Cfg --workers 16 --resume
& $Py $V4 --stage BuildV4CompactLockedV1 --config $Cfg
& $Py $V4 --stage AuditV4CompactLockedV1 --config $Cfg
& $Py $V4 --stage CalibrateV4CompactV1 --config $Cfg
& $Py $V4 --stage EvaluateV4CompactLockedV1 --config $Cfg   # 一次性！
& $Py $V4 --stage AuditV4PredictiveGeneralizationGateV1 --config $Cfg
```

Gate 分支：exit 0=pass（authorizes_closed_loop=true，继续闭环）；
3=underpowered（只许用预冻结 accrual，不改模型/阈值）；
5=scientific_fail（冻结负结果，禁闭环/Challenge/Formal，开 V4.2）。

### 3.2 风险矩阵

| 风险 | 影响 | 缓解 |
|---|---|---|
| Gate scientific_fail（CV R² 已为负） | 闭环全线停止 | 预先接受；负结果入档；V4.2 用新事件 |
| Locked 前误改代码 | 全链 stale | 改码前先跑完 Locked+Gate（路线 B）或按路线 A 整体重盖 |
| 误删 intent 重跑 Locked | 证据作废 | 绝对禁止；见 runbook §21 Step 8 |
| 重复进程写同一输出根 | manifest 损坏 | 单实例后台 + PID 轮询 |

---

## 4. 代码与文件路径

### 4.1 核心模块（V4.1 新增/修改）

| 文件 | 职责 |
|---|---|
| `sewerrtc/v4/pipeline.py` | ALL_STAGES / STAGE_ARTIFACTS / RUN_STAGE_PLANS / PREREQUISITES / build_registry（Phase-1+2+闭环门全部接线；`PlanExactClosedLoop` 现在 gate 于 `AuditV4PredictiveGeneralizationGateV1`+`AuditGATClosedLoopReadiness`，旧 `EvaluateV4Locked` 已无授权能力） |
| `sewerrtc/v4/pipeline_v4_compact.py` | Phase-1 handler（Freeze/§3–§11） |
| `sewerrtc/v4/pipeline_v4_compact_eval.py` | Phase-2 handler（§12–§16；Plan 已可 materialize 可执行 plan + branch plan；§16 一次性保护） |
| `sewerrtc/v4/v4_compact_model_ops.py` | `CompactHeadSpecificModel`（`to_bytes`=pickle；无 from_bytes） |
| `sewerrtc/v4/v4_compact_eval_ops.py` | 纯逻辑：split 计划/审计、calibrate、locked 评价、gate 打分 |
| `sewerrtc/v4/pipeline_v4_closed_loop.py` | 下游门：GAT readiness、闭环 Plan（development-only）、Run 占位（故意 fail-closed） |
| `sewerrtc/v4/online_v4_compact.py` | V4.1 在线特征适配器（570 维冻结合同；缺字段拒绝零填充） |
| `sewerrtc/v4/closed_loop.py` | UCB 门（PFV-first）、SURROGATE_ABLATIONS |

### 4.2 关键数据/证据路径（输出根 `outputs\project6_dual_reference_v4\final_v4`）

- `audits/stage_status/*.json` — 每 stage 权威状态（先看这里）
- `models/v4_compact_v1/` — 模型 pkl、selection、completion、消融证据
- `v4_compact_eval/planning/` — plan freeze、可执行 plan、schedules、branch plan
- `v4_compact_eval/calibration/runs/**` — 100 case 四分支 SWMM 证据
- `inventory/event_usage_ledger.csv` — 事件账本（reserve 分配）
- 契约：`docs/contracts/PROJECT6_V4_PREDICTIVE_GENERALIZATION_GATE_V1.json`
- Runbook：`docs/runbooks/PROJECT6_PFVFIRST_DUALFALLBACK_V3_RUNBOOK.md` §21–§22

### 4.3 未完成缺口清单（路线 A 的工作量）

1. **未注册 Stage**（不在 ALL_STAGES，无 handler）：
   `EvaluateModelSafetyGateV4`、`PlanChallenge`、`PlanFormalBlind`。
2. **Run 闭环占位**：`RunExactClosedLoop` / `RunSurrogateClosedLoop` 返回
   blocked（`v41_online_feature_and_reference_forecaster_required`）。需要把
   `online_v4_compact.CompactOnlinePredictor` 绑定进 `pyswmm_runner` 的
   10-min 滚动控制（只执行首步；A/B/C/D 消融映射；禁止 true-state 进 D、
   禁止 Exact evaluator 进正式 D；OOD/超时 fallback；cumulative budget；
   actual/readback 记账）。
3. **缺失测试文件**（规格 §15 要求）：`test_v4_gat_closed_loop.py`、
   `test_v4_challenge_pipeline.py`、`test_v4_model_safety_gate_v41.py`。
4. Challenge（≥8 全新事件、一次性）与 Formal Blind（≥24 全新事件）链、
   `LockPolicy` 需按规格补 `build_policy_lock()` 的 15 项 SHA 冻结校验。
5. 论文产物 Stage（BuildPaperResults/Figures/Tables/ReproducibilityBundle）
   已注册但 handler 内容需核对是否覆盖 A/B/C/D + bootstrap + Wilcoxon。

---

## 5. 代码逻辑验证要点（新对话应先自查）

1. `working_code_sha(Path('.'))` 是否仍等于 `3e1f3d47…`（不等 → 全链 stale，
   走路线 A 重盖）。
2. `models/v4_compact_v1/` 下 4 个 Phase-2 产物（calibration/intent/result/gate）
   是否仍不存在（存在 intent/result → 一次性已烧，绝不能删）。
3. `PREREQUISITES["PlanExactClosedLoop"]` 必须含
   `AuditV4PredictiveGeneralizationGateV1`（防止旧门授权闭环）。
4. `pipeline_v4_closed_loop.predictive_gate_authorizes_closed_loop` 只认
   `status=="pass" and authorizes_closed_loop is True`。
5. 41 项 v4 测试是否仍绿（§1.5 命令）。

## 6. 快速启动指南（新对话开场三条）

```powershell
# 1) 核对断点与 SHA
Set-Location 'E:\RTC_sewer\Project6'
.\.venv\Scripts\python.exe -c "from pathlib import Path; from sewerrtc.v4.runtime import working_code_sha; print(working_code_sha(Path('.').resolve()))"
# 2) 核对 stage 状态（脚本见 §1.2 所列 stage 名循环读 stage_status）
# 3) 跑测试基线
.\.venv\Scripts\python.exe -m pytest tests\ -q -k "v4"
```

## 7. 关键决策记录

1. §13 SWMM 采用 V3 pilot 四分支 runner 复用（用户已确认），reference root
   共享 `pilot` 参考缓存；后经用户手改，Plan stage 直接 materialize 可执行
   plan（复用 `materialize_pilot_candidates`+`build_pilot_branch_plan`）。
2. §16 一次性保护顺序改进：输入齐全才写 intent；intent 用 plan-freeze SHA，
   避免提前读 Locked 标签。
3. Gate verdict→exit：pass=0 / underpowered=3 / scientific_fail=5。
4. 模型选择只用 Train-grouped 证据（selection.json 有显式 reads_* 标志）。
5. 闭环授权只能来自新 V4.1 预测泛化门；旧 EvaluateV4Locked 降级为无授权。
6. Run 闭环在在线适配器绑定完成前故意 fail-closed，防止 V3 控制器顶替。
7. V4.0 Offline Gate 已加 `gate_role=offline_integrity_only`，不授权闭环。

## 8. 当前阻塞项

| # | 阻塞 | 解除条件 |
|---|---|---|
| 1 | Gate 未评价（Locked 未消费） | 跑完 §3.1 序列 |
| 2 | 3 个 Stage 未注册 + 闭环 Run 占位 | 完成 §4.3 缺口（路线 A） |
| 3 | 闭环/Challenge/Formal 全部 blocked | Gate=pass 且 §4.3 完成 |
| 4 | CV R² 为负的泛化风险 | 由一次性 Locked+Gate 给出最终裁决 |

---

*本文档由 2026-07-30 磁盘证据生成；引用的隔夜执行记录见用户上传的
对话导出（最后一轮止于 Calibration 16-worker 启动，磁盘证据显示其后
Calibration 已 100/100 完成，Locked 未启动）。*
