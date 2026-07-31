# Project6 V4 Final — Opportunity 覆盖审计科学门裁决请求（含本轮两处已修复缺陷备案）

- 日期：2026-07-23
- 提交人：监督执行 Agent（按用户纪律：机械 bug 经确认后本地修复，科学标准变更交付方裁决）
- 状态：**链条暂停在 AuditOpportunityCoverage exit=5（scientific_fail）**，等待 codex 裁决

---

## 1. 当前链路状态（全部真实运行证据）

| Stage | 结果 | 证据 |
|---|---|---|
| AuditContracts | exit=0 pass | audits/stage_status/AuditContracts.json |
| BuildEventInventory | exit=0 pass（358 事件 / 244 合格） | inventory/event_inventory.csv |
| PlanOpportunityPool | exit=0 pass（244 案例，runner 白名单=run_swmm_fixed_action） | opportunities/opportunity_scan_plan.csv |
| ScanOpportunityPool | exit=0 pass（**244/244 case 真实 SWMM 全部 pass**，16 并行约 30 分钟） | opportunities/runs/*/completion.json（status 全 pass） |
| BuildOpportunityPool | exit=0 pass（1152 checkpoints；908 responsive / 244 low_opportunity） | opportunities/opportunity_pool.csv |
| AuditOpportunityCoverage | **exit=5 scientific_fail** | opportunities/opportunity_coverage_audit.json |

当前 SHA：config_sha=8e2c0afe…，code_git_sha=3c1c7c0a…（含本轮两处本地修复，见 §4）。

## 2. 科学门失败详情

`opportunity_coverage_audit.json` 8 项检查仅 1 项失败：

```json
"four_responsive_per_event": false   // 其余 7 项全部 true
```

每事件 responsive checkpoint 数分布（244 事件全部有 responsive 与 low 对照）：

| responsive 数/事件 | 事件数 |
|---|---|
| 4 | 182 |
| 3 | 56 |
| 2 | 6 |

## 3. 根因：规划器与审计器口径矛盾（非运行时错误，重跑无效）

- 规划器（`sewerrtc/v4/opportunity.py::plan_opportunity_scans`）按事件时长布点，每事件总 checkpoint 3~5 个（含 1 个 low_opportunity 对照），且受 30 分钟最小间距约束。短历时事件物理上放不下 4 个 responsive 点（总数=3 的事件只能有 2 个 responsive）。
- 审计器（`sewerrtc/v4/opportunity.py::audit_opportunity_coverage`，checks 键 `four_responsive_per_event`：`by_event.size().ge(4).all()`）要求**全部 244 事件**每个 ≥4 responsive。
- 两者同为本次交付代码，标准互斥：只要事件库包含短历时事件，此门恒 fail。

## 4. 备案：本轮已确认并本地修复的两处机械缺陷（请在上游吸收）

1. `sewerrtc/v4/simulation.py::run_prepared_case`——原代码只检查 `hotstart_dir is not None` 未从 kwargs 移除该键，而 `run_swmm_fixed_action` 签名无此参数，导致 244 case 全部 `TypeError`（此前表现为 BrokenProcessPool）。已改为 `kwargs.pop("hotstart_dir", None)`，fail-closed 语义不变。修复后 244/244 真实 SWMM pass。
2. `sewerrtc/v4/opportunity.py::audit_opportunity_coverage`——checks 含 `numpy.bool_`（`Series.ge().all()` 产物），`json.dump(allow_nan=False)` 拒绝序列化导致阶段崩溃 exit=1。已在 return 前加 `checks = {k: bool(v) for k, v in checks.items()}`。

以上两处均有回归测试通过记录（相关 v4 测试 24 passed；opportunity 系列 12 passed）。

## 5. 请 codex 裁决（二选一，或给出第三方案）

- **方案 A（改规划器）**：短事件加密 checkpoint 或放宽间距，使所有合格事件可满足 ≥4 responsive；副作用：需重跑受影响事件的 SWMM 扫描（约 62 事件）。
- **方案 B（改审计口径）**：`four_responsive_per_event` 改为可满足标准，例如"≥N 个事件满足 4-responsive 且所有事件 ≥2"（当前数据：182 个事件满足 4，最低 2）；或将开发集限定为满足 4-responsive 的事件子集。
- 无论哪个方案，请同步给出：新验收阈值、是否需要重新生成计划/重扫、以及测试更新。

## 6. 复现命令（供验证）

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& 'E:\RTC_sewer\Project6\scripts\project6_runs\RUN_PROJECT6_V4_FINAL.ps1' `
  -Stage AuditOpportunityCoverage `
  -Config 'E:\RTC_sewer\Project6\configs\wuhan_project6_v4_final.yaml'
# 预期 exit=5，opportunity_coverage_audit.json 中仅 four_responsive_per_event=false
```
