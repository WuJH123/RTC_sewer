# Project6 V4 Final 缺陷报告：Opportunity→Peak Boundary 管线断链

日期：2026-07-28（监督运行首轮）
状态：阻断（ScanOpportunityPool exit=3，无法通过重试恢复）

## 已验证通过的部分（无需改动）

- `python -m compileall sewerrtc\v4 scripts\project6_v4_final.py` → exit=0
- Runbook 列出的 21 个测试文件 → 59 passed in 15.56s
- `AuditContracts -DryRun` → exit=0，scope_complete=true，
  active_dwf_flow_rows=0，network SHA c44b315c…、physical SHA 96b5bd36… 均匹配
- `BuildEventInventory` → exit=0，events=358，eligible=244

## 阻断现场

```text
Stage: ScanOpportunityPool -Workers 16 -Resume
exit_code=3, status=incomplete
reason: "executable plan missing"
plan:   outputs\project6_dual_reference_v4\final_v4\opportunities\opportunity_scan_plan.csv
run_uuid: 6005f9c6-4f4c-4220-86bc-ccd892ad10e7
config_sha: 3a5a6e74b2314c90a418e8c66382e594849915ecf750390a1a1d4470cbcfeeee
code_git_sha: f1bcd2fefae666c3d3c34735d1cfd50627b6dda048d8437643940d4bf6d5a40c
```

## 根因：3 处缺失生成者（代码层断链，非环境问题）

### 1. `opportunities/opportunity_scan_plan.csv` 无任何 Stage 生成

- `sewerrtc/v4/pipeline.py` L186 将 `ScanOpportunityPool` 注册为 run-case
  阶段，L346-L365 要求计划表存在且含列
  `case_id / runner_function / runner_kwargs`。
- 52 个 Stage（含 `BuildEventInventory`，其 handler 只写事件清单）中没有
  任何 handler 写出该文件；全仓库检索 `opportunity_scan_plan` 仅
  pipeline.py L186 一处引用。
- Stage 注册表中 `BuildEventInventory`(2) → `ScanOpportunityPool`(3) 之间
  没有 Plan 阶段。

### 2. `opportunities/opportunity_pool.csv` 列合同不满足

- `ScanOpportunityPool` 的 artifact（pipeline.py L133）由
  `_run_case_stage_handler` L390-395 写入，内容是 **completion manifest**
  （案例完成清单）。
- 下游 `AuditOpportunityCoverage`（`sewerrtc/v4/opportunity.py`
  `audit_opportunity_coverage`）要求列：
  `event_id / checkpoint_min / opportunity_class / phase /
  rainfall_family / risk_level`，且 `opportunity_class` 取值需含
  `responsive` 与 `low_opportunity`。
- 即使补上扫描计划，该阶段也会以 `blocked: missing_columns` 失败。
  缺失的是"扫描 detail → opportunity 评分 → checkpoint 分类汇聚"这一层。

### 3. `opportunities/peak_candidate_catalog.csv` 无任何 Stage 生成

- `PlanPeakBoundary`（pipeline.py L457-L467）以它为唯一输入，缺失时
  exit=3 `peak_candidate_catalog_missing`。
- 全仓库（py 源码）检索 `peak_candidate_catalog` 仅 pipeline.py 两处引用，
  无生成者。Peak Boundary 段同样无法推进。

## 对修复的合同要求（供实现方对齐）

1. 计划生成必须从 `BuildEventInventory` 的 244 个 eligible 事件出发，
   `runner_function` 仅允许 `run_swmm_fixed_action` /
   `run_swmm_dynamic_internal`（`sewerrtc/v4/simulation.py` L33-38 白名单），
   `runner_kwargs` 需完全物化（INP/降雨路径、时长、输出路径），禁止
   hotstart（L47-48 fail-closed 已存在）。
2. 汇聚阶段产物必须满足 `audit_opportunity_coverage` 的列合同与
   `peak_candidate_catalog.csv` 的下游 `build_peak_boundary_plan` 输入合同。
3. 不得改动阈值、K≤8、5/10 分钟时间合同、DWF 审计与 SHA 溯源逻辑；
   新 Stage 需进注册表并保持 52-Stage 文档与
   `PREREQUISITES`（pipeline.py L199 起）一致。
4. 网络文件保持冻结：runner 不得在运行时修改 DWF。

## 验收条件（修复后由监督方复跑）

```powershell
Invoke-V4Stage -Stage BuildEventInventory            # exit=0（已通过）
Invoke-V4Stage -Stage ScanOpportunityPool -Workers 16 -Resume   # exit=0, scope_complete=true
Invoke-V4Stage -Stage AuditOpportunityCoverage       # exit=0（>=8事件、>=32 responsive、间隔30min、4相位、>=3雨型、>=3风险级）
Invoke-V4Stage -Stage PlanPeakBoundary               # exit=0
```

任一非零即停，保留 `final_v4/audits/stage_status/<Stage>.json` 与 logs。
