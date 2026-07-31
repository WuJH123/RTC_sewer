# Project6 V4 Oracle/Pareto 可行性分析

## 这段代码回答什么问题

它不是在线控制器，而是离线“上限试验”：对每个开发事件，使用真实 SWMM 评估一组广覆盖的完整事件动作时序，然后建立 PFV–TFV–Peak–动作成本 Pareto 前沿，并检查是否存在同时满足以下合同的方案：

- PFV ≤ min(No-control, Executable Passive) + 冻结容差；
- TFV ≤ Internal rules + 冻结容差；
- Peak ≤ Internal rules + 冻结容差。

若受工程约束的 Oracle 可行而 Proposed 不可行，主要问题在模型、候选搜索或控制器；若放松 rate/dwell/K 后可行而工程约束下不可行，主要问题在动作约束；若两者都不可行且搜索覆盖与 Pareto 前沿已收敛，则当前 36 个设施与三目标可能存在物理冲突。后者仍是“在声明的动作邻域内未发现可行解”，不是数学上的全局不可行证明。

## 安装位置

把 `oracle_pareto_v4.py` 复制为：

```text
E:\RTC_sewer\Project6\scripts\206_oracle_pareto_v4.py
```

## 先做 3 事件 Smoke

```powershell
cd E:\RTC_sewer\Project6
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"

& $Py scripts\206_oracle_pareto_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml `
  --engineering-config configs\wuhan_project6_engineering36.yaml `
  --actuators-csv outputs\closed_loop_paired_no_controls\formal\project6_no_control_repair_formal_30_v8\control_actuator_table.csv `
  --stage all `
  --event-limit 3 `
  --workers 4 `
  --resume
```

## 正式开发分析

选择 12–20 个从未进入 Calibration、Validation 或 Formal 的开发事件：

```powershell
& $Py scripts\206_oracle_pareto_v4.py `
  --config configs\wuhan_project6_dual_reference_v4.yaml `
  --engineering-config configs\wuhan_project6_engineering36.yaml `
  --actuators-csv outputs\closed_loop_paired_no_controls\formal\project6_no_control_repair_formal_30_v8\control_actuator_table.csv `
  --stage all `
  --event-ids "EVENT_A,EVENT_B,EVENT_C" `
  --workers 12 `
  --resume
```

建议分阶段运行，任何阶段非 0 都停止：

```powershell
& $Py scripts\206_oracle_pareto_v4.py ... --stage references --workers 12 --resume
& $Py scripts\206_oracle_pareto_v4.py ... --stage plan
& $Py scripts\206_oracle_pareto_v4.py ... --stage run --workers 12 --resume
& $Py scripts\206_oracle_pareto_v4.py ... --stage analyze
```

## 关键输出

```text
outputs/project6_dual_reference_v4/oracle_pareto/
├─ oracle_run_manifest.json
├─ reference_results.csv
├─ reference_audit.json
├─ oracle_case_plan.csv
├─ oracle_case_plan_audit.json
├─ oracle_case_results.csv
├─ oracle_generation_audit.json
└─ analysis/
   ├─ all_event_candidate_feasibility.csv
   ├─ event_feasibility_summary.csv
   ├─ aggregate_feasibility_report.json
   ├─ aggregate_feasibility_classes.png
   └─ events/<event_id>/
      ├─ pareto_3d.csv
      ├─ pareto_4d.csv
      ├─ convergence.csv
      ├─ event_feasibility_summary.json
      ├─ pareto_pfv_tfv.png
      └─ pareto_pfv_peak.png
```

## 结果分类

- `feasible_found`：至少一个方案同时通过 PFV、TFV、Peak；
- `operational_constraints_block_feasibility`：放松 rate/dwell/K 后可行，工程约束下不可行；
- `pfv_safe_but_internal_performance_unreachable`：能守住 PFV，但 TFV/Peak 达不到 Internal；
- `internal_performance_reachable_but_pfv_unsafe`：能达到 Internal 性能，但 PFV 不安全；
- `objectives_reachable_separately_not_jointly`：各目标分别可达，但不能同时达到；
- `no_feasible_neighbourhood_solution`：搜索邻域内没有任何可行方案。

## 重要限制

1. Oracle 可以利用完整事件信息，因此不是可部署的在线策略。
2. 代码对已声明候选库做精确非支配筛选，但不能证明 36×多时间步连续空间的全局最优。
3. 无可行解时，必须同时检查候选覆盖、事件数量和 Pareto 收敛；不能直接宣称物理不可行。
4. 所有开发事件必须与 Calibration、Validation、Formal 严格隔离。
5. 不允许降低 V4 冻结门槛来制造“可行”。

## 方法依据

- PySWMM 支持逐步推进模拟以及运行时设置 link target setting，适合离线动作时序评估。
- EPA SWMM 是本项目权威水力模拟器。
- RTC 潜力评价应区分现有策略表现、静态基线和可实现上限。
- Pareto 前沿使用非支配关系，不以单一加权和掩盖 PFV、TFV 与 Peak 的冲突。
