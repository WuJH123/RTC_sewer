# V4.2 核心代码文件清单

> 生成时间：2026-07-31  
> 项目根目录：`E:\RTC_sewer\Project6\`  
> 所有文件均经实际目录扫描确认存在

---

## Step 1 — 稀疏状态恢复 (GAT)

### `sewerrtc/state/` (27 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `__init__.py` | 3 | 模块入口，导出 PFV 双回退状态契约 |
| `augmented_state.py` | 161 | 增强状态向量构建，拼接 GAT 重建特征 |
| `gat_audit.py` | 618 | GAT 模型审计逻辑，验证重建质量与一致性 |
| `gat_compatibility.py` | 360 | GAT 模型版本兼容性检查与迁移 |
| `gat_event_catalog.py` | 246 | GAT 训练事件目录管理，枚举可用事件 |
| `gat_event_leakage.py` | 177 | 事件级数据泄漏检测，确保训练/验证隔离 |
| `gat_holdout_eligibility.py` | 57 | 判定事件是否满足 holdout 留出条件 |
| `gat_holdout_trajectory.py` | 429 | 留出轨迹构建，生成独立验证数据 |
| `gat_independent_validation.py` | 281 | 独立验证集评估，检验 GAT 泛化能力 |
| `gat_registry.py` | 199 | GAT 模型注册表，管理多版本模型元数据 |
| `gat_robustness.py` | 997 | GAT 鲁棒性测试，覆盖噪声/缺失/扰动场景 |
| `gat_robustness_gate.py` | 155 | GAT 鲁棒性门控，判定是否通过验收 |
| `gat_selection.py` | 235 | GAT 模型选择逻辑，基于验证指标选优 |
| `gat_validation_provenance.py` | 539 | GAT 验证溯源，记录验证数据来源与链路 |
| `hotstart_acceleration.py` | 672 | SWMM 热启动加速，缓存初始状态减少预热 |
| `interface_cache.py` | 482 | 接口缓存层，缓存仿真中间结果避免重复计算 |
| `local_flow_features.py` | 43 | 局部流量特征提取（子模块） |
| `prompt2_completion_gate.py` | 99 | Prompt 2 完成门控，检查阶段准入条件 |
| `prompt2_gat_readiness.py` | 79 | Prompt 2 GAT 就绪检查，确认数据/模型到位 |
| `runtime_state_features.py` | 362 | 运行时状态特征提取，生成实时 PFV 特征 |
| `same_state_replay.py` | 568 | 同状态回放，相同初始条件下对比不同控制策略 |
| `state_clone_contract.py` | 99 | 状态克隆契约，定义状态复制的接口规范 |
| `state_clone_equivalence.py` | 903 | 状态克隆等价性验证，确保克隆前后一致 |
| `state_contract.py` | 159 | 状态契约基类，定义状态输入/输出接口 |
| `state_input_manifest.py` | 517 | 状态输入清单，记录每次仿真的输入参数 |
| `state_quality.py` | ~40 | 状态质量检查，验证状态向量合理性 |
| `temporal_state_buffer.py` | ~90 | 时序状态缓冲区，维护滑动窗口状态历史 |

### `sewerrtc/models/` — GAT 相关 (3 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `gat_reconstructor.py` | 90 | GAT 图注意力重建器，从稀疏观测恢复全场状态 |
| `graph_surrogate.py` | 124 | 图代理模型，基于图结构的快速水力代理 |
| `temporal_graph_surrogate.py` | 551 | 时序图代理模型，融合时间维度的图 surrogate |

### `sewerrtc/graph/` (4 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `__init__.py` | 1 | 模块入口 |
| `graph_builder.py` | 125 | 排水管网图结构构建，从 INP 生成邻接矩阵与节点特征 |
| `priority_zone.py` | 33 | 优先区域定义，标注关键监测分区 |
| `sensor_selection.py` | 29 | 传感器选址逻辑，选择最优监测点集合 |

---

## Step 2 — 多 Reference 水力代理模型

### `sewerrtc/v4/models_v42/` (13 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `__init__.py` | 4 | 模块入口，导出 V4.2 模型组件 |
| `actuator_action_encoder.py` | 105 | 执行器动作编码器，将控制动作嵌入隐空间 |
| `counterfactual_twin_dynamics.py` | 235 | 反事实孪生动力学，建模有/无控制的对照轨迹 |
| `global_system_decoder.py` | 103 | 全局系统解码器，从隐状态解码全局指标 |
| `graph_state_encoder.py` | 108 | 图状态编码器，将图结构观测压缩为隐向量 |
| `local_priority_decoder.py` | 113 | 局部优先解码器，输出关键节点的预测值 |
| `physics_losses.py` | 151 | 物理损失函数，水量平衡/质量守恒等约束 |
| `rainfall_encoder.py` | 48 | 降雨编码器，将降雨序列编码为特征向量 |
| `ranking_losses.py` | 145 | 排序损失函数，用于学习优先级排序 |
| `trainer.py` | 189 | 模型训练器，封装训练循环与优化策略 |
| `trajectory_losses.py` | 126 | 轨迹损失函数，衡量预测与真实轨迹差异 |
| `uncertainty.py` | 147 | 不确定性量化模块，估计预测置信区间 |
| `water_balance_backbone.py` | 96 | 水量平衡骨干网络，确保物理守恒的架构设计 |

### `sewerrtc/v4/` — 训练核心文件

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `v42_trainer.py` | ~1400 | V4.2 主训练器，协调多分支训练与门控验收 |
| `dataset.py` | ~300 | 数据集加载器，读取仿真轨迹并构建训练样本 |
| `v42_trajectory_builder.py` | ~800 | V4.2 轨迹构建器，从 SWMM 输出生成训练轨迹 |
| `v42_priority_contract.py` | ~300 | V4.2 优先级契约，定义 PFV 优先级的数据规范 |

---

## Step 3 — PFV-first 滚动 MPC

### `sewerrtc/control/` (37 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `__init__.py` | 1 | 模块入口 |
| `action_sequence_generator.py` | 476 | 动作序列生成器，枚举候选控制动作组合 |
| `actuator_scope.py` | 61 | 执行器作用域管理，确定当前可控设施 |
| `candidate_generator.py` | 389 | 候选动作生成器，基于启发式生成控制方案 |
| `canonical_action_order.py` | 93 | 规范动作排序，确保动作向量顺序一致 |
| `dual_reference_v4.py` | 330 | V4 双参照管理器，维护有/无控制双基准 |
| `event_pfv_budget.py` | 54 | 事件 PFV 预算，管理单次事件的 flooding 容许量 |
| `fallback_contract.py` | 38 | 回退契约，定义 MPC 失败时的降级策略 |
| `fallback_selector.py` | 29 | 回退选择器，在候选不可用时选择降级方案 |
| `formal_policy.py` | 29 | 形式化策略接口，定义控制策略的抽象基类 |
| `generic_gat_mpc.py` | 873 | 通用 GAT-MPC 控制器，基于图代理的滚动优化 |
| `generic_initial_policy.py` | 40 | 通用初始策略，MPC 冷启动时的默认策略 |
| `hierarchical_core26_residual10.py` | 172 | 分层控制器，26 核心设施 + 10 残余设施的分层优化 |
| `horizon_action_features.py` | 418 | 预测窗口动作特征，编码未来 H 步控制序列 |
| `horizon_objective.py` | 231 | 预测窗口目标函数，定义 MPC 优化目标 |
| `horizon_rollout.py` | 393 | 预测窗口滚动引擎，执行 H 步滚动优化 |
| `influence_candidate_generator.py` | 42 | 影响力候选生成器，基于管网拓扑生成候选 |
| `internal_fallback.py` | 25 | 内部回退逻辑，MPC 内部异常处理 |
| `mpc_controller.py` | 882 | MPC 主控制器，封装模型预测控制的完整流程 |
| `native_rule_audit.py` | 68 | 原生规则审计，检查 SWMM 内置控制规则 |
| `no_control_reference_predictor.py` | 61 | 无控制参照预测器，生成不干预的基准轨迹 |
| `nominal_policy.py` | 22 | 标称策略，正常运行时的控制策略 |
| `observation_driven_mpc.py` | 17 | 观测驱动 MPC，基于实时观测修正预测 |
| `passive_fallback.py` | 46 | 被动回退策略，不执行任何控制的兜底方案 |
| `pfvfirst_dualfallback.py` | 205 | PFV 优先双回退策略，核心控制策略入口 |
| `policy_base.py` | 22 | 策略基类，定义控制策略的公共接口 |
| `reference_roles.py` | ~60 | 参照角色定义，区分有控/无控参照的职责 |
| `retrofit_assets.py` | ~80 | 改造设施管理，维护管网改造资产清单 |
| `safety_filter.py` | ~15 | 安全滤波器，过滤不安全的控制动作 |
| `safety_guards.py` | ~130 | 安全守卫，多重安全检查防止危险控制 |
| `temporal_joint_36_controller.py` | ~1100 | 时序联合 36 设施控制器，全设施联合优化 |
| `temporal_joint_candidate_search.py` | ~350 | 时序联合候选搜索，在时序空间中搜索最优 |
| `temporal_joint_safety.py` | ~170 | 时序联合安全检查，确保时序动作序列安全 |
| `uncertainty_gate.py` | ~40 | 不确定性门控，基于预测不确定性决定是否控制 |
| `v4_action_authority.py` | ~140 | V4 动作授权，验证控制动作的合法性 |
| `v4_candidate_generator.py` | ~700 | V4 候选生成器，V4 专用的控制候选生成 |
| `v4_opportunity.py` | ~220 | V4 机会识别，发现可改善的控制时机 |

### `sewerrtc/simulation/` — 仿真核心 (11 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `__init__.py` | 1 | 模块入口 |
| `action_policies.py` | 489 | 动作策略集合，定义多种控制策略实现 |
| `baseline_trajectory.py` | 300 | 基准轨迹生成，无控制场景的仿真基线 |
| `branch_runner.py` | 11 | 分支运行器，管理仿真分支（薄封装） |
| `continuation_policy.py` | 16 | 续跑策略，支持仿真中断后恢复 |
| `controller_state.py` | 34 | 控制器状态管理，维护 MPC 运行时状态 |
| `kpi_metrics.py` | 72 | KPI 指标计算，flooding/peak 等关键指标 |
| `pyswmm_runner.py` | 3759 | SWMM 仿真运行器，核心仿真引擎（最大文件） |
| `runtime_contracts.py` | 270 | 运行时契约，定义仿真输入/输出的接口规范 |
| `swmm_event_builder.py` | 41 | SWMM 事件构建器，组装单次降雨事件仿真 |
| `trajectory_writer.py` | 17 | 轨迹写入器，将仿真结果写入磁盘 |

---

## Step 4 — 闭环评价

### `sewerrtc/evaluation/` (8 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `__init__.py` | 1 | 模块入口 |
| `evaluate_closed_loop.py` | 145 | 闭环评估主入口，执行有/无控制对比评估 |
| `kpi_contract.py` | 146 | KPI 契约定义，评估指标的规范与阈值 |
| `policy_sets.py` | 87 | 策略集管理，定义待评估的策略组合 |
| `project5_formal_gate.py` | 373 | Project5 正式门控，多指标综合验收判定 |
| `risk_stratified.py` | 384 | 风险分层评估，按风险等级分组评价控制效果 |
| `significance.py` | 55 | 统计显著性检验，判断改善是否具有统计意义 |
| `smoke_functionality_gate.py` | 181 | 冒烟功能门控，快速验证基本功能可用性 |

### `sewerrtc/v4/` — 验证与门控

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `v42_full_verification.py` | ~850 | V4.2 全量验证，执行完整的多维验收检查 |
| `v42_gate.py` | ~650 | V4.2 门控逻辑，各阶段准入/准出判定 |
| `v42_admission_gate.py` | ~580 | V4.2 准入门控，数据/样本进入训练池的资格审查 |

---

## 支撑模块

### Pipeline 流水线

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `sewerrtc/v4/pipeline.py` | ~4000 | 主流水线，编排六阶段训练的完整流程 |
| `sewerrtc/v4/pipeline_v42.py` | ~2300 | V4.2 专用流水线，含双参照与门控集成 |

### 交叉验证

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `sewerrtc/v4/v42_cv.py` | ~420 | V4.2 交叉验证框架，支持分组/嵌套 CV |
| `sewerrtc/v4/v42_sampling.py` | ~450 | V4.2 采样策略，控制训练样本的选取与平衡 |
| `sewerrtc/v4/v42_grouped_splits.py` | ~370 | V4.2 分组切分，按事件/来源分组防止泄漏 |

### 数据集

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `sewerrtc/v4/v42_final_datasets.py` | ~1200 | V4.2 最终数据集，封装训练/验证/测试数据 |

### 审计

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `sewerrtc/v4/v42_pool_audit.py` | ~820 | V4.2 数据池审计，检查数据完整性与质量 |
| `sewerrtc/v4/v42_pool_statistics.py` | ~620 | V4.2 数据池统计，生成数据分布与质量报告 |

### 合同 `docs/contracts/` (30 文件)

| 文件名 | 格式 | 功能简述 |
|--------|------|----------|
| `PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.md` | MD | PFV 优先双回退 10 分钟策略 V3 规范 |
| `PROJECT6_V3_CURRENT_TRUTH_CONTRACT.json` | JSON | V3 当前真理契约，定义基准真值 |
| `PROJECT6_V42_PRIORITY_PFV_CONTRACT.json` | JSON | V4.2 优先级 PFV 契约 |
| `PROJECT6_V4_CONTROL_SCOPE_CONTRACT.json` | JSON | V4 控制范围契约 |
| `PROJECT6_V4_CONTROL_SCOPE_CONTRACT_V2.json` | JSON | V4 控制范围契约 V2（扩展版） |
| `PROJECT6_V4_DATASET_CONTRACT.json` | JSON | V4 数据集契约，定义数据格式与质量要求 |
| `PROJECT6_V4_DATA_GENERATION_AUTHORIZATION_V3.json` | JSON | V4 数据生成授权 V3 |
| `PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json` | JSON | V4 最终流水线契约 |
| `PROJECT6_V4_LEARNING_TASK_V3.json` | JSON | V4 学习任务定义 V3 |
| `PROJECT6_V4_LOCKED_VALIDATION_ACCRUAL_V3.json` | JSON | V4 锁定验证累积 V3 |
| `PROJECT6_V4_MODEL_SAFETY_GATE_V3.json` | JSON | V4 模型安全门控 V3 |
| `PROJECT6_V4_MODEL_TRAINING_AUTHORIZATION_V4.json` | JSON | V4 模型训练授权 V4 |
| `PROJECT6_V4_NETWORK_NO_BASE_INFLOW_PROVENANCE.json` | JSON | V4 管网无基流入溯源 |
| `PROJECT6_V4_PILOT_FEASIBILITY_GATE_P3.json` | JSON | V4 Pilot 可行性门控 P3 |
| `PROJECT6_V4_PILOT_GATE_V2.json` | JSON | V4 Pilot 门控 V2 |
| `PROJECT6_V4_PREDICTIVE_GENERALIZATION_GATE_V1.json` | JSON | V4 预测泛化门控 V1 |
| `PROJECT6_V4_RECOVERY_CONTRACT_V2.json` | JSON | V4 恢复契约 V2 |
| `PROJECT6_V4_RECOVERY_CONTRACT_V3.json` | JSON | V4 恢复契约 V3 |
| `PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.json` | JSON | V4 恢复真理契约 |
| `PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.schema.json` | JSON | V4 恢复真理契约 Schema |
| `PROJECT6_V4_TRAIN1600_DATASET_V3.json` | JSON | V4 Train1600 数据集契约 V3 |
| `baseline_trajectory_plan_contract.json` | JSON | 基准轨迹计划契约 |
| `execution_status.schema.json` | JSON | 执行状态 Schema |
| `facility_semantics_contract.json` | JSON | 设施语义契约 |
| `forecast_contract.json` | JSON | 预报契约 |
| `gat_primary_selection_decision.json` | JSON | GAT 主选择决策记录 |
| `kpi_contract.json` | JSON | KPI 指标契约 |
| `project6_prompt2_import_contract.json` | JSON | Prompt 2 导入契约 |
| `project6_prompt3a_time_recovery_contract.json` | JSON | Prompt 3a 时间恢复契约 |
| `sentinel_nodes_provenance.json` | JSON | 哨兵节点溯源 |

### 配置 `configs/` (48 个 YAML 文件)

| 文件名 | 功能简述 |
|--------|----------|
| `debug.yaml` | 调试模式配置 |
| `full.yaml` | 完整运行配置 |
| `open_pystorms_beta.yaml` | PyStorms 开放基准测试配置 |
| `open_pystorms_beta_p0_only.yaml` | PyStorms P0 仅配置 |
| `oracle_pareto_v4_config_example.yaml` | Oracle-Pareto V4 配置示例 |
| `wuhan.yaml` | 武汉管网基础配置 |
| `wuhan_no_control_repair.yaml` | 无控制修复配置 |
| `wuhan_project4_native_probe.yaml` | Project4 原生探测配置 |
| `wuhan_project6.yaml` | Project6 基础配置 |
| `wuhan_project6_26_temporal_joint_ablation.yaml` | 26 设施时序联合消融配置 |
| `wuhan_project6_36_actual_repair_deploy_v1.yaml` | 36 设施实际修复部署 V1 |
| `wuhan_project6_36_actual_repair_deploy_windows_v1.yaml` | 36 设施 Windows 部署 V1 |
| `wuhan_project6_36_actual_repair_legacyonly_v1.yaml` | 36 设施仅遗留修复 V1 |
| `wuhan_project6_36_actual_repair_stable_residual_v1.yaml` | 36 设施稳定残差修复 V1 |
| `wuhan_project6_36_actual_repair_tfv_boost_profile_v1.yaml` | 36 设施 TFV 增强配置 V1 |
| `wuhan_project6_36_actual_repair_v1.yaml` | 36 设施实际修复 V1 |
| `wuhan_project6_36_causal_effect_coverage_v2.yaml` | 36 设施因果效应覆盖 V2 |
| `wuhan_project6_36_causal_effect_v3.yaml` | 36 设施因果效应 V3 |
| `wuhan_project6_36_core26_residual10_eventbudget_v1.yaml` | 26+10 分层事件预算 V1 |
| `wuhan_project6_36_engineering_templates_v1.yaml` | 工程模板 V1 |
| `wuhan_project6_36_engineering_templates_v2.yaml` | 工程模板 V2 |
| `wuhan_project6_36_engineering_templates_v3.yaml` | 工程模板 V3 |
| `wuhan_project6_36_engineering_templates_v4_verified_strata.yaml` | 工程模板 V4 已验证分层 |
| `wuhan_project6_36_hierarchical_eventbudget_h120_v2.yaml` | H120 分层事件预算 V2 |
| `wuhan_project6_36_hierarchical_eventbudget_h120_v2_residualfit.yaml` | H120 分层残差拟合 |
| `wuhan_project6_36_hierarchical_residual_v1.yaml` | 分层残差 V1 |
| `wuhan_project6_36_hierarchical_residual_v7.yaml` | 分层残差 V7 |
| `wuhan_project6_36_hierarchical_residual_v8_gateupdate.yaml` | 分层残差 V8 门控更新 |
| `wuhan_project6_36_recovered_v8_groups_stratified_v2.yaml` | 恢复 V8 分组分层 V2 |
| `wuhan_project6_36_recovered_v8_groups_v1.yaml` | 恢复 V8 分组 V1 |
| `wuhan_project6_36_temporal_joint.yaml` | 时序联合配置 |
| `wuhan_project6_36_temporal_joint_peakfixed.yaml` | 时序联合峰值修复 |
| `wuhan_project6_36_temporal_joint_recovery_v2.yaml` | 时序联合恢复 V2 |
| `wuhan_project6_acceptance_probe.yaml` | 验收探测配置 |
| `wuhan_project6_amplitudeaware_probe.yaml` | 振幅感知探测配置 |
| `wuhan_project6_dual_reference_v4.yaml` | 双参照 V4 配置 |
| `wuhan_project6_engineering36.yaml` | 工程 36 设施配置 |
| `wuhan_project6_local_effect_probe.yaml` | 局部效应探测配置 |
| `wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml` | PFV 优先双回退 10min V3 主配置 |
| `wuhan_project6_pfvfirst_dualfallback_10min_v3_1.yaml` | PFV 双回退 V3.1 变体 |
| `wuhan_project6_pfvfirst_dualfallback_10min_v3_2.yaml` | PFV 双回退 V3.2 变体 |
| `wuhan_project6_pfvfirst_dualfallback_10min_v3_3.yaml` | PFV 双回退 V3.3 变体 |
| `wuhan_project6_v4_final.yaml` | V4 最终运行配置 |
| `wuhan_project6_v4_gate5r.yaml` | V4 Gate 5R 配置 |
| `wuhan_project6_v8_storage.yaml` | V8 蓄水改造配置 |
| `wuhan_project6_v8_storage_36.yaml` | V8 蓄水 36 设施配置 |

### 测试 `tests/` — `test_v42_*` 系列 (37 文件)

| 文件名 | 行数 | 功能简述 |
|--------|------|----------|
| `test_v42_13frame_real_rebuild.py` | 120 | 13 帧真实重建测试 |
| `test_v42_action_shuffle.py` | 169 | 动作乱序一致性测试 |
| `test_v42_actual_readback_admission.py` | 90 | 实际回读准入测试 |
| `test_v42_cross_fitted_calibration.py` | 64 | 交叉拟合校准测试 |
| `test_v42_derived_supervision.py` | 135 | 派生监督信号测试 |
| `test_v42_dwf_no_dwf_group_leakage.py` | 73 | DWF 组间无泄漏测试 |
| `test_v42_dwf_source_admission.py` | 140 | DWF 数据源准入测试 |
| `test_v42_evaluation_availability.py` | 57 | 评估可用性测试 |
| `test_v42_event_balanced_sampler.py` | 46 | 事件均衡采样器测试 |
| `test_v42_event_usage_ledger.py` | 175 | 事件使用台账测试 |
| `test_v42_final_dataset_admission_gate.py` | 155 | 最终数据集准入门控测试 |
| `test_v42_fold_local_oversampling.py` | 58 | 折内局部过采样测试 |
| `test_v42_four_branch_same_state.py` | 99 | 四分支同状态测试 |
| `test_v42_fresh_evaluation_split.py` | 78 | 新鲜评估集切分测试 |
| `test_v42_grouped_cv.py` | 81 | 分组交叉验证测试 |
| `test_v42_hard_negative_sampling.py` | 68 | 困难负样本采样测试 |
| `test_v42_head_activation_and_optimizer.py` | 232 | 头部激活与优化器测试 |
| `test_v42_nested_cv_no_leakage.py` | 53 | 嵌套 CV 无泄漏测试 |
| `test_v42_no_sentinel_fallback_for_pfv.py` | 70 | PFV 无哨兵回退测试 |
| `test_v42_pairwise_ranking.py` | 65 | 成对排序测试 |
| `test_v42_pfv_core8_oracle.py` | 197 | PFV 核心 8 节点 Oracle 测试 |
| `test_v42_physical_sample_dedup.py` | 101 | 物理样本去重测试 |
| `test_v42_physics_gradient_and_perturbation.py` | 165 | 物理梯度与扰动测试 |
| `test_v42_physics_units.py` | 176 | 物理单位一致性测试 |
| `test_v42_pipeline_dependencies.py` | 94 | 流水线依赖关系测试 |
| `test_v42_priority_contract_core8.py` | — | 优先级契约核心 8 节点测试 |
| `test_v42_priority_missing_fail_closed.py` | — | 优先级缺失 fail-closed 测试 |
| `test_v42_priority_sentinel_separation.py` | — | 优先级哨兵分离测试 |
| `test_v42_reference_dedup.py` | — | 参照去重测试 |
| `test_v42_reserved_evaluation_isolation.py` | — | 保留评估集隔离测试 |
| `test_v42_target_keys_and_ranking.py` | — | 目标键与排序测试 |
| `test_v42_tfv_peak_recomputation.py` | — | TFV 峰值重计算测试 |
| `test_v42_train_grouped_gate.py` | — | 训练分组门控测试 |
| `test_v42_training_summary.py` | — | 训练摘要测试 |
| `test_v42_twin_branch_and_aliasing.py` | — | 孪生分支与别名测试 |
| `test_v42_unified_development_pool.py` | — | 统一开发数据池测试 |
| `test_v42_v4_round_lineage_dedup.py` | — | V4 轮次谱系去重测试 |

---

## 统计

| 指标 | 数值 |
|------|------|
| **sewerrtc/ 总 .py 文件数** | 265 |
| **sewerrtc/ 总 .py 行数** | ~90,066 |
| **test_v42_* 测试文件数** | 37 |
| **docs/contracts/ 合同文件数** | 30 |
| **configs/ YAML 配置文件数** | 48 |

### 各目录文件数与行数

| 目录 | .py 文件数 | 总行数 |
|------|-----------|--------|
| `sewerrtc/v4/` | 88 | ~46,970 |
| `sewerrtc/state/` | 27 | ~7,820 |
| `sewerrtc/control/` | 37 | ~7,094 |
| `sewerrtc/prompt3/` | 12 | ~10,417 |
| `sewerrtc/data/` | 24 | ~4,817 |
| `sewerrtc/simulation/` | 11 | ~4,750 |
| `sewerrtc/models/` | 13 | ~2,143 |
| `sewerrtc/models_v42/` | 13 | ~1,337 |
| `sewerrtc/evaluation/` | 8 | ~1,232 |
| `sewerrtc/experiments/` | 8 | ~1,208 |
| `sewerrtc/io/` | 7 | ~768 |
| `sewerrtc/status/` | 2 | ~656 |
| `sewerrtc/contracts/` | 4 | ~412 |
| `sewerrtc/network/` | 2 | ~195 |
| `sewerrtc/graph/` | 4 | ~167 |
| `sewerrtc/execution/` | 2 | ~63 |
| `sewerrtc/` (根) | 2 | ~16 |
| `sewerrtc/hydraulics/` | 1 | ~1 |
