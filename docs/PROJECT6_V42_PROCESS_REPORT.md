# Project6 V4.2 完整处理流程汇报

> **生成日期**: 2026-07-31  
> **项目路径**: `E:\RTC_sewer\Project6`  
> **Python 环境**: `.venv\Scripts\python.exe`（Python 3.10+）  
> **合同 ID**: `PFV_CORE8_V1`（冻结日期 2026-07-31）

---

## 1. 项目研究目标

### 1.1 核心科学问题

**稀疏传感条件下武汉排水管网水力状态恢复与预测**

在仅有少量传感器观测的条件下，利用图神经网络（GAT）和代理模型重建全网水力状态，并预测不同控制策略在未来 120 分钟内的水力后果，最终实现基于模型预测控制（MPC）的实时排水调度。

### 1.2 输入/输出规格

| 方向 | 内容 | 说明 |
|------|------|------|
| **输入** | 稀疏传感器观测 | 2 个深度哨兵节点 + 有限监测 |
| | 传感器 Mask | 标识哪些节点有观测 |
| | 降雨预测 | 48 种事件（T3–T100），120 min 预测窗口 |
| | 网络拓扑 | 932 节点图结构（INP 模型） |
| | 历史动作 | 36 个受控设施的历史调度序列 |
| | 静态属性 | 节点底高程、最大深度、设施语义 |
| **输出** | 全网节点水深/水头/充满度 | 932 维时间轨迹 |
| | 关键 Storage 状态 | 蓄水量、进出流 |
| | 流量摘要 | PFV / TFV / Peak 三维度 KPI |
| | GAT 不确定性 | 基于 ensemble 的置信度估计 |

### 1.3 核心任务

**预测不同合法 Candidate 未来 120 分钟的水力后果**

给定当前水力状态和降雨预测，对每个候选控制方案（Candidate）推演未来 12 步（每步 10 min）的管网水力轨迹，计算 PFV（优先级防洪量）、TFV（总防洪量）、Peak（峰值洪水流量），为 MPC 控制器提供决策依据。

---

## 2. 四步工作流

### Step 1: 稀疏状态恢复 (GAT)

**目标**: 从稀疏传感器观测恢复全网 932 节点的水力状态

| 属性 | 值 |
|------|-----|
| 代码目录 | `sewerrtc/state/`（25 个 .py 模块） |
| 核心模型 | `sewerrtc/models/gat_reconstructor.py` |
| 输入 | 稀疏观测 + 图拓扑 + 降雨 + 历史帧 |
| 输出 | 全网节点水深/水头/充满度估计 |

**关键文件**:

| 文件 | 功能 |
|------|------|
| `sewerrtc/state/gat_robustness.py` | GAT 鲁棒性验证（997 行） |
| `sewerrtc/state/state_clone_equivalence.py` | 状态克隆等价性验证（903 行） |
| `sewerrtc/state/gat_audit.py` | GAT 审计框架（618 行） |
| `sewerrtc/state/gat_selection.py` | GAT 模型选择（235 行） |
| `sewerrtc/state/augmented_state.py` | 增强状态特征构建（161 行） |
| `sewerrtc/state/gat_holdout_trajectory.py` | 留出轨迹验证（429 行） |
| `sewerrtc/state/hotstart_acceleration.py` | 热启动加速（672 行） |
| `sewerrtc/state/state_input_manifest.py` | 状态输入清单（517 行） |

### Step 2: 多 Reference 水力代理模型

**目标**: 共享动力模型推演四种分支的未来水力轨迹

| 属性 | 值 |
|------|-----|
| 代码目录 | `sewerrtc/v4/models_v42/`（13 个 .py 模块） |
| 训练器 | `sewerrtc/v4/v42_trainer.py`（1237 行） |
| 验证器 | `sewerrtc/v4/v42_full_verification.py`（847 行） |

**共享动力模型**:

$$Y_b = F_\theta(X_{t-60:t},\; R_{t:t+120},\; U_b,\; G)$$

- $X_{t-60:t}$: 13 帧历史状态（每帧 10 min，覆盖过去 120 min）
- $R_{t:t+120}$: 未来 120 min 降雨预测
- $U_b$: 分支 $b$ 的动作序列（36 维 × 12 步）
- $G$: 管网图拓扑（932 节点）

**四分支推演**:

| 分支 | 角色 | 基线用途 |
|------|------|----------|
| **Candidate** | 候选控制方案 | MPC 搜索空间 |
| **No-control** | 全开（all-open） | 诊断参照 |
| **Dynamic Internal** | SWMM 内置规则 | PFV 安全约束基线 |
| **Hold Previous** | 保持当前动作 | 在线安全参照 |

**预测优先级**: 先预测未来水力过程（节点水深、flooding rate、Storage），然后派生 PFV / TFV / Peak 指标。

**关键文件**:

| 文件 | 功能 |
|------|------|
| `sewerrtc/v4/models_v42/counterfactual_twin_dynamics.py` | 反事实孪生动力学（235 行） |
| `sewerrtc/v4/models_v42/graph_state_encoder.py` | 图状态编码器（108 行） |
| `sewerrtc/v4/models_v42/trajectory_losses.py` | 轨迹损失函数（118 行） |
| `sewerrtc/v4/models_v42/physics_losses.py` | 物理约束损失（151 行） |
| `sewerrtc/v4/models_v42/water_balance_backbone.py` | 水量平衡骨干网（96 行） |
| `sewerrtc/v4/models_v42/ranking_losses.py` | 排序损失函数（145 行） |
| `sewerrtc/v4/models_v42/uncertainty.py` | 不确定性估计（147 行） |
| `sewerrtc/v4/models_v42/local_priority_decoder.py` | 局部优先级解码器（113 行） |
| `sewerrtc/v4/models_v42/global_system_decoder.py` | 全局系统解码器（103 行） |
| `sewerrtc/v4/models_v42/actuator_action_encoder.py` | 执行器动作编码（105 行） |
| `sewerrtc/v4/models_v42/rainfall_encoder.py` | 降雨编码器（48 行） |

### Step 3: PFV-first 滚动 MPC

**目标**: 在安全约束下最小化 ΔTFV

| 属性 | 值 |
|------|-----|
| 代码目录 | `sewerrtc/control/`（37 个 .py 模块） |
| 控制步长 | 10 min（600 s） |
| 预测窗口 | 120 min（12 步） |

**决策逻辑**:

1. **安全筛选**: 先筛选安全 Candidate（PFV 安全、Peak 安全、约束满足）
2. **目标优化**: 在安全集中最小化 ΔTFV（相对 Dynamic Internal 基线）
3. **Fallback**: 安全集为空时使用冻结 Fallback 策略

**关键文件**:

| 文件 | 功能 |
|------|------|
| `sewerrtc/control/generic_gat_mpc.py` | 通用 GAT-MPC 控制器（873 行） |
| `sewerrtc/control/v4_candidate_generator.py` | V4 稀疏候选生成器（5 家族） |
| `sewerrtc/control/dual_reference_v4.py` | V4 双参考控制策略（330 行） |
| `sewerrtc/control/pfvfirst_dualfallback.py` | PFV-first 双回退控制器（205 行） |
| `sewerrtc/control/safety_guards.py` | 安全守卫 |
| `sewerrtc/control/mpc_controller.py` | MPC 控制器核心（882 行） |
| `sewerrtc/control/horizon_rollout.py` | 滚动展开（393 行） |
| `sewerrtc/control/horizon_objective.py` | 目标函数（231 行） |
| `sewerrtc/control/horizon_action_features.py` | 动作特征（418 行） |
| `sewerrtc/control/candidate_generator.py` | 基础候选生成器（389 行） |

### Step 4: 闭环与盲测评价

**目标**: 逐步验证控制策略的科学有效性

| 属性 | 值 |
|------|-----|
| 代码目录 | `sewerrtc/evaluation/`（7 个 .py 模块） |
| 验证器 | `sewerrtc/v4/v42_full_verification.py` |

**评价阶段**（依次推进）:

| 阶段 | 名称 | 说明 |
|------|------|------|
| A | True-state 验证 | 使用真实状态评估代理模型精度 |
| B | Exact SWMM 闭环 | 代理模型 + SWMM 权威仿真闭环 |
| C | Surrogate 闭环 | 纯代理模型闭环（快速验证） |
| D | GAT 接入 | GAT 状态恢复接入闭环 |
| E | Policy Lock | 策略冻结 + SHA 校验 |
| F | Challenge | ≥8 全新事件挑战 |
| G | Formal Blind | ≥24 全新事件正式盲测 |

**评价模块**:

| 文件 | 功能 |
|------|------|
| `sewerrtc/evaluation/evaluate_closed_loop.py` | 闭环评价框架（145 行） |
| `sewerrtc/evaluation/kpi_contract.py` | KPI 合同验证（146 行） |
| `sewerrtc/evaluation/policy_sets.py` | 策略集管理（87 行） |
| `sewerrtc/evaluation/project5_formal_gate.py` | 正式门禁（373 行） |
| `sewerrtc/evaluation/risk_stratified.py` | 风险分层评估（384 行） |
| `sewerrtc/evaluation/significance.py` | 统计显著性检验（55 行） |
| `sewerrtc/evaluation/smoke_functionality_gate.py` | 功能烟雾测试（181 行） |

---

## 3. V4.2 处理时间线

### 3.1 V4.1 管线执行与 scientific_fail

- V4.1 管线完整执行，Phase-1（消融 + 选择 + 确定性五种子训练）全部通过
- Calibration 100/100 完成，四分支 SWMM 生成
- 最终 Gate 判定为 **scientific_fail**（exit=5）：KPI 损失全为零，CV R² 为负
- 这是合法科学结果，冻结负结果，禁止用 Locked 调参，开 V4.2

### 3.2 三个关键 Bug 发现与修复（Task 15）

V4.1 的 scientific_fail 揭示了三个关键 Bug：

1. **KPI 损失恒为 0.0**: `trajectory_losses.py` 中 target key 使用 `pfv_gt` 而非 `pfv_delta`，导致损失计算完全失效
2. **物理损失未激活**: `v42_trainer.py` 中 `loss_b` / `loss_c` 仅包含 `non_negative`，缺少完整的物理约束
3. **PFV 方向反转**: Bug 1 的直接后果，修复 key 后自然解决

### 3.3 V4.2 数据池重建（Tasks 16–21）

| Task | 内容 | 状态 |
|------|------|------|
| Task 16 | 事件账本（Event Usage Ledger） | ✅ 完成 |
| Task 17 | 统一数据池（Unified Pool） | ✅ 完成 |
| Task 18 | 派生监督（Derived Supervision） | ✅ 完成 |
| Task 19 | 嵌套 CV + 采样器 + Gate | ✅ 完成 |
| Task 20 | Pipeline 注册（216 阶段） | ✅ 完成 |
| Task 21 | 测试文件（37 个 V4.2 测试） | ✅ 完成 |
| — | Fresh Evaluation 分割 | ✅ 完成 |

### 3.4 Priority 合同修复

**发现**: V4.2 代码中 4 个文件使用 2 个 Sentinel 节点而非 8 个 PFV Core 节点

**修复**:
- 创建 `sewerrtc/v4/v42_priority_contract.py` — 统一节点合同管理
- 修复 4 个文件使其正确导入 8 个 PFV Core 节点
- Config 修复：确保优先级节点列表一致
- 合同 `PFV_CORE8_V1` 冻结，SHA-256 校验

### 3.5 数据审计体系

- **15+ 审计模块**: PFV Oracle、TFV/Peak Oracle、DWF 完整性、去重、历史帧重建、合同冲突等
- **12 个最终数据集**: 构建于 `data/v42_final_unified/`
- **673 行数据审核文档**: `docs/PROJECT6_V42_DATA_AUDIT_REPORT.md`

### 3.6 快速可学性验证

运行 `quick_learnability_check`，四阶段训练（A→B→C→D）结果：

| 阶段 | 内容 | 关键发现 |
|------|------|----------|
| A | 基础轨迹回归 | 损失正常下降（1.88 → 0.78） |
| B | 物理约束 + 动力学 | 物理损失激活，mass_balance 收敛 |
| C | KPI 头 + 排序 | PFV/TFV/Peak 头加入 |
| D | 全目标联合训练 | **TFV 尺度灾难暴露** |

**TFV 尺度灾难**:
- Stage D 最终: `train_tfv_kpi ≈ 38,916`，`val_tfv_kpi ≈ 36,834`
- TFV loss 占总 loss 的 **99.9%**（train_loss ≈ 38,955 中 TFV 贡献 ≈ 38,916）
- PFV R² = −0.48，TFV R² = −0.17，均未能学习
- 根因：TFV 汇总全部 932 节点，数值比 PFV 大约 4 个数量级，未归一化

---

## 4. 修复记录

| # | 问题 | 根因 | 修复 | 验证状态 |
|---|------|------|------|----------|
| 1 | KPI 损失恒为 0.0 | `trajectory_losses.py` target key `pfv_gt` 应为 `pfv_delta` | 修复 key 名称 | ✓ 已验证 |
| 2 | 物理损失未激活 | `v42_trainer.py` loss_b/loss_c 仅含 `non_negative` | 扩展为 6 项物理损失（mass_balance, storage_continuity, capacity_bounds, flooding_consistency, shared_init_state, kpi_trajectory_consistency） | ✓ 已验证 |
| 3 | PFV 方向反转 | Bug 1 的直接后果 | Bug 1 修复后自然解决 | ✓ 已验证 |
| 4 | Priority/Sentinel 混用 | 4 个 V4.2 文件读取 2 个 sentinel 节点而非 8 个 PFV Core 节点 | 创建 `v42_priority_contract.py` + 修复 4 文件 | ✓ 已验证 |
| 5 | `priority_node_indices` 索引越界 | 张量放入 per-sample dataset 导致维度不匹配 | 移至 `shared_tensors` | ✓ 已验证 |
| 6 | `TrajectoryLosses` KeyError | Stage B 基础模型不输出 `pfv_delta` | 增加 pred 键存在性检查 | ✓ 已验证 |
| 7 | TFV 尺度灾难 | TFV 汇总全部 932 节点，数值比 PFV 大约 4 个数量级 | 待修复（归一化） | 🔄 进行中 |

---

## 5. 当前代码状态

### 5.1 编译与测试

| 指标 | 值 |
|------|-----|
| `sewerrtc/v4/` 编译状态 | ✅ 全部通过 |
| V4.2 测试文件 | 37 个 |
| V4.2 测试函数 | 257 个 |
| V4 总测试文件 | 113 个 |
| V4 总测试函数 | 720 个 |

### 5.2 Pipeline 规模

| 指标 | 值 |
|------|-----|
| 总阶段数 | 216 |
| Handler 函数 | 46+ |
| Pipeline 文件 | `sewerrtc/v4/pipeline.py`（4070 行） |

### 5.3 数据资产

| 指标 | 值 |
|------|-----|
| 最终任务数据集 | 12 个已构建 |
| 总样本数 | 1200 |
| 唯一物理样本 | 422 |
| 降雨事件覆盖 | 48 个（T3–T100） |
| 排序对 | 2400 |
| 合同 ID | `PFV_CORE8_V1`（冻结） |

### 5.4 已知问题

| 问题 | 严重程度 | 状态 |
|------|----------|------|
| TFV 尺度待归一化 | 🔴 高 | 阻塞可学性 |
| 25+ 处硬编码路径待清理 | 🟡 中 | 阻塞发布 |
| PFV Oracle 审计未通过 | 🟡 中 | 需排查标签一致性 |
| 哨兵节点溯源未完成 | 🟢 低 | 不影响核心指标 |

---

## 6. 下一步计划

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | TFV/PFV/Peak 目标归一化 | 解决 TFV 尺度灾难，使三个 KPI 头均衡学习 |
| **P0** | 重新训练 → 验证可学性 | 归一化后重跑 quick_learnability_check，确认 R² > 0 |
| **P1** | 硬编码路径清理 | 清理 25+ 处硬编码路径，统一使用 contract 模块 |
| **P1** | GitHub 上传准备 | 代码清理后准备公开仓库 |
| **P2** | 完整 5×5 嵌套 CV 训练 | 使用分组交叉验证评估泛化性 |
| **P2** | 四步工作流端到端验证 | Step 1–4 全链路集成测试 |
| **P3** | Formal Blind 评价 | ≥24 全新事件正式盲测 |

---

## 7. 核心代码文件清单

### Step 1: 稀疏状态恢复

| 文件路径 | 功能 |
|----------|------|
| `sewerrtc/state/__init__.py` | 模块入口 |
| `sewerrtc/state/gat_robustness.py` | GAT 鲁棒性验证 |
| `sewerrtc/state/state_clone_equivalence.py` | 状态克隆等价性 |
| `sewerrtc/state/gat_audit.py` | GAT 审计框架 |
| `sewerrtc/state/gat_selection.py` | GAT 模型选择 |
| `sewerrtc/state/augmented_state.py` | 增强状态特征 |
| `sewerrtc/state/gat_holdout_trajectory.py` | 留出轨迹验证 |
| `sewerrtc/state/gat_independent_validation.py` | 独立验证 |
| `sewerrtc/state/gat_registry.py` | GAT 注册表 |
| `sewerrtc/state/gat_robustness_gate.py` | 鲁棒性门禁 |
| `sewerrtc/state/gat_validation_provenance.py` | 验证溯源 |
| `sewerrtc/state/gat_event_catalog.py` | 事件目录 |
| `sewerrtc/state/gat_event_leakage.py` | 事件泄漏检测 |
| `sewerrtc/state/gat_holdout_eligibility.py` | 留出资格 |
| `sewerrtc/state/hotstart_acceleration.py` | 热启动加速 |
| `sewerrtc/state/interface_cache.py` | 接口缓存 |
| `sewerrtc/state/local_flow_features.py` | 局部流量特征 |
| `sewerrtc/state/runtime_state_features.py` | 运行时状态特征 |
| `sewerrtc/state/same_state_replay.py` | 同状态重放 |
| `sewerrtc/state/state_clone_contract.py` | 状态克隆合同 |
| `sewerrtc/state/state_contract.py` | 状态合同 |
| `sewerrtc/state/state_input_manifest.py` | 状态输入清单 |
| `sewerrtc/state/prompt2_completion_gate.py` | Prompt2 完成门禁 |
| `sewerrtc/state/prompt2_gat_readiness.py` | GAT 就绪检查 |
| `sewerrtc/models/gat_reconstructor.py` | GAT 重建器核心 |

### Step 2: 多 Reference 水力代理模型

| 文件路径 | 功能 |
|----------|------|
| `sewerrtc/v4/models_v42/__init__.py` | 模块入口 |
| `sewerrtc/v4/models_v42/counterfactual_twin_dynamics.py` | 反事实孪生动力学 |
| `sewerrtc/v4/models_v42/graph_state_encoder.py` | 图状态编码器 |
| `sewerrtc/v4/models_v42/water_balance_backbone.py` | 水量平衡骨干网 |
| `sewerrtc/v4/models_v42/local_priority_decoder.py` | 局部优先级解码器 |
| `sewerrtc/v4/models_v42/global_system_decoder.py` | 全局系统解码器 |
| `sewerrtc/v4/models_v42/actuator_action_encoder.py` | 执行器动作编码 |
| `sewerrtc/v4/models_v42/rainfall_encoder.py` | 降雨编码器 |
| `sewerrtc/v4/models_v42/trajectory_losses.py` | 轨迹损失 |
| `sewerrtc/v4/models_v42/physics_losses.py` | 物理约束损失 |
| `sewerrtc/v4/models_v42/ranking_losses.py` | 排序损失 |
| `sewerrtc/v4/models_v42/uncertainty.py` | 不确定性估计 |
| `sewerrtc/v4/models_v42/trainer.py` | 训练器 |
| `sewerrtc/v4/v42_trainer.py` | V4.2 训练管线 |
| `sewerrtc/v4/v42_full_verification.py` | V4.2 完整验证 |
| `sewerrtc/v4/v42_priority_contract.py` | 优先级节点合同 |

### Step 3: PFV-first 滚动 MPC

| 文件路径 | 功能 |
|----------|------|
| `sewerrtc/control/generic_gat_mpc.py` | 通用 GAT-MPC 控制器 |
| `sewerrtc/control/v4_candidate_generator.py` | V4 稀疏候选生成器 |
| `sewerrtc/control/dual_reference_v4.py` | V4 双参考策略 |
| `sewerrtc/control/pfvfirst_dualfallback.py` | PFV-first 双回退 |
| `sewerrtc/control/mpc_controller.py` | MPC 控制器核心 |
| `sewerrtc/control/candidate_generator.py` | 基础候选生成器 |
| `sewerrtc/control/horizon_rollout.py` | 滚动展开 |
| `sewerrtc/control/horizon_objective.py` | 目标函数 |
| `sewerrtc/control/horizon_action_features.py` | 动作特征 |
| `sewerrtc/control/action_sequence_generator.py` | 动作序列生成 |
| `sewerrtc/control/safety_guards.py` | 安全守卫 |
| `sewerrtc/control/hierarchical_core26_residual10.py` | 分层控制 |
| `sewerrtc/control/actuator_scope.py` | 执行器作用域 |
| `sewerrtc/control/canonical_action_order.py` | 规范动作排序 |
| `sewerrtc/control/event_pfv_budget.py` | 事件 PFV 预算 |
| `sewerrtc/control/fallback_contract.py` | 回退合同 |
| `sewerrtc/control/fallback_selector.py` | 回退选择器 |
| `sewerrtc/control/formal_policy.py` | 形式化策略 |
| `sewerrtc/control/internal_fallback.py` | 内部回退 |
| `sewerrtc/control/no_control_reference_predictor.py` | 无控制参照预测 |

### Step 4: 闭环与盲测评价

| 文件路径 | 功能 |
|----------|------|
| `sewerrtc/evaluation/evaluate_closed_loop.py` | 闭环评价框架 |
| `sewerrtc/evaluation/kpi_contract.py` | KPI 合同验证 |
| `sewerrtc/evaluation/policy_sets.py` | 策略集管理 |
| `sewerrtc/evaluation/project5_formal_gate.py` | 正式门禁 |
| `sewerrtc/evaluation/risk_stratified.py` | 风险分层评估 |
| `sewerrtc/evaluation/significance.py` | 统计显著性 |
| `sewerrtc/evaluation/smoke_functionality_gate.py` | 功能烟雾测试 |

### Pipeline 与基础设施

| 文件路径 | 功能 |
|----------|------|
| `sewerrtc/v4/pipeline.py` | 主流水线（216 阶段，4070 行） |
| `sewerrtc/v4/pipeline_v4_compact.py` | Phase-1 handler |
| `sewerrtc/v4/pipeline_v4_compact_eval.py` | Phase-2 handler |
| `sewerrtc/v4/pipeline_v4_closed_loop.py` | 闭环门 handler |
| `sewerrtc/v4/v4_compact_model_ops.py` | 模型操作 |
| `sewerrtc/v4/v4_compact_eval_ops.py` | 评价操作 |
| `sewerrtc/v4/online_v4_compact.py` | 在线特征适配器 |
| `sewerrtc/v4/closed_loop.py` | 闭环逻辑 |
| `scripts/project6_v4_final.py` | V4 最终驱动脚本 |

### 配置文件

| 文件路径 | 功能 |
|----------|------|
| `configs/wuhan_project6_dual_reference_v4.yaml` | V4 双参考主配置 |
| `configs/wuhan_project6_v4_final.yaml` | V4 Final 配置 |
| `configs/wuhan_project6_v4_gate5r.yaml` | V4 Gate 5R 配置 |

### 测试文件（V4.2 专项，37 个）

| 文件路径 | 测试内容 |
|----------|----------|
| `tests/test_v42_13frame_real_rebuild.py` | 13 帧真实重建 |
| `tests/test_v42_action_shuffle.py` | 动作洗牌 |
| `tests/test_v42_actual_readback_admission.py` | 实际回读准入 |
| `tests/test_v42_cross_fitted_calibration.py` | 交叉拟合校准 |
| `tests/test_v42_derived_supervision.py` | 派生监督 |
| `tests/test_v42_dwf_no_dwf_group_leakage.py` | DWF 泄漏检测 |
| `tests/test_v42_dwf_source_admission.py` | DWF 源准入 |
| `tests/test_v42_evaluation_availability.py` | 评价可用性 |
| `tests/test_v42_event_balanced_sampler.py` | 事件平衡采样 |
| `tests/test_v42_event_usage_ledger.py` | 事件使用账本 |
| `tests/test_v42_final_dataset_admission_gate.py` | 最终数据集准入 |
| `tests/test_v42_fold_local_oversampling.py` | 折内过采样 |
| `tests/test_v42_four_branch_same_state.py` | 四分支同状态 |
| `tests/test_v42_fresh_evaluation_split.py` | 新鲜评价分割 |
| `tests/test_v42_grouped_cv.py` | 分组 CV |
| `tests/test_v42_hard_negative_sampling.py` | 硬负采样 |
| `tests/test_v42_head_activation_and_optimizer.py` | 头激活与优化器 |
| `tests/test_v42_nested_cv_no_leakage.py` | 嵌套 CV 无泄漏 |
| `tests/test_v42_no_sentinel_fallback_for_pfv.py` | PFV 禁止 Sentinel 回退 |
| `tests/test_v42_pairwise_ranking.py` | 成对排序 |
| `tests/test_v42_pfv_core8_oracle.py` | PFV Core8 Oracle |
| `tests/test_v42_physical_sample_dedup.py` | 物理样本去重 |
| `tests/test_v42_physics_gradient_and_perturbation.py` | 物理梯度与扰动 |
| `tests/test_v42_physics_units.py` | 物理单位 |
| `tests/test_v42_pipeline_dependencies.py` | Pipeline 依赖 |

### 数据资产

| 文件路径 | 功能 |
|----------|------|
| `data/v42_final_unified/pfv_constraint_core8.parquet` | PFV 安全约束训练 |
| `data/v42_final_unified/tfv_objective.parquet` | TFV 优化目标 |
| `data/v42_final_unified/peak_constraint.parquet` | Peak 约束 |
| `data/v42_final_unified/actuator_effect.parquet` | 执行器效应 |
| `data/v42_final_unified/dynamics_pretrain.parquet` | 动力学预训练 |
| `data/v42_final_unified/within_state_ranking_pairs.parquet` | 状态内排序对 |
| `data/v42_final_unified/sample_lineage.parquet` | 样本溯源 |
| `data/project5_design/priority_pfv_core_nodes.txt` | 8 PFV Core 节点定义 |
| `data/project6_v3_sentinel_nodes.txt` | 哨兵节点定义 |
| `data/project6_v3_facility_semantics_36.csv` | 36 设施语义表 |
| `data/wuhan_v8_storage_retrofit.inp` | 唯一物理网络模型 |

---

*报告结束。Project6 V4.2 团队，2026-07-31。*
