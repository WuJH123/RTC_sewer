# Project6 V4.2 数据审核报告

> **文档状态**: 正式审核报告  
> **生成日期**: 2026-07-31  
> **合同 ID**: `PFV_CORE8_V1`  
> **冻结日期**: 2026-07-31  

---

## 1. 概述

### 1.1 报告目的

本报告为 Project6 V4.2 最终数据池的官方审核文档，旨在：

- 记录冻结数据池的完整结构与统计特征
- 明确指标计算公式与 Delta 约定
- 提供数据读取方式与文件清单
- 分析因果结构与研究目标对齐度
- 列出已知问题与限制

### 1.2 关键标识

| 项目 | 值 |
|------|-----|
| 合同 ID | `PFV_CORE8_V1` |
| 优先级节点文件 SHA-256 | `915908de0c3205ee143d4187710ef6680cc49d1839161995d165144df83313a6` |
| 哨兵节点文件 SHA-256 | `06816141dc99bc3c9ea79d6f261a197a8705faedd9e428e9169a599ce088cd1f` |
| 敏感区文件 SHA-256 | `17ad0a51656ace056a9edcbb723a62616fc9b341a84afc87af102ae006a7f2b4` |
| 节点总数 (INP schema) | 932 |
| 受控设施数 | 36 |
| 控制步长 dt_sec | 600 s (10 min) |
| 历史帧数 | 13 |
| 预测步数 | 12 (覆盖 120 min) |

### 1.3 冻结 8 节点合同摘要

合同 `PFV_CORE8_V1` 冻结了三组节点集合：

| 集合 | 数量 | 角色 | 来源文件 |
|------|------|------|----------|
| PFV Core 节点 | 8 | 优先级防洪指标计算 | `data/project5_design/priority_pfv_core_nodes.txt` |
| 深度哨兵节点 | 2 | 仅监测特征 | `data/project6_v3_sentinel_nodes.txt` |
| 敏感区节点 | 11 | 仅辅助分析 | `data/project2_design/priority_zone_nodes.csv` |

所有 V4.2 代码必须通过 `sewerrtc/v4/v42_priority_contract.py` 导入节点列表，禁止硬编码。

---

## 2. 数据读取方式

### 2.1 数据流总览

```
SWMM INP 模型 (932 节点, 36 设施)
       │
       ▼
降雨库生成 (48 事件 × 多检查点)
       │
       ▼
分支轨迹 CSV (candidate / no_control / dynamic_internal / hold_previous)
       │
       ▼
轨迹 Manifest (Parquet)
       │
       ▼
张量缓存构建 (13 历史帧 + 12 预测步)
       │
       ▼
V4.2 统一数据池 (1200 样本)
       │
       ├── pfv_constraint_core8.parquet   → PFV 安全约束训练
       ├── tfv_objective.parquet          → TFV 优化目标训练
       ├── peak_constraint.parquet        → Peak 约束训练
       ├── actuator_effect.parquet        → 执行器效应训练
       ├── dynamics_pretrain.parquet      → 动力学预训练
       ├── within_state_ranking_pairs.parquet → 状态内排序对
       ├── sample_lineage.parquet         → 样本溯源
       └── 其他辅助文件
```

### 2.2 文件位置与格式

所有最终数据集位于 `data/v42_final_unified/`：

| 文件名 | 格式 | 样本数 | 用途 |
|--------|------|--------|------|
| `pfv_constraint_core8.parquet` | Parquet | 1200 | PFV Core8 安全约束 |
| `tfv_objective.parquet` | Parquet | 1200 | TFV 优化目标 |
| `peak_constraint.parquet` | Parquet | 1200 | Peak 防洪约束 |
| `actuator_effect.parquet` | Parquet | 1200 | 执行器效应学习 |
| `dynamics_pretrain.parquet` | Parquet | 1200 | 动力学预训练 |
| `within_state_ranking_pairs.parquet` | Parquet | 2400 | 状态内排序对 |
| `sample_lineage.parquet` | Parquet | 1200 | 完整溯源信息 |
| `target_no_dwf_full_supervision.parquet` | Parquet | 0 | 目标域无DWF全监督 (预留) |
| `source_dwf_full_supervision.parquet` | Parquet | 0 | 源域DWF全监督 (预留) |
| `consumed_development.parquet` | Parquet | 0 | 已消耗开发样本 (预留) |
| `reserved_evaluation_manifest.csv` | CSV | 0 | 预留评估集 |
| `rejected_samples.csv` | CSV | 0 | 拒绝样本 |

审计文件位于 `audits/v42_final_pool/`：

| 文件名 | 内容 |
|--------|------|
| `pfv_oracle_audit.json` | PFV 独立审计结果 |
| `tfv_peak_oracle_audit.json` | TFV/Peak 独立审计结果 |
| `sample_classification_summary.json` | 样本分类统计 |
| `dwf_audit_summary.json` | DWF 审计摘要 |
| `deduplication_audit.json` | 去重审计 |
| `history_rebuild_audit.json` | 历史帧重建审计 |
| `pfv_label_mismatches.csv` | PFV 标签不匹配详情 |
| `*_contract_conflicts.csv` | 各类合同冲突 (均为空) |

### 2.3 数据加载方式

**数据来源**: 所有 1200 样本均来自 `round0`（V4 3000 轮次的首轮生成），涵盖 48 个降雨事件，每事件 25 个检查点样本。

**加载路径**:
- 轨迹数据: 从 `outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/` 下的 manifest 加载
- 统一数据集: 直接读取 `data/v42_final_unified/*.parquet`
- 溯源信息: 通过 `sample_lineage.parquet` 关联原始 manifest

**DWF 审计完整性** (全部 1200/1200 通过):
- `dwf_node_inflow_present`: 1200 ✓
- `h120_dwf_sequence_readable`: 1200 ✓
- `branches_share_dwf`: 1200 ✓
- `model_input_dwf_present`: 1200 ✓
- `actual_actions_present`: 1200 ✓
- `hydraulic_trajectories_complete`: 1200 ✓
- `labels_recomputable`: 1200 ✓

### 2.4 关键 Parquet 文件列 Schema

**主数据集列** (以 `pfv_constraint_core8.parquet` 为例):

| 列名 | 说明 |
|------|------|
| `sample_id` | 样本唯一标识 |
| `sample_idx` | 样本序号 |
| `event_id` | 降雨事件 ID |
| `checkpoint_id` | 检查点 ID |
| `state_key` | 状态键 (`event_id::checkpoint_id`) |
| `split` | 数据划分 |
| `source_round` | 来源轮次 |
| `grade` | 质量等级 |
| `priority_contract_id` | 合同 ID (`PFV_CORE8_V1`) |
| `pfv_delta` | PFV Delta 值 (m³) |
| `pfv_safe_label` | PFV 安全标签 |
| `trajectory_candidate_pfv_core8` | 候选轨迹在 8 节点的 PFV |
| `trajectory_no_control_pfv_core8` | 无控制轨迹在 8 节点的 PFV |
| `priority_node_ids` | 优先级节点 ID 列表 |

**溯源列** (`sample_lineage.parquet` 额外列):

| 列名 | 说明 |
|------|------|
| `source_manifest` | 来源 manifest 路径 |
| `original_case_id` | 原始案例 ID |
| `rainfall_fingerprint` | 降雨指纹 |
| `prefix_state_hash` | 前缀状态哈希 |
| `candidate_id` / `candidate_family` | 候选标识 |
| `candidate_action_sha` / `ref_nc_action_sha` / `ref_di_action_sha` | 动作序列 SHA |
| `candidate_trajectory_sha` / `ref_nc_trajectory_sha` / `ref_di_trajectory_sha` | 轨迹 SHA |
| `k_actual` / `k_target` | 实际/目标控制步数 |

---

## 3. 指标计算方式

### 3.1 PFV — 优先级防洪量 (Priority Flooding Volume)

**定义**: 在冻结的 8 个优先级节点上，对洪水流量进行时间积分。

**公式**:

$$\text{PFV} = \sum_{t} \sum_{i \in \text{Core8}} \max\bigl(q_{\text{flood}}^{(i)}(t),\, 0\bigr) \cdot \Delta t$$

- $q_{\text{flood}}^{(i)}(t)$: 节点 $i$ 在时刻 $t$ 的洪水流量 (m³/s)
- $\Delta t = 600$ s (控制步长)
- 求和范围: 8 个优先级节点 × 所有时间步
- **单位**: m³

**Delta 约定**:

$$\Delta\text{PFV} = \text{PFV}_{\text{candidate}} - \text{PFV}_{\text{no\_control}}$$

- 正值表示候选方案比无控制方案 flooding 更多（更差）
- 负值表示候选方案减少了 flooding（更好）
- 安全标签: `pfv_safe_label = (ΔPFV ≤ 0)`

### 3.2 TFV — 总防洪量 (Total Flooding Volume)

**定义**: 在所有 932 个节点上，对洪水流量进行时间积分。

**公式**:

$$\text{TFV} = \sum_{t} \sum_{j \in \text{AllNodes}} \max\bigl(q_{\text{flood}}^{(j)}(t),\, 0\bigr) \cdot \Delta t$$

- 求和范围: 全部 932 个节点 × 所有时间步
- **单位**: m³

**Delta 约定**:

$$\Delta\text{TFV} = \text{TFV}_{\text{candidate}} - \text{TFV}_{\text{dynamic\_internal}}$$

- 基线为 `dynamic_internal`（动态内部基线），而非 `no_control`
- 改善标签: `tfv_improved_label = (ΔTFV ≤ 0)`

### 3.3 Peak — 峰值防洪流量 (Peak Total Flooding Rate)

**定义**: 所有节点总洪水流量在时间维度上的最大值。

**公式**:

$$\text{Peak} = \max_{t} \sum_{j \in \text{AllNodes}} \max\bigl(q_{\text{flood}}^{(j)}(t),\, 0\bigr)$$

- **单位**: m³/s

**Delta 约定**:

$$\Delta\text{Peak} = \max_{t}\bigl(\text{TotalFloodRate}_{\text{candidate}}(t)\bigr) - \max_{t}\bigl(\text{TotalFloodRate}_{\text{DI}}(t)\bigr)$$

> **注意**: Peak Delta 是先分别取各分支的时间最大值，再相减。**不是** `max_t(Candidate - DI)`。

- 非劣标签: `peak_noninferior_label = (ΔPeak ≤ 0)`

### 3.4 时间积分说明

根据 KPI 合同 (`docs/contracts/kpi_contract.json`)：
- 10 min 为控制步长，**非** SWMM 水力积分步长
- KPI 积分必须使用实际时间戳或逐样本 dt 值
- 积分方式: `timestamp_delta_sum_rate_times_dt`
- 缺失值策略: `fail_fast`

---

## 4. 冻结 8 节点集合

### 4.1 PFV Core 8 节点

合同 ID: `PFV_CORE8_V1`，冻结日期: 2026-07-31

| # | 节点 ID | 坐标 (X, Y) | 角色 | 最大深度 (m) | 底高程 (m) |
|---|---------|-------------|------|-------------|-----------|
| 1 | `MSLBZW001` | 527556.24, 384267.86 | extra_hydraulic_bottleneck | 1.90 | 22.10 |
| 2 | `HS1316314` | 528871.68, 389936.46 | near_sensitive_bottleneck | 3.15 | 19.57 |
| 3 | `YS2530050` | 525320.75, 389285.88 | near_sensitive_bottleneck | 1.80 | 18.03 |
| 4 | `HS2529198` | 525181.41, 384371.63 | near_sensitive_bottleneck | 2.55 | 18.88 |
| 5 | `MH0200773` | 522778.68, 384274.98 | extra_hydraulic_bottleneck | 2.10 | 21.48 |
| 6 | `HS1330349` | 527500.61, 389391.21 | near_sensitive_bottleneck | 3.08 | 17.16 |
| 7 | `HS2529139` | 525175.75, 383994.21 | near_sensitive_bottleneck | 2.61 | 18.97 |
| 8 | `HS2529052` | 525163.75, 383583.14 | waterlogging_sensitive | 2.80 | 21.57 |

坐标来源: `data/project2_design/priority_zone_nodes.csv`

### 4.2 深度哨兵节点 (2 个)

| # | 节点 ID | 坐标 (X, Y) | 角色 | 最大深度 (m) | 底高程 (m) |
|---|---------|-------------|------|-------------|-----------|
| 1 | `MH0200770` | 522913.13, 384266.76 | near_sensitive_bottleneck | 2.43 | 21.24 |
| 2 | `HS1355904` | 525438.52, 389017.59 | waterlogging_sensitive | 3.49 | 17.28 |

**角色**: `monitoring_feature_only` — 仅用于监测特征提取，不参与 PFV 计算。

**溯源状态**: `human_resolution_required` — 哨兵节点源自 Project4/Project5 交接上下文，原始选择依据尚未独立验证。

### 4.3 敏感区节点 (11 个)

| # | 节点 ID | 角色 |
|---|---------|------|
| 1 | `YS2530050` | near_sensitive_bottleneck |
| 2 | `HS1316314` | near_sensitive_bottleneck |
| 3 | `HS2529139` | near_sensitive_bottleneck |
| 4 | `HS1330349` | near_sensitive_bottleneck |
| 5 | `MH0200770` | near_sensitive_bottleneck |
| 6 | `HS2529198` | near_sensitive_bottleneck |
| 7 | `HS1355904` | waterlogging_sensitive |
| 8 | `MH0249284` | waterlogging_sensitive |
| 9 | `HS2529052` | waterlogging_sensitive |
| 10 | `MSLBZW001` | extra_hydraulic_bottleneck |
| 11 | `MH0200773` | extra_hydraulic_bottleneck |

**角色**: `secondary_analysis_only` — 仅用于辅助分析。

### 4.4 一致性验证

节点集合在以下位置保持一致使用：

- **合同模块**: `sewerrtc/v4/v42_priority_contract.py` — 导入时 SHA-256 校验 + 数量检查 + 重叠检查
- **数据清单**: `data/v42_final_unified/dataset_manifest.json` — `priority_node_ids` 字段
- **审计合同**: `docs/contracts/PROJECT6_V42_PRIORITY_PFV_CONTRACT.json` — `status: FROZEN`
- **PFV 审计**: `audits/v42_final_pool/pfv_oracle_audit.json` — 使用相同 8 节点列表
- **重叠检查**: PFV Core ∩ Sentinel = ∅ (PASS)

---

## 5. PFV / TFV / Peak 统计分布

### 5.1 数据源说明

以下统计基于 `audits/v42_final_pool/` 中的审计文件以及 `data/v42_final_unified/` 中的数据集清单。

> **注意**: 当前审计目录中未包含 `pool_statistics.json`，以下分布数据来源于语义摘要文件和 Oracle 审计文件。

### 5.2 PFV 统计

**Oracle 审计结果** (`pfv_oracle_audit.json`):

| 指标 | 值 |
|------|-----|
| 审计样本数 | 5 |
| 不匹配数 | 5 |
| 最大绝对误差 | 1413.70 m³ |
| 平均绝对误差 | 672.62 m³ |
| 存储 PFV 范围 | [0.0, 9.47] m³ |
| 重算 PFV 范围 | [-546.70, 1413.70] m³ |
| 审计结果 | **FAIL** |

**语义摘要** (`semantic_source_summary.csv`):
- round0 平均 ΔPFV: **1.20 m³** (接近零，表明候选方案与无控制方案在优先级节点的 flooding 差异较小)

### 5.3 TFV 统计

**Oracle 审计结果** (`tfv_peak_oracle_audit.json`):

| 指标 | 值 |
|------|-----|
| 审计样本数 | 5 |
| TFV 最大绝对误差 | 130,588.81 m³ |
| TFV 平均绝对误差 | 63,615.74 m³ |
| TFV 最大相对误差 | 3.87 |
| TFV 中位相对误差 | 3.48 |
| 存储 TFV 范围 | [-64,439.0, 55,198.38] m³ |
| 重算 TFV 范围 | [21,345.27, 76,152.14] m³ |

**语义摘要**:
- round0 平均 ΔTFV: **25,543.40 m³** (候选方案相比 DI 基线有显著 flooding 增加)

### 5.4 Peak 统计

**Oracle 审计结果** (`tfv_peak_oracle_audit.json`):

| 指标 | 值 |
|------|-----|
| 审计样本数 | 5 |
| Peak 最大绝对误差 | 43.05 m³/s |
| Peak 平均绝对误差 | 33.87 m³/s |
| Peak 最大相对误差 | 1.33 |
| Peak 中位相对误差 | 1.13 |
| 存储 Peak 范围 | [-40.26, 6.74] m³/s |
| 重算 Peak 范围 | [-0.56, 15.73] m³/s |

### 5.5 Delta 分布概览

基于语义摘要的 round0 汇总:

| 指标 | 均值 | 说明 |
|------|------|------|
| ΔPFV | 1.20 m³ | 接近零，优先级节点 flooding 变化微小 |
| ΔTFV | 25,543.40 m³ | 正值，候选方案总 flooding 高于 DI 基线 |

### 5.6 事件分布

48 个降雨事件，每事件 25 个样本，均匀分布:

| 重现期 | 持续时间组合数 | 事件数 |
|--------|---------------|--------|
| T3 | 6 | 6 |
| T7 | 4 | 4 |
| T10 | 6 | 6 |
| T15 | 3 | 3 |
| T20 | 6 | 6 |
| T30 | 8 | 8 |
| T50 | 7 | 7 |
| T75 | 5 | 5 |
| T100 | 3 | 3 |

雨型包括: `chicago_early`, `chicago_center`, `chicago_late`, `double_peak`, `block`

### 5.7 PFV 安全/边界/不安全比例

当前数据集中所有 1200 样本等级为 `TARGET_RECOMPUTABLE`。由于 `pool_statistics.json` 不存在，精确的安全/边界/不安全比例需从 `pfv_constraint_core8.parquet` 中直接统计。

基于 Oracle 审计的 5 样本观察：
- 存储的 PFV delta 值接近零 (范围 [0, 9.47] m³)，表明大多数样本在优先级节点上 flooding 极小
- 重算值出现负 delta (-546.70 m³)，表明部分样本候选方案优于无控制

### 5.8 去重统计

| 指标 | 值 |
|------|-----|
| 总样本数 | 1200 |
| 唯一物理样本数 | 422 |
| 重复组数 | 296 |
| 重复样本数 | 778 |

> 说明: 同一物理状态可能因不同候选策略家族产生多条记录，这些在去重审计中被标记但保留用于训练多样性。

---

## 6. 因果结构分析

### 6.1 因果链描述

本项目的因果链结构为：

```
外生变量 (Exogenous)
  ├── 降雨事件 (重现期 T3-T100, 持续时间 D105-D300, 雨型)
  ├── DWF 水平 (dry weather flow)
  └── 初始水力状态 (前缀状态哈希)
         │
         ▼
    控制器动作 (Action)
    ├── 36 个设施的调度策略
    ├── 动作序列 SHA 可追溯
    └── 候选/NC/DI/HP 四种策略
         │
         ▼
    水力响应 (Hydraulic Response)
    ├── 设施流量变化
    ├── 节点水深变化
    └── 管网水力状态轨迹 (13 帧历史 + 12 步预测)
         │
         ▼
    结果指标 (Outcome)
    ├── PFV (8 优先级节点累积 flooding, m³)
    ├── TFV (全部 932 节点累积 flooding, m³)
    └── Peak (全网最大瞬时总 flooding rate, m³/s)
```

### 6.2 混杂因素

| 混杂因素 | 描述 | 数据中是否可控 |
|----------|------|---------------|
| 降雨强度 | 不同重现期/持续时间/雨型 | ✓ 有 48 种事件覆盖 |
| DWF 水平 | 旱天流量基线 | ✓ 所有样本标记为 SOURCE_DWF_FULL_SUPERVISION |
| 初始蓄水状态 | 事件开始时的管网充水程度 | ✓ 通过 `prefix_state_hash` 标识 |
| 检查点时刻 | 事件内不同时间点的水力状态 | ✓ 通过 `checkpoint_id` 标识 |
| 设施配置 | 36 个可控设施的物理参数 | ✓ 固定 INP 模型 |

### 6.3 模型可学性与限制

**数据支持的能力**:

1. **条件策略学习**: 给定 (降雨, DWF, 初始状态)，学习何种动作序列导致更优的 PFV/TFV/Peak
2. **状态内排序**: 通过 `within_state_ranking_pairs` (2400 对)，同一状态下不同动作的相对优劣
3. **执行器效应**: 通过 `actuator_effect` 数据集，学习动作到指标变化的映射
4. **动力学预测**: 通过 `dynamics_pretrain` (1200 样本)，学习状态转移规律

**数据不支持的能力**:

1. **反事实推理**: 缺少同一状态下的完整反事实轨迹（仅有 4 种参考分支）
2. **长期因果效应**: 仅覆盖 120 min 预测窗口，无法评估更长期的累积效应
3. **跨事件泛化验证**: 所有样本来自 round0，缺少独立轮次的验证数据
4. **未观测混杂**: 管网拓扑变化、设施退化等未建模因素

### 6.4 因果推断 vs 相关性评估

**当前数据更倾向于支持条件相关性学习**，原因：

1. **非随机化分配**: 动作序列由启发式策略生成，非随机实验
2. **有限干预变异**: 4 种参考分支的变异有限，可能无法充分探索动作空间
3. **单一来源**: 所有样本来自 round0，可能存在生成策略的偏差

**增强因果学习的建议**:
- 引入随机化动作探索 (exploration noise)
- 增加多轮次数据 (Round1-5) 以覆盖更多策略空间
- 利用 `within_state_ranking_pairs` 进行配对比较，减少混杂影响

---

## 7. 研究目标对齐

### 7.1 研究目标

> 学习 **为什么** 在不同 DWF / 降雨 / 水力状态下，特定动作会改变设施流量、局部状态和 PFV / TFV / Peak。  
> **不是**: 记忆 "动作 X 通常好/坏"。

### 7.2 当前数据 Schema 对目标的支持评估

| 支持维度 | 评估 | 说明 |
|----------|------|------|
| 状态条件化 | ✓ 强 | 13 帧历史 + 降雨预测提供充分状态上下文 |
| 动作可追溯性 | ✓ 强 | 所有动作序列有 SHA 校验 |
| 多参考对比 | ✓ 强 | 4 种参考分支 (NC/DI/HP/Candidate) |
| 指标多维性 | ✓ 中 | PFV/TFV/Peak 三维度，但 PFV Core8 Oracle 审计未通过 |
| 事件多样性 | ✓ 中 | 48 事件覆盖 T3-T100，但仅 round0 |
| 因果可识别性 | △ 弱 | 非随机化动作，有限干预变异 |
| 状态内对比 | ✓ 强 | 2400 排序对直接支持相对比较学习 |

### 7.3 增强因果学习的建议

**特征工程建议**:

1. **增加降雨强度特征**: 当前有 `rainfall_forecast`，建议显式提取峰值强度、累积量、前期干燥时间等标量特征
2. **增加初始状态摘要**: 从 13 帧历史中提取关键统计量 (管网总蓄水量、关键节点水深、设施利用率)
3. **增加动作摘要特征**: 动作序列的统计描述 (总调节量、调节幅度方差、调节方向变化率)

**标签增强建议**:

1. **逐节点 PFV 分解**: 当前仅有 8 节点汇总 PFV，逐节点分解可帮助模型学习空间归因
2. **时间分布标签**:  flooding 发生的时间分布 (早期/晚期占比)，帮助模型理解时间动态
3. **设施效应标签**: 每个设施对 PFV/TFV 的边际贡献，建立动作→设施→指标的显式映射

**模型输入/输出设计建议**:

1. **输入**: 状态编码 (历史帧) + 降雨预测 + 动作序列 → 隐式条件化
2. **输出**: 多任务学习 — 同时预测 PFV_delta, TFV_delta, Peak_delta + 逐节点 flooding 轨迹
3. **损失函数**: 利用排序对构造 pairwise loss，结合 pointwise 回归损失
4. **正则化**: 鼓励模型关注状态-动作交互项，而非单独的状态或动作主效应

---

## 8. 数据池统计摘要

### 8.1 总体统计

| 指标 | 值 |
|------|-----|
| 总分类样本数 | 1200 |
| 唯一物理样本数 | 422 |
| 唯一事件数 | 48 |
| 唯一降雨指纹数 | 48 |
| 来源轮次 | round0 (100%) |
| 质量等级 | TARGET_RECOMPUTABLE (100%) |
| 受控设施数 | 36 |
| 图节点数 | 932 |

### 8.2 各数据集样本数

| 数据集 | 样本数 | 等级过滤 |
|--------|--------|----------|
| `dynamics_pretrain` | 1200 | DYNAMICS_PRETRAIN_ONLY 及以上 |
| `actuator_effect` | 1200 | ACTUATOR_EFFECT_ONLY 及以上 |
| `pfv_constraint_core8` | 1200 | 所有非 REJECT/RESERVED |
| `tfv_objective` | 1200 | 所有非 REJECT/RESERVED |
| `peak_constraint` | 1200 | 所有非 REJECT/RESERVED |
| `within_state_ranking_pairs` | 2400 | RANKING_ONLY 及以上 |
| `sample_lineage` | 1200 | ALL |
| `target_no_dwf_full_supervision` | 0 | TARGET_FULL_SUPERVISION |
| `source_dwf_full_supervision` | 0 | SOURCE_DWF_FULL_SUPERVISION |
| `consumed_development` | 0 | CONSUMED_DEVELOPMENT |
| `reserved_evaluation_manifest` | 0 | RESERVED_EVALUATION |
| `rejected_samples` | 0 | REJECT |

### 8.3 事件覆盖

48 个降雨事件，每事件恰好 25 个样本:

| 重现期 | 事件数 | 样本数 |
|--------|--------|--------|
| T3 | 6 | 150 |
| T7 | 4 | 100 |
| T10 | 6 | 150 |
| T15 | 3 | 75 |
| T20 | 6 | 150 |
| T30 | 8 | 200 |
| T50 | 7 | 175 |
| T75 | 5 | 125 |
| T100 | 3 | 75 |
| **合计** | **48** | **1200** |

### 8.4 设施覆盖

- 36 个可控设施全部包含在 INP 模型中
- 每个样本包含 36 维动作向量 (`actual_actions`, `candidate_action_seq`)
- 动作序列 SHA 可追溯 (`actual_schedule_sha`, `candidate_action_sha`)

### 8.5 历史帧完整性

| 指标 | 值 |
|------|-----|
| 尝试重建样本数 | 1 |
| 完整 13 帧样本数 | 1 |
| 不完整历史样本数 | 0 |
| 未来聚合方式 | 状态变量取均值 (depth, volume, rainfall)；离散动作/设置取最后值 |

---

## 9. 未解决问题与限制

### 9.1 合同冲突检查

| 冲突类型 | 数量 | 状态 |
|----------|------|------|
| Action Contract Conflicts | 0 | ✓ PASS |
| Network Mapping Conflicts | 0 | ✓ PASS |
| Reference Contract Conflicts | 0 | ✓ PASS |
| Time Contract Conflicts | 0 | ✓ PASS |

### 9.2 PFV Oracle 审计未通过

**严重程度**: 高

PFV 独立 Oracle 审计 (`pfv_oracle_audit.json`) 结果为 **FAIL**:
- 5 个审计样本全部存在标签不匹配
- 最大绝对误差: 1413.70 m³
- 存储值范围 [0.0, 9.47] m³ vs 重算值范围 [-546.70, 1413.70] m³
- 存储的 PFV delta 值与从原始轨迹重算的值存在显著偏差

**可能原因**:
- 存储的 PFV 可能使用了不同的积分方式或节点子集
- 轨迹数据在后续处理中可能被修改
- 需要检查 PFV 计算代码版本与 Oracle 审计代码的一致性

详细不匹配记录见 `audits/v42_final_pool/pfv_label_mismatches.csv`。

### 9.3 TFV/Peak Oracle 审计未通过

**严重程度**: 中

TFV/Peak Oracle 审计 (`tfv_peak_oracle_audit.json`) 结果同样为 **FAIL**:
- TFV 最大绝对误差: 130,588.81 m³，相对误差 3.87
- Peak 最大绝对误差: 43.05 m³/s，相对误差 1.33
- 存储值与重算值范围存在显著偏差

### 9.4 哨兵节点溯源未完成

**严重程度**: 低

2 个哨兵节点 (`MH0200770`, `HS1355904`) 的溯源状态为 `human_resolution_required`:
- 源自 Project4/Project5 交接上下文
- 原始选择依据文件路径为 `unverified_prior_project_context`
- SHA-256 未记录 (`source_sha256: null`)
- 阈值状态: `uncalibrated`

**影响**: 哨兵节点仅用于监测特征提取，不参与 PFV 计算，因此对核心指标无直接影响。

### 9.5 数据多样性限制

| 限制 | 说明 |
|------|------|
| 单一轮次 | 所有 1200 样本来自 round0，缺少 Round1-5 的迭代数据 |
| 单一来源 | `source_counts` 标记为 `unknown`，溯源信息不完整 |
| DWF 全覆盖 | 所有样本均为 `SOURCE_DWF_FULL_SUPERVISION`，缺少非 DWF 场景 |
| 预留集为空 | `target_no_dwf`, `source_dwf`, `consumed_development`, `reserved_evaluation` 均为 0 样本 |

### 9.6 去重考量

- 1200 样本中仅 422 个唯一物理样本，778 个为重复组内的副本
- 重复源于不同候选策略家族在同一状态下的多条记录
- 训练时需注意：若不做去重，模型可能对特定物理状态过拟合

### 9.7 已知数据缺失

| 缺失项 | 影响 |
|--------|------|
| `pool_statistics.json` | 无法提供完整的分位数统计和直方图 |
| `sample_classification_summary.json` 中 source 为 `unknown` | 无法精确追溯样本生成来源 |
| 无 Round1+ 数据 | 无法评估迭代改进效果 |
| 无独立评估集 | 所有数据均用于训练，缺少泛化性验证 |

---

## 附录 A: 文件索引

| 路径 | 说明 |
|------|------|
| `data/v42_final_unified/dataset_manifest.json` | 数据集清单 |
| `docs/contracts/PROJECT6_V42_PRIORITY_PFV_CONTRACT.json` | PFV Core8 合同 |
| `docs/contracts/kpi_contract.json` | KPI 定义合同 |
| `docs/contracts/sentinel_nodes_provenance.json` | 哨兵节点溯源 |
| `sewerrtc/v4/v42_priority_contract.py` | 节点合同代码实现 |
| `audits/v42_final_pool/` | 审计文件目录 |
| `data/project5_design/priority_pfv_core_nodes.txt` | 8 节点定义文件 |
| `data/project6_v3_sentinel_nodes.txt` | 2 哨兵节点定义文件 |
| `data/project2_design/priority_zone_nodes.csv` | 11 敏感区节点定义文件 |

---

*报告结束。如有疑问，请联系 Project6 数据审核团队。*
