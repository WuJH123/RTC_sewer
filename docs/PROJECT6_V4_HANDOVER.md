# Project6 V4 开发交接文档

> 生成日期：2026-07-26  
> 项目路径：`E:\RTC_sewer\Project6`  
> Python环境：`.venv\Scripts\python.exe`（Python 3.10+）  
> 代码规模：sewerrtc 162个模块 / scripts 231个脚本 / tests 102个测试

---

## 1. 已完成工作总结

### 1.1 整体架构与目标

Project6 是武汉排水系统实时控制（RTC）研究项目，核心目标是：
- 基于图神经网络（GAT）重建 + 代理模型 + MPC闭环控制，实现污水管网水力恢复能力分析
- 通过多层级Gate门禁系统（Gate 0→5）逐步验证控制策略的科学有效性
- 使用SWMM水力模型作为权威仿真引擎

### 1.2 已完成的Gate阶段

| Gate | 状态 | 核心产出 |
|------|------|----------|
| Gate 0: 资产审计 | ✅ PASS | INP/YAML/Python SHA256冻结，执行链追踪 |
| Gate 1: Truth Contract | ✅ PASS | 零错误Truth Contract JSON，18个字段 |
| Gate 2: Reference Validity | ✅ CONDITIONAL_PASS | 6大审计函数，8个pytest，Dynamic Internal语义重构 |
| Gate 2.5 | ✅ PASS（经修复） | 共享前缀修复，假通过根因诊断 |
| Gate 3: H120 | ✅ PASS | 10个检查全部通过 |
| Gate 3: FULL | ⚠️ PARTIAL | 0个recovery-qualified（需Dynamic Internal补跑） |
| Gate 3.5: Recovery Proof | ✅ PASS | Recovery Contract V2，72h无雨基线，16个降雨事件 |
| Gate 4-H120 Batch 0 | ✅ 完成 | 13个CSV（3 reference + 10 candidate），标签计算 |
| Gate 5: Exact-SWMM诊断 | ❌ FAIL | 18候选全部相同输出，candidate_coverage_failure |

### 1.3 Gate 4-H120 Batch 0 详细成果

**已完成**：
- 10个粗糙候选方案（uniform_90pct, all_open, all_closed, random等）
- H120标签独立计算验证
- Batch 0真实性审计（script 242）
- Gate 3.5 v2最终证据汇总（script 241）

**关键数据**：
```
NC (No Control):    PFV=9.20 m³,  TFV=76.92 m³
DI (Dynamic Internal): PFV=16.25 m³, TFV=35.17 m³
uniform_90pct:      PFV=13.82 m³, TFV=32.39 m³
```

### 1.4 Gate 5 消融分析成果

**16路并行SWMM运行**（55次仿真，50分钟完成）：
- 36个leave-one-actuator-out（LOO）
- 7个leave-one-group-out
- 12个perturbation（top-3设施 × 4个delta）

**核心发现**：
- 26/36个设施LOO结果完全相同 → 大多数设施在此事件中无边际效应
- 仅6个设施有差异化影响：
  - `HS2512760.1`: 恢复后TFV降低2.1 m³（tfv_improving）
  - `gbz1.8`: 恢复后TFV降低2.7 m³（tfv_improving）
  - `RTC_IN/OUT_02`, `RTC_OUT_03`: storage设施有显著影响
  - `ADD301.2/3`: 二值泵有影响

### 1.5 已创建的代码模块

| 文件 | 功能 |
|------|------|
| `sewerrtc/control/v4_candidate_generator.py` | V4稀疏候选生成器（5家族） |
| `scripts/240_audit_v4_gate4_h120_batch0.py` | Batch 0审计 |
| `scripts/241_finalize_v4_gate35_evidence.py` | Gate 3.5最终证据汇总 |
| `scripts/242_reaudit_v4_gate4_batch0.py` | Batch 0真实性审计 |
| `scripts/243_exact_ablate_uniform90_v4.py` | 16路并行消融分析 |
| `scripts/244_scan_existing_data_reuse.py` | 旧数据复用扫描 |
| `scripts/245_run_v4_gate5_exact_candidate_diagnosis.py` | Gate 5诊断（16并行） |
| `scripts/246_audit_v4_gate5_exact_candidate_diagnosis.py` | Gate 5审计 |
| `scripts/247_build_eng36_sensitivity_map.py` | Engineering36敏感度图 |

---

## 2. 技术难点与踩过的坑

### 2.1 H120标签计算Bug（严重）

**问题**：`compute_h120_labels()` 返回全零，因为查找的列名与实际CSV列名不匹配。

**根因**：函数查找 `flood:NODENAME` 列，但CSV中的列名格式不同。

**修复**：确保列名匹配逻辑正确：
```python
flood_cols = [c for c in window.columns if c.startswith("flood:")]
pfv_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in flood_cols]
```

**教训**：标签计算函数必须与数据输出格式严格对齐，需要独立验证。

### 2.2 JSON序列化错误

**问题**：`TypeError: Object of type bool is not JSON serializable`

**根因**：numpy的bool_/int64类型不能直接JSON序列化。

**修复**：将所有numpy类型转换为Python原生类型：
```python
bool(all_k_ok)   # 而非 all_k_ok (numpy bool)
int(n_noop)      # 而非 n_noop (numpy int)
```

### 2.3 并行SWMM运行的临时文件冲突

**问题**：多个pyswmm实例同时运行时会互相覆盖临时文件。

**修复**：每个并行worker必须拥有独立的INP副本和独立的工作目录：
```python
run_dir = Path(run_dir)
run_dir.mkdir(parents=True, exist_ok=True)
local_inp = run_dir / Path(source_inp).name
shutil.copy2(str(source_inp), str(local_inp))
```

### 2.4 Gate 5水力平坦问题（核心失败）

**问题**：18个候选全部产生完全相同的PFV/TFV/Peak输出。

**根因分析**：
1. DI规则在checkpoint时刻将所有36个设施推到极端值{0.0, 1.0}
2. V4CandidateGenerator在DI基础上做小扰动（±0.05~0.20）
3. 从0.0变到0.1或从1.0变到0.9，对水力结果零影响
4. 系统在这个极端操作点附近是完全平坦的

**消融证据**：
- 将设施从0.9大幅恢复到0.5时，只有HS2512760.1和gbz1.8有响应
- 从极端值{0,1}微调±0.1没有任何效果

**设计决策的权衡**：
- K≤8约束限制了搜索空间大小
- 二值泵合规约束进一步收窄了可行域
- 但核心问题是搜索方向错误（应该大跨度变化，而非微调）

### 2.5 数据扫描超时

**问题**：`rglob`在大型outputs目录上超时。

**修复**：限制扫描深度为1层，缩小扫描目录范围。

### 2.6 格式化错误（NoneType）

**问题**：`TypeError: unsupported format string passed to NoneType.__format__`

**根因**：`dwf_audit.get('mean_effective_dwf_m3s')` 返回None时尝试格式化。

**修复**：先检查None再格式化。

---

## 3. 后续工作注意事项

### 3.1 最高优先级：解决Gate 5水力平坦问题

**核心问题**：当前候选生成策略在DI基准附近完全无效。

**建议方案**（按优先级排序）：

1. **大跨度扰动**：
   - 不要从DI的{0,1}极端值微调±0.05
   - 改为：将敏感设施从极端值大幅移动到中间值（如0→0.5, 1→0.3）
   - 步长至少0.2-0.3，而非0.05

2. **聚焦敏感设施**：
   - 消融已识别的敏感设施：`HS2512760.1`, `gbz1.8`, `RTC_IN/OUT_02`, `RTC_OUT_03`
   - 只变化这些设施，保持其他设施不变
   - 这比随机变化36个设施更有效

3. **更换基准点**：
   - DI基准本身可能是最差情况（所有设施全开/全关）
   - 考虑以uniform_90pct或midpoint(0.5)作为候选生成基准
   - 从这些基准出发，变化幅度更有意义

4. **扩大K值限制**：
   - 当前K≤8可能太严格
   - 考虑K≤12或K≤18，允许更多设施同时变化

### 3.2 关键风险点

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| DI基准水力平坦 | 候选搜索无效 | 更换基准或大跨度扰动 |
| 敏感设施仅4-6个 | 搜索空间过小 | 聚焦这些设施做精细搜索 |
| Storage interlocks | 违反约束导致无效候选 | 生成时强制interlock约束 |
| 二值泵语义 | ADD301.2/3必须{0,1} | 生成后强制二值化 |
| SWMM运行时间 | 每个~5分钟 | 保持16路并行 |

### 3.3 业务逻辑注意事项

1. **设施语义差异**：
   - `ADD301.2`, `ADD301.3`：严格二值泵，动作集{0, 1}
   - `add350.1`：变速泵，连续动作（bounds待确认）
   - 28个orifice/weir：连续[0,1]
   - Storage inlet/outlet：不能同时开启（interlock）

2. **Reference角色**：
   - No-control = all-open（诊断用，非oracle）
   - Dynamic Internal = SWMM内置规则（PFV有效性基准）
   - Safe fallback = 在线安全和动作必要性基准

3. **标签计算**：
   - PFV：priority节点的flood累积（H120窗口内）
   - TFV：所有节点flood累积
   - Peak：H120窗口内最大flood rate（flood/300s）
   - 死区：TFV > 0.5 m³, Peak > 0.001 m³/s 才算有效信号

### 3.4 运行约束

- **禁止**：批量生成训练数据、训练Surrogate、修改INP、降低阈值
- **必须**：每个SWMM运行独立目录，避免临时文件冲突
- **必须**：所有标签独立重新计算，不信任缓存值
- **并行度**：最多16路SWMM并行（用户确认电脑可承受）

---

## 4. 代码和文件路径信息

### 4.1 核心数据文件

| 路径 | 用途 |
|------|------|
| `data/wuhan_v8_storage_retrofit.inp` | 唯一物理网络（V3证据） |
| `data/project6_v3_facility_semantics_36.csv` | 36个受管设施语义表 |
| `data/project6_v8_storage_retrofit_control_enabled_ids.txt` | 36个受管设施ID列表 |
| `data/project6_v3_sentinel_nodes.txt` | 哨兵节点列表 |
| `outputs/rainfall_library_v8_storage_variablepump/rainfall_event_table.csv` | 降雨事件表 |

### 4.2 关键输出目录

| 路径 | 用途 |
|------|------|
| `outputs/project6_dual_reference_v4/recovery_capability_v2/gate4_h120_batch0/` | Gate 4/5主工作目录 |
| `.../gate4_h120_batch0/work/` | Batch 0 SWMM运行detail CSV |
| `.../gate4_h120_batch0/ablation_uniform90/` | 消融分析结果 |
| `.../gate4_h120_batch0/ablation_uniform90/parallel_runs/` | 55个并行运行detail |
| `.../gate4_h120_batch0/gate5_exact_diagnosis/` | Gate 5诊断结果 |
| `.../gate4_h120_batch0/gate5_exact_diagnosis/parallel_runs/` | 18个候选并行运行 |

### 4.3 关键证据文件

| 文件 | 内容 |
|------|------|
| `batch0_results.csv` | Batch 0的13行结果（3 ref + 10 cand） |
| `batch0_truth_reaudit.json` | Batch 0真实性审计（verdict=diagnostic_only） |
| `v4_gate35_final_evidence_summary.json` | Gate 3.5 v2全8阶段证据 |
| `ablation_uniform90/uniform90_ablation_audit.json` | 消融审计（36 LOO + 7 group + 12 pert） |
| `ablation_uniform90/facility_marginal_effects.csv` | 36个设施LOO边际效应 |
| `gate5_exact_diagnosis/gate5_candidate_audit.json` | Gate 5审计（FAIL） |
| `gate5_exact_diagnosis/gate5_candidate_results.csv` | 18个候选结果（全相同） |
| `gate5_exact_diagnosis/eng36_sensitivity_map.csv` | 36设施敏感度分类 |
| `gate5_exact_diagnosis/eng36_sensitivity_audit.json` | 敏感度审计 |

### 4.4 核心代码模块

| 模块 | 功能 |
|------|------|
| `sewerrtc/simulation/pyswmm_runner.py` | SWMM仿真运行器（run_swmm_fixed_action等） |
| `sewerrtc/io/swmm_mutation.py` | INP文件变异（mutate_inp_for_event） |
| `sewerrtc/data/round0_prompt2.py` | 数据加载（_priority_nodes等） |
| `sewerrtc/control/v4_candidate_generator.py` | V4候选生成器（5家族） |
| `sewerrtc/control/dual_reference_v4.py` | V4双参考控制策略 |
| `sewerrtc/control/pfvfirst_dualfallback.py` | PFV-first双回退控制器 |
| `sewerrtc/contracts/prompt3a.py` | Prompt3A契约定义 |

### 4.5 配置文件

| 文件 | 用途 |
|------|------|
| `configs/wuhan_project6_dual_reference_v4.yaml` | V4双参考主配置 |
| `configs/wuhan_v8_storage_retrofit.yaml` | V8 storage retrofit配置 |
| `configs/wuhan_project6_36_*.yaml` | 36设施系列实验配置 |

### 4.6 模块依赖关系

```
scripts/245 (Gate 5运行)
  ├── sewerrtc/control/v4_candidate_generator.py
  │     └── 读取 facility_semantics, sensitivity_map
  ├── sewerrtc/simulation/pyswmm_runner.py
  │     └── run_swmm_fixed_action() → SWMM仿真
  ├── sewerrtc/io/swmm_mutation.py
  │     └── mutate_inp_for_event() → INP准备
  └── sewerrtc/data/round0_prompt2.py
        └── _priority_nodes() → priority节点列表

scripts/243 (消融分析)
  ├── sewerrtc/simulation/pyswmm_runner.py
  ├── sewerrtc/io/swmm_mutation.py
  └── sewerrtc/data/round0_prompt2.py

scripts/247 (敏感度图)
  └── 读取 ablation_uniform90/facility_marginal_effects.csv
```

---

## 5. 代码逻辑验证要点

### 5.1 最高优先级：V4CandidateGenerator的搜索策略

**问题**：当前生成器在DI极端动作附近产生水力平坦的候选。

**需要检查**：
```python
# sewerrtc/control/v4_candidate_generator.py
# generate_di_neighborhood() 方法
# 问题：delta扰动太小（±0.05~0.20），对极端值{0,1}无效
```

**验证方法**：
1. 打印DI动作向量，确认是否全在{0,1}
2. 打印候选动作向量，确认与DI的差异
3. 对比候选输出与DI输出，确认是否有差异
4. 如果所有候选输出相同，说明搜索策略需要根本性改变

**检查点**：
```python
# 在 scripts/245 中添加诊断：
for cand in candidates[:5]:
    diff = cand.action - di_action
    print(f"{cand.candidate_id}: K={cand.k_actual}, "
          f"nonzero_changes={np.sum(np.abs(diff) > 0.01)}")
```

### 5.2 H120标签计算的正确性

**需要验证**：
- `compute_h120_labels()` 函数的列名匹配逻辑
- PFV/TFV/Peak的计算公式
- 时间窗口边界（checkpoint_min到checkpoint_min + h120_min）

**独立验证方法**：
```python
# 手动读取detail CSV，独立计算标签
df = pd.read_csv(detail_csv)
window = df[(df["elapsed_min"] >= adj_cp) & (df["elapsed_min"] < adj_cp + 120)]
flood_cols = [c for c in window.columns if c.startswith("flood:")]
# 对比函数输出与手动计算结果
```

### 5.3 并行运行的隔离性

**需要验证**：
- 每个worker是否使用独立的INP副本
- 是否存在临时文件冲突
- detail CSV是否写入正确目录

**检查方法**：
```bash
# 检查parallel_runs目录结构
Get-ChildItem outputs/.../parallel_runs -Directory | ForEach-Object {
    $files = Get-ChildItem $_.FullName -File
    "$($_.Name): $($files.Count) files"
}
```

### 5.4 消融结果的一致性

**需要验证**：
- LOO结果中，26个"无影响"设施是否真的无影响
- 还是因为SWMM运行的确定性导致所有运行结果相同

**验证方法**：
```python
# 检查不同LOO运行的detail CSV是否真的相同
import hashlib
files = list(ABLATION_DIR.glob("parallel_runs/loo_*/ablation_*_detail.csv"))
hashes = [hashlib.md5(open(f,'rb').read()).hexdigest() for f in files]
print(f"Unique hashes: {len(set(hashes))}/{len(hashes)}")
# 如果只有1个unique hash，说明所有运行完全相同（可能是bug）
```

### 5.5 设施语义约束

**需要验证**：
- ADD301.2/ADD301.3是否严格{0,1}
- Storage inlet/outlet是否遵守interlock
- add350.1的bounds是否正确

**检查方法**：
```python
# 从gate5_candidate_results.csv检查action_hash唯一性
# 从detail CSV检查实际动作值
for detail in gate5_details:
    df = pd.read_csv(detail)
    for bp in ["ADD301.2", "ADD301.3"]:
        col = f"a:{bp}"
        if col in df.columns:
            unique_vals = df[col].unique()
            assert set(unique_vals).issubset({0.0, 1.0}), f"{bp} not binary!"
```

### 5.6 需要进一步审查的代码区域

| 区域 | 风险等级 | 审查重点 |
|------|----------|----------|
| `v4_candidate_generator.py` | 🔴 高 | 搜索策略、步长、基准选择 |
| `pyswmm_runner.py` | 🟡 中 | 并行安全性、临时文件管理 |
| `compute_h120_labels()` | 🟡 中 | 列名匹配、窗口边界 |
| `swmm_mutation.py` | 🟢 低 | INP修改逻辑已稳定 |
| `reference_validity_v4.py` | 🟢 低 | 审计函数已验证 |

---

## 6. 快速启动指南

### 6.1 运行Gate 5诊断（当前最新）

```powershell
cd E:\RTC_sewer\Project6
.venv\Scripts\python.exe scripts\245_run_v4_gate5_exact_candidate_diagnosis.py
```

预期输出：
- `gate5_candidate_results.csv`：候选结果表
- `gate5_candidate_audit.json`：审计JSON
- 运行时间：~20分钟（16并行）

### 6.2 运行消融分析

```powershell
.venv\Scripts\python.exe scripts\243_exact_ablate_uniform90_v4.py
```

预期输出：
- `facility_marginal_effects.csv`：36个设施边际效应
- `uniform90_ablation_audit.json`：消融审计
- 运行时间：~50分钟（16并行）

### 6.3 运行敏感度图

```powershell
.venv\Scripts\python.exe scripts\247_build_eng36_sensitivity_map.py
```

前提：消融分析已完成。

### 6.4 运行测试

```powershell
.venv\Scripts\python.exe -m pytest tests/ -x -q
```

---

## 7. 关键决策记录

| 决策 | 原因 | 影响 |
|------|------|------|
| 使用16路并行SWMM | 用户确认可承受 | 55次仿真从4.5h降到50min |
| K≤8约束 | 稀疏控制需求 | 搜索空间受限 |
| DI作为基准 | 合约规定 | 导致水力平坦问题 |
| 不训练Surrogate | Gate 5未过 | 无法批量生成数据 |
| 不修改INP | 合约冻结 | 无法改变网络拓扑 |
| 不降低阈值 | 科学严谨性 | Gate 5标准不妥协 |

---

## 8. 当前阻塞项

1. **Gate 5 FAIL**：候选空间水力平坦，需要重新设计搜索策略
2. **Gate 3 FULL = PARTIAL**：需要Dynamic Internal真实补跑才能完成
3. **Surrogate未训练**：Gate 5未过，不能开始数据生成
4. **Pilot数据计划取消**：依赖Gate 5 PASS

**解除阻塞的关键**：修改V4CandidateGenerator，使用大跨度扰动（非微调），聚焦敏感设施。
