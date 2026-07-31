"""V4.2 逐样本纳入分类模块。

基于所有上游审计模块（lineage、semantic、DWF、PFV oracle、TFV/Peak oracle、
history rebuild）的结果，为每个样本分配恰好一个纳入等级（AdmissionGrade）。

10 个等级（按优先级从高到低）:
  RESERVED_EVALUATION → REJECT → TARGET_FULL_SUPERVISION →
  TARGET_RECOMPUTABLE → SOURCE_DWF_FULL_SUPERVISION →
  DYNAMICS_PRETRAIN_ONLY → ACTUATOR_EFFECT_ONLY → RANKING_ONLY →
  DIAGNOSTIC_ONLY → CONSUMED_DEVELOPMENT

Output → audits/v42_final_pool/
  - sample_classification.parquet
  - sample_classification_summary.json
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sewerrtc._project_root import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_OUTPUT_DIR = "audits/v42_final_pool"

# 保留评估 split 值 — 这些样本完全禁止参与任何开发处理
_RESERVED_SPLITS = {"calibration", "locked_validation", "formal", "challenge"}

# 期望值
_EXPECTED_HISTORY_FRAMES = 13
_EXPECTED_HISTORY_COVERAGE_MIN = 60
_EXPECTED_HORIZON_STEPS = 12
_EXPECTED_HORIZON_COVERAGE_MIN = 120
_HISTORY_INTERVAL_MIN = 5
_HORIZON_INTERVAL_MIN = 10


# ---------------------------------------------------------------------------
# AdmissionGrade 枚举
# ---------------------------------------------------------------------------
class AdmissionGrade(str, Enum):
    """样本纳入等级。每个样本恰好获得一个等级。"""

    TARGET_FULL_SUPERVISION = "TARGET_FULL_SUPERVISION"
    TARGET_RECOMPUTABLE = "TARGET_RECOMPUTABLE"
    SOURCE_DWF_FULL_SUPERVISION = "SOURCE_DWF_FULL_SUPERVISION"
    DYNAMICS_PRETRAIN_ONLY = "DYNAMICS_PRETRAIN_ONLY"
    ACTUATOR_EFFECT_ONLY = "ACTUATOR_EFFECT_ONLY"
    RANKING_ONLY = "RANKING_ONLY"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    CONSUMED_DEVELOPMENT = "CONSUMED_DEVELOPMENT"
    RESERVED_EVALUATION = "RESERVED_EVALUATION"
    REJECT = "REJECT"


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _safe_get(row: pd.Series, col: str, default: Any = None) -> Any:
    """安全获取 DataFrame 行中的列值。"""
    if col in row.index:
        val = row[col]
        if pd.isna(val):
            return default
        return val
    return default


def _is_reserved_split(split_val: str) -> bool:
    """判断 split 是否属于保留评估集。"""
    if not split_val:
        return False
    return str(split_val).strip().lower() in _RESERVED_SPLITS


def _load_pfv_per_sample(project_root: Path) -> Optional[pd.DataFrame]:
    """加载 PFV oracle 逐样本结果。"""
    p = project_root / _OUTPUT_DIR / "pfv_label_recomputation.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def _load_tfv_peak_per_sample(project_root: Path) -> Optional[pd.DataFrame]:
    """加载 TFV/Peak oracle 逐样本结果。"""
    p = project_root / _OUTPUT_DIR / "tfv_peak_recomputation.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def _load_history_per_sample(project_root: Path) -> Optional[pd.DataFrame]:
    """加载 history rebuild 逐样本结果。"""
    p = project_root / _OUTPUT_DIR / "history_rebuild_details.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def _build_sample_key(event_id: Any, checkpoint_id: Any) -> str:
    """构建样本唯一键。"""
    return f"{event_id}__{checkpoint_id}"


def _build_pfv_lookup(pfv_df: pd.DataFrame) -> Dict[str, bool]:
    """构建 PFV oracle 逐样本 pass 查找。"""
    lookup: Dict[str, bool] = {}
    for _, row in pfv_df.iterrows():
        key = _build_sample_key(row.get("event_id"), row.get("checkpoint_id"))
        lookup[key] = not bool(row.get("mismatch", True))
    return lookup


def _build_tfv_peak_lookup(tfv_df: pd.DataFrame) -> Dict[str, bool]:
    """构建 TFV/Peak oracle 逐样本 pass 查找。"""
    lookup: Dict[str, bool] = {}
    for _, row in tfv_df.iterrows():
        key = _build_sample_key(row.get("event_id"), row.get("checkpoint_id"))
        tfv_ok = float(row.get("tfv_abs_error_m3", 1.0)) < 1e-6
        peak_ok = float(row.get("peak_abs_error_m3s", 1.0)) < 1e-6
        lookup[key] = tfv_ok and peak_ok
    return lookup


def _build_history_lookup(hist_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """构建 history rebuild 逐样本查找。"""
    lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in hist_df.iterrows():
        csv_path = str(row.get("detail_csv", ""))
        # 从 detail_csv 路径提取 event_id/checkpoint_id 比较困难
        # 使用 n_frames_found 和 history_incomplete 作为指标
        # 这里用行索引作为 fallback key
        key = str(row.get("detail_csv", f"row_{row.name}"))
        lookup[key] = {
            "n_frames_found": int(row.get("n_frames_found", 0)),
            "future_steps_found": int(row.get("future_steps_found", 0)),
            "history_incomplete": bool(row.get("history_incomplete", True)),
        }
    return lookup


# ---------------------------------------------------------------------------
# 核心分类逻辑
# ---------------------------------------------------------------------------

def _classify_single_sample(
    sample_idx: int,
    lineage_row: Optional[pd.Series],
    semantic_row: Optional[pd.Series],
    dwf_row: Optional[pd.Series],
    pfv_pass: Optional[bool],
    tfv_peak_pass: Optional[bool],
    history_info: Optional[Dict[str, Any]],
    is_physical_duplicate: bool,
    has_actual_action: bool,
    has_facility_response: bool,
    has_real_kpi_difference: bool,
    n_actual_unique_candidates: int,
) -> tuple[AdmissionGrade, List[str], Dict[str, Any]]:
    """对单个样本进行分类，返回 (grade, reason_codes, details)。

    分类优先级: RESERVED > REJECT > 其余等级。
    """
    reasons: List[str] = []
    details: Dict[str, Any] = {}

    # --- 条件 0: RESERVED_EVALUATION (最高优先级) ---
    split_val = ""
    if lineage_row is not None:
        split_val = str(_safe_get(lineage_row, "split", ""))
    if semantic_row is not None and not split_val:
        split_val = str(_safe_get(semantic_row, "split", ""))

    if _is_reserved_split(split_val):
        return (
            AdmissionGrade.RESERVED_EVALUATION,
            [f"reserved_split={split_val}"],
            {"split": split_val},
        )

    # --- 收集各维度信息 ---
    # 1) 来源域
    source_round = ""
    if lineage_row is not None:
        source_round = str(_safe_get(lineage_row, "source_round", ""))
    is_dwf_source = False
    if dwf_row is not None:
        is_dwf_source = bool(_safe_get(dwf_row, "is_dwf_source", False))

    # 2) 网络/Engineering36 对齐
    network_alignable = True
    if semantic_row is not None:
        n_fac = _safe_get(semantic_row, "n_facilities", 0)
        network_alignable = int(n_fac) == 36 if n_fac is not None else False

    # 3) 13 帧历史覆盖 60 min
    history_frames_ok = False
    if semantic_row is not None:
        hist_frames = _safe_get(semantic_row, "history_frames", 0)
        if hist_frames is not None:
            history_frames_ok = int(hist_frames) >= _EXPECTED_HISTORY_FRAMES
    # 也从 history audit 补充
    if history_info is not None:
        n_found = history_info.get("n_frames_found", 0)
        if n_found >= _EXPECTED_HISTORY_FRAMES:
            history_frames_ok = True

    # 4) 12 future steps covering H120
    horizon_ok = False
    if semantic_row is not None:
        h_steps = _safe_get(semantic_row, "horizon_steps", 0)
        if h_steps is not None:
            horizon_ok = int(h_steps) >= _EXPECTED_HORIZON_STEPS
    if history_info is not None:
        fut_found = history_info.get("future_steps_found", 0)
        if fut_found >= _EXPECTED_HORIZON_STEPS:
            horizon_ok = True

    # 5) 四分支完整 (Candidate, NC, DI, Hold)
    four_branches = False
    if semantic_row is not None:
        fbc = _safe_get(semantic_row, "four_branches_complete", None)
        if fbc is not None:
            four_branches = bool(fbc)
        else:
            # 从各分支 depth 推断
            cand_d = int(_safe_get(semantic_row, "candidate_action_depth", 0) or 0)
            nc_d = int(_safe_get(semantic_row, "nc_branch_depth", 0) or 0)
            di_d = int(_safe_get(semantic_row, "di_branch_depth", 0) or 0)
            hold_d = int(_safe_get(semantic_row, "hold_branch_depth", 0) or 0)
            four_branches = all(d > 0 for d in [cand_d, nc_d, di_d, hold_d])

    # 6) 四分支共享同一数值前缀状态
    prefix_state_ok = False
    if lineage_row is not None:
        psh = _safe_get(lineage_row, "prefix_state_hash", "")
        prefix_state_ok = bool(psh) and str(psh) != ""

    # 7) actual/readback actions complete
    actual_actions_ok = False
    if lineage_row is not None:
        asha = _safe_get(lineage_row, "actual_schedule_sha", "")
        actual_actions_ok = bool(asha) and str(asha) != ""

    # 8) PFV_CORE8 独立可重算
    pfv_recomputable = pfv_pass if pfv_pass is not None else False

    # 9) TFV and Peak 独立可重算
    tfv_peak_recomputable = tfv_peak_pass if tfv_peak_pass is not None else False

    # 10) Reference contract correct
    ref_contract_ok = True
    if semantic_row is not None:
        ref_contract_ok = bool(_safe_get(semantic_row, "branch_contract_ok", True))

    # 11) No future input leakage — 由 time_contract_ok 代理
    no_leakage = True
    if semantic_row is not None:
        no_leakage = bool(_safe_get(semantic_row, "time_contract_ok", True))

    # 12) Not reserved evaluation (已在条件 0 处理)
    # 13) Not a physical duplicate
    not_duplicate = not is_physical_duplicate

    # --- TARGET_FULL_SUPERVISION: 13 条件全满足 ---
    cond1_no_dwf = not is_dwf_source
    cond2 = network_alignable
    cond3 = history_frames_ok
    cond4 = horizon_ok
    cond5 = four_branches
    cond6 = prefix_state_ok
    cond7 = actual_actions_ok
    cond8 = pfv_recomputable
    cond9 = tfv_peak_recomputable
    cond10 = ref_contract_ok
    cond11 = no_leakage
    cond12 = True  # not reserved — 已在前面排除
    cond13 = not_duplicate

    all_13 = all([
        cond1_no_dwf, cond2, cond3, cond4, cond5, cond6, cond7,
        cond8, cond9, cond10, cond11, cond12, cond13,
    ])

    if all_13:
        reasons.append("all_13_conditions_met")
        return AdmissionGrade.TARGET_FULL_SUPERVISION, reasons, details

    # --- 记录缺失条件 ---
    missing_conds: List[str] = []
    if not cond1_no_dwf:
        missing_conds.append("cond1_is_dwf_source")
    if not cond2:
        missing_conds.append("cond2_network_not_alignable")
    if not cond3:
        missing_conds.append("cond3_history_frames_incomplete")
    if not cond4:
        missing_conds.append("cond4_horizon_incomplete")
    if not cond5:
        missing_conds.append("cond5_four_branches_incomplete")
    if not cond6:
        missing_conds.append("cond6_prefix_state_mismatch")
    if not cond7:
        missing_conds.append("cond7_actual_actions_missing")
    if not cond8:
        missing_conds.append("cond8_pfv_not_recomputable")
    if not cond9:
        missing_conds.append("cond9_tfv_peak_not_recomputable")
    if not cond10:
        missing_conds.append("cond10_ref_contract_broken")
    if not cond11:
        missing_conds.append("cond11_future_leakage")
    if not cond13:
        missing_conds.append("cond13_physical_duplicate")

    details["missing_conditions"] = missing_conds

    # --- TARGET_RECOMPUTABLE: 原始分支完整，可重建 13 帧并重算标签 ---
    branches_present = four_branches and actual_actions_ok
    can_rebuild = cond3 or cond4  # 至少有部分历史/未来数据可重建
    if branches_present and not is_dwf_source and not_duplicate:
        # 有分支数据但历史或 PFV/TFV 不完整 → 可重建
        if (not cond3 or not cond8 or not cond9) and cond5 and cond7:
            reasons.append("branches_complete_rebuildable")
            return AdmissionGrade.TARGET_RECOMPUTABLE, reasons, details

    # --- SOURCE_DWF_FULL_SUPERVISION: DWF 源域但其他条件大部分满足 ---
    if is_dwf_source and branches_present and not_duplicate:
        # DWF 源但分支完整
        n_met = sum([cond2, cond3, cond4, cond5, cond6, cond7,
                     cond10, cond11])
        if n_met >= 6:
            reasons.append("dwf_source_most_conditions_met")
            return AdmissionGrade.SOURCE_DWF_FULL_SUPERVISION, reasons, details

    # --- DYNAMICS_PRETRAIN_ONLY: 状态/强迫/动作/未来水力可信，但 Reference 或 KPI 不完整 ---
    state_forcing_ok = cond2 and cond3 and cond4
    actions_future_ok = cond7 and (cond5 or can_rebuild)
    kpi_incomplete = not (cond8 and cond9 and cond10)
    if state_forcing_ok and actions_future_ok and kpi_incomplete and not_duplicate:
        reasons.append("dynamics_credible_kpi_incomplete")
        return AdmissionGrade.DYNAMICS_PRETRAIN_ONLY, reasons, details

    # --- ACTUATOR_EFFECT_ONLY: 能展示实际动作变化和设施流量响应 ---
    if has_actual_action and has_facility_response and not_duplicate:
        reasons.append("actual_action_and_response_observed")
        return AdmissionGrade.ACTUATOR_EFFECT_ONLY, reasons, details

    # --- RANKING_ONLY: 同状态多实际唯一 Candidate 且 KPI 差异超死区 ---
    if n_actual_unique_candidates > 1 and has_real_kpi_difference and not_duplicate:
        reasons.append("multiple_candidates_with_kpi_difference")
        return AdmissionGrade.RANKING_ONLY, reasons, details

    # --- DIAGNOSTIC_ONLY: 部分数据可用于调试/探索 ---
    has_some_data = (
        lineage_row is not None
        or semantic_row is not None
    )
    if has_some_data and not_duplicate:
        # 有数据但不满足更高等级
        reasons.append("partial_data_diagnostic")
        return AdmissionGrade.DIAGNOSTIC_ONLY, reasons, details

    # --- CONSUMED_DEVELOPMENT: 旧评估结果已查看，允许作为开发数据 ---
    # 如果样本来自旧评估流程且已被消费
    if source_round in ("calibration", "locked_validation"):
        reasons.append("consumed_development_data")
        return AdmissionGrade.CONSUMED_DEVELOPMENT, reasons, details

    # --- REJECT: 不满足任何条件 ---
    reasons.append("no_criteria_met")
    if not not_duplicate:
        reasons.append("physical_duplicate")
    if lineage_row is None and semantic_row is None:
        reasons.append("no_audit_data_available")
    return AdmissionGrade.REJECT, reasons, details


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def classify_samples(
    project_root: Path,
    output_root: Path,
    lineage_df: pd.DataFrame | None = None,
    semantic_df: pd.DataFrame | None = None,
    dwf_df: pd.DataFrame | None = None,
    pfv_audit: dict | None = None,
    tfv_peak_audit: dict | None = None,
    history_audit: dict | None = None,
) -> pd.DataFrame:
    """将每个样本分类到恰好一个 AdmissionGrade。

    基于所有上游审计模块的结果，对每个样本应用 10 级分类体系。
    分类优先级: RESERVED_EVALUATION > REJECT > 其他等级（从高到低）。

    Parameters
    ----------
    project_root : Path
        项目根目录。
    output_root : Path
        输出根目录。
    lineage_df : pd.DataFrame | None
        ``build_sample_lineage`` 的输出。若 None 则尝试从磁盘加载。
    semantic_df : pd.DataFrame | None
        ``build_semantic_inventory`` 的输出。若 None 则尝试从磁盘加载。
    dwf_df : pd.DataFrame | None
        DWF 审计结果。若 None 则保守分类。
    pfv_audit : dict | None
        ``run_pfv_oracle_audit`` 的输出摘要。
    tfv_peak_audit : dict | None
        ``run_tfv_peak_oracle_audit`` 的输出摘要。
    history_audit : dict | None
        ``rebuild_13frame_histories`` 的输出摘要。

    Returns
    -------
    pd.DataFrame
        列: ``sample_id``, ``grade``, ``reason_codes``, ``details_json``。
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # ------------------------------------------------------------------
    # 1. 加载/验证审计数据
    # ------------------------------------------------------------------
    # Lineage
    if lineage_df is None:
        p = project_root / _OUTPUT_DIR / "sample_lineage.parquet"
        if p.exists():
            lineage_df = pd.read_parquet(p)
            logger.info("Loaded lineage from disk: %d samples", len(lineage_df))

    # Semantic
    if semantic_df is None:
        p = project_root / _OUTPUT_DIR / "semantic_sample_inventory.parquet"
        if p.exists():
            semantic_df = pd.read_parquet(p)
            logger.info("Loaded semantic inventory from disk: %d samples", len(semantic_df))

    # ------------------------------------------------------------------
    # 2. 构建逐样本查找表
    # ------------------------------------------------------------------
    # Lineage lookup by sample_idx
    lineage_lookup: Dict[int, pd.Series] = {}
    if lineage_df is not None and not lineage_df.empty:
        for _, row in lineage_df.iterrows():
            sidx = int(_safe_get(row, "sample_idx", -1))
            if sidx >= 0:
                lineage_lookup[sidx] = row

    # Semantic lookup by sample_idx
    semantic_lookup: Dict[int, pd.Series] = {}
    if semantic_df is not None and not semantic_df.empty:
        for _, row in semantic_df.iterrows():
            sidx = int(_safe_get(row, "sample_idx", -1))
            if sidx >= 0:
                semantic_lookup[sidx] = row

    # DWF lookup by sample_idx
    dwf_lookup: Dict[int, pd.Series] = {}
    if dwf_df is not None and not dwf_df.empty:
        for _, row in dwf_df.iterrows():
            sidx = int(_safe_get(row, "sample_idx", -1))
            if sidx >= 0:
                dwf_lookup[sidx] = row

    # PFV per-sample lookup
    pfv_per_sample: Dict[str, bool] = {}
    pfv_df = _load_pfv_per_sample(project_root)
    if pfv_df is not None:
        pfv_per_sample = _build_pfv_lookup(pfv_df)

    # TFV/Peak per-sample lookup
    tfv_peak_per_sample: Dict[str, bool] = {}
    tfv_df = _load_tfv_peak_per_sample(project_root)
    if tfv_df is not None:
        tfv_peak_per_sample = _build_tfv_peak_lookup(tfv_df)

    # History per-sample lookup (by detail_csv path)
    history_per_sample: Dict[str, Dict[str, Any]] = {}
    hist_df = _load_history_per_sample(project_root)
    if hist_df is not None:
        history_per_sample = _build_history_lookup(hist_df)

    # Physical duplicate set
    duplicate_indices: set = set()
    if lineage_df is not None and "is_duplicate" in lineage_df.columns:
        dup_mask = lineage_df["is_duplicate"].fillna(False)
        duplicate_indices = set(
            int(lineage_df.iloc[i]["sample_idx"])
            for i in range(len(lineage_df)) if dup_mask.iloc[i]
        )

    # ------------------------------------------------------------------
    # 3. 确定要分类的样本全集
    # ------------------------------------------------------------------
    all_sample_indices: set = set()
    if lineage_df is not None and not lineage_df.empty:
        all_sample_indices.update(int(x) for x in lineage_df["sample_idx"].tolist())
    if semantic_df is not None and not semantic_df.empty:
        all_sample_indices.update(int(x) for x in semantic_df["sample_idx"].tolist())

    if not all_sample_indices:
        logger.warning("No samples found in any audit data")
        return pd.DataFrame(columns=["sample_id", "grade", "reason_codes", "details_json"])

    # ------------------------------------------------------------------
    # 4. 构建 ranking 辅助信息：按 state_key 分组统计实际唯一 Candidate 数
    # ------------------------------------------------------------------
    state_candidate_map: Dict[str, set] = {}
    if lineage_df is not None:
        for _, row in lineage_df.iterrows():
            sk = str(_safe_get(row, "state_key", ""))
            cid = str(_safe_get(row, "candidate_id", ""))
            actual_sha = str(_safe_get(row, "actual_schedule_sha", ""))
            if sk and cid:
                if sk not in state_candidate_map:
                    state_candidate_map[sk] = set()
                # 用 actual_schedule_sha 区分实际唯一
                state_candidate_map[sk].add(f"{cid}_{actual_sha}")

    # ------------------------------------------------------------------
    # 5. 逐样本分类
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    grade_counts: Dict[str, int] = {g.value: 0 for g in AdmissionGrade}

    for sidx in sorted(all_sample_indices):
        lin_row = lineage_lookup.get(sidx)
        sem_row = semantic_lookup.get(sidx)
        dwf_row = dwf_lookup.get(sidx)

        # 构建样本 key 用于 PFV/TFV 查找
        if lin_row is not None:
            skey = _build_sample_key(
                _safe_get(lin_row, "event_id", ""),
                _safe_get(lin_row, "checkpoint_id", ""),
            )
        elif sem_row is not None:
            skey = _build_sample_key(
                _safe_get(sem_row, "event_id", ""),
                _safe_get(sem_row, "checkpoint_id", ""),
            )
        else:
            skey = ""

        pfv_pass = pfv_per_sample.get(skey) if skey else None
        tfv_peak_pass = tfv_peak_per_sample.get(skey) if skey else None

        # History info — 尝试匹配
        hist_info = None
        # history_per_sample 以 detail_csv path 为 key，无法直接按 sample_idx 匹配
        # 保守处理：仅在 history_audit 全局信息可用时使用

        is_dup = sidx in duplicate_indices

        # 是否有 actual action
        has_actual = False
        if lin_row is not None:
            asha = _safe_get(lin_row, "actual_schedule_sha", "")
            has_actual = bool(asha) and str(asha) != ""

        # 是否有设施响应（从 PFV delta 非零推断）
        has_facility_response = False
        if lin_row is not None:
            pfv_d = _safe_get(lin_row, "pfv_delta", 0)
            has_facility_response = pfv_d is not None and abs(float(pfv_d)) > 1e-12

        # 同状态实际唯一 Candidate 数
        state_key = ""
        if lin_row is not None:
            state_key = str(_safe_get(lin_row, "state_key", ""))
        elif sem_row is not None:
            state_key = str(_safe_get(sem_row, "state_key", ""))
        n_actual_unique = len(state_candidate_map.get(state_key, set()))

        # 是否有真实 KPI 差异
        has_kpi_diff = False
        if lin_row is not None:
            pfv_d = _safe_get(lin_row, "pfv_delta", 0)
            tfv_d = _safe_get(lin_row, "tfv_delta", 0)
            has_kpi_diff = (
                (pfv_d is not None and abs(float(pfv_d)) > 1e-6)
                or (tfv_d is not None and abs(float(tfv_d)) > 1e-6)
            )

        grade, reasons, det = _classify_single_sample(
            sample_idx=sidx,
            lineage_row=lin_row,
            semantic_row=sem_row,
            dwf_row=dwf_row,
            pfv_pass=pfv_pass,
            tfv_peak_pass=tfv_peak_pass,
            history_info=hist_info,
            is_physical_duplicate=is_dup,
            has_actual_action=has_actual,
            has_facility_response=has_facility_response,
            has_real_kpi_difference=has_kpi_diff,
            n_actual_unique_candidates=n_actual_unique,
        )

        # 构建 sample_id
        event_id = ""
        checkpoint_id = ""
        if lin_row is not None:
            event_id = str(_safe_get(lin_row, "event_id", ""))
            checkpoint_id = str(_safe_get(lin_row, "checkpoint_id", ""))
        elif sem_row is not None:
            event_id = str(_safe_get(sem_row, "event_id", ""))
            checkpoint_id = str(_safe_get(sem_row, "checkpoint_id", ""))

        sample_id = f"{event_id}__{checkpoint_id}" if event_id else f"sample_{sidx}"

        results.append({
            "sample_id": sample_id,
            "sample_idx": sidx,
            "event_id": event_id,
            "checkpoint_id": checkpoint_id,
            "grade": grade.value,
            "reason_codes": "|".join(reasons),
            "details_json": json.dumps(det, ensure_ascii=False, default=str),
        })
        grade_counts[grade.value] += 1

    classified_df = pd.DataFrame(results)
    logger.info(
        "Classification complete: %d samples → %s",
        len(classified_df),
        {k: v for k, v in grade_counts.items() if v > 0},
    )

    # ------------------------------------------------------------------
    # 6. 写出
    # ------------------------------------------------------------------
    out_dir = output_root / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    classified_df.to_parquet(out_dir / "sample_classification.parquet", index=False)
    logger.info("Wrote sample_classification.parquet → %s", out_dir)

    # Summary JSON
    summary = summarize_classification(classified_df)
    summary_path = out_dir / "sample_classification_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Wrote sample_classification_summary.json → %s", out_dir)

    return classified_df


def summarize_classification(classified_df: pd.DataFrame) -> dict:
    """生成分类摘要：按等级、来源、事件的计数统计。

    Parameters
    ----------
    classified_df : pd.DataFrame
        ``classify_samples`` 的输出。

    Returns
    -------
    dict
        包含 ``grade_counts``, ``source_counts``, ``event_counts``,
        ``total_samples``, ``grade_percentages`` 的摘要字典。
    """
    if classified_df.empty:
        return {
            "total_samples": 0,
            "grade_counts": {},
            "grade_percentages": {},
            "source_counts": {},
            "event_counts": {},
        }

    # 按等级计数
    grade_counts = classified_df["grade"].value_counts().to_dict()
    total = len(classified_df)
    grade_pct = {k: round(v / total * 100, 2) for k, v in grade_counts.items()}

    # 按来源计数（从 details_json 提取 source_round）
    source_counts: Dict[str, int] = {}
    for _, row in classified_df.iterrows():
        det = {}
        try:
            det = json.loads(row.get("details_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
        src = det.get("source_round", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    # 按事件计数
    event_counts: Dict[str, int] = {}
    if "event_id" in classified_df.columns:
        event_counts = (
            classified_df["event_id"]
            .value_counts()
            .to_dict()
        )

    return {
        "total_samples": total,
        "grade_counts": grade_counts,
        "grade_percentages": grade_pct,
        "source_counts": source_counts,
        "event_counts": event_counts,
    }


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main(project_root: str | Path | None = None) -> None:
    """独立运行入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if project_root is None:
        project_root = PROJECT_ROOT
    project_root = Path(project_root)

    classified_df = classify_samples(project_root, project_root)
    summary = summarize_classification(classified_df)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
