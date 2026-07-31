"""V4.2 最终统一数据集组装模块。

从分类结果中组装 12 个最终任务数据集，输出到 ``data/v42_final_unified/``。

12 个数据集:
  1. target_no_dwf_full_supervision.parquet
  2. source_dwf_full_supervision.parquet
  3. dynamics_pretrain.parquet
  4. actuator_effect.parquet
  5. pfv_constraint_core8.parquet
  6. tfv_objective.parquet
  7. peak_constraint.parquet
  8. within_state_ranking_pairs.parquet
  9. consumed_development.parquet
  10. reserved_evaluation_manifest.csv
  11. rejected_samples.csv
  12. sample_lineage.parquet

Output → data/v42_final_unified/
  - 上述 12 个文件
  - dataset_manifest.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sewerrtc._project_root import PROJECT_ROOT
from sewerrtc.v4.v42_sample_classifier import AdmissionGrade
from sewerrtc.v4.v42_priority_contract import CONTRACT_ID, PFV_CORE_8_IDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_OUTPUT_DIR_NAME = "v42_final_unified"
_AUDIT_DIR = "audits/v42_final_pool"
_V42_MANIFEST_PARQUET = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_manifest_v42.parquet"
)
_N_HISTORY_FRAMES = 13
_N_HORIZON_STEPS = 12
_N_FACILITIES = 36
_PRIORITY_CONTRACT_ID = CONTRACT_ID  # "PFV_CORE8_V1"


# ---------------------------------------------------------------------------
# Schema 定义 — 每个数据集的列
# ---------------------------------------------------------------------------

def _base_schema_columns() -> List[str]:
    """所有数据集共享的基础列。"""
    return [
        "sample_id", "sample_idx", "event_id", "checkpoint_id",
        "state_key", "split", "source_round", "grade",
        "priority_contract_id",
    ]


def _full_supervision_schema() -> List[str]:
    """全监督数据集 schema（target + source_dwf）。"""
    return _base_schema_columns() + [
        "rainfall_sha",
        "prefix_state_hash",
        "actual_schedule_sha",
        # 13-frame history (stored as serialized arrays)
        "history_frames",         # int, expected 13
        "history_interval_min",   # int, expected 5
        # 12-step future
        "horizon_steps",          # int, expected 12
        "horizon_interval_min",   # int, expected 10
        # Actual actions: 12×36
        "actual_actions",         # serialized 12×36 array
        # Branch trajectories (12×n_nodes flooding rate)
        "trajectory_candidate",
        "trajectory_no_control",
        "trajectory_dynamic_internal",
        "trajectory_hold_previous",
        # PFV_CORE8 labels
        "pfv_delta",
        "pfv_safe_label",
        # TFV / Peak labels
        "tfv_delta",
        "tfv_improved_label",
        "peak_delta",
        "peak_noninferior_label",
        # Priority node IDs (frozen)
        "priority_node_ids",
        # Branch action sequences
        "candidate_action_seq",
        "ref_no_control_action_seq",
        "ref_dynamic_internal_action_seq",
        "ref_hold_previous_action_seq",
        # Rainfall forecast
        "rainfall_forecast",
        # DWF flag
        "dwf_flag",
    ]


def _dynamics_pretrain_schema() -> List[str]:
    """动力学预训练 schema。"""
    return _base_schema_columns() + [
        "rainfall_sha", "prefix_state_hash",
        "history_frames", "history_interval_min",
        "horizon_steps", "horizon_interval_min",
        "trajectory_candidate", "trajectory_no_control",
        "trajectory_dynamic_internal", "trajectory_hold_previous",
        "candidate_action_seq", "ref_no_control_action_seq",
        "ref_dynamic_internal_action_seq", "ref_hold_previous_action_seq",
        "rainfall_forecast",
        "pfv_delta", "tfv_delta", "peak_delta",
        "pfv_safe_label", "tfv_improved_label", "peak_noninferior_label",
        "priority_node_ids",
    ]


def _actuator_effect_schema() -> List[str]:
    """执行器效应学习 schema。"""
    return _base_schema_columns() + [
        "actual_schedule_sha",
        "actual_actions",
        "candidate_action_seq",
        "trajectory_candidate",
        "trajectory_no_control",
        "pfv_delta", "tfv_delta", "peak_delta",
        "priority_node_ids",
    ]


def _pfv_constraint_schema() -> List[str]:
    """PFV 约束数据集 schema（仅 8 节点）。"""
    return _base_schema_columns() + [
        "pfv_delta", "pfv_safe_label",
        "trajectory_candidate_pfv_core8",
        "trajectory_no_control_pfv_core8",
        "priority_node_ids",
    ]


def _tfv_objective_schema() -> List[str]:
    """TFV 目标数据集 schema。"""
    return _base_schema_columns() + [
        "tfv_delta", "tfv_improved_label",
        "trajectory_candidate", "trajectory_dynamic_internal",
        "priority_node_ids",
    ]


def _peak_constraint_schema() -> List[str]:
    """Peak 约束数据集 schema。"""
    return _base_schema_columns() + [
        "peak_delta", "peak_noninferior_label",
        "trajectory_candidate", "trajectory_dynamic_internal",
        "priority_node_ids",
    ]


def _ranking_pairs_schema() -> List[str]:
    """同状态排序对 schema。"""
    return [
        "pair_id", "state_key",
        "sample_id_a", "sample_id_b",
        "pfv_delta_a", "pfv_delta_b",
        "peak_delta_a", "peak_delta_b",
        "tfv_delta_a", "tfv_delta_b",
        "pfv_safe_a", "pfv_safe_b",
        "tfv_improved_a", "tfv_improved_b",
        "peak_noninferior_a", "peak_noninferior_b",
        "event_id", "checkpoint_id",
        "priority_contract_id",
    ]


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _load_audit_parquet(project_root: Path, name: str) -> pd.DataFrame:
    """加载 audits/v42_final_pool/ 下的 parquet 文件。"""
    p = project_root / _AUDIT_DIR / name
    if p.exists():
        return pd.read_parquet(p)
    logger.warning("Audit parquet not found: %s", p)
    return pd.DataFrame()


def _load_trajectory_manifest(project_root: Path) -> pd.DataFrame:
    """加载 V42 trajectory manifest。"""
    p = project_root / _V42_MANIFEST_PARQUET
    if p.exists():
        return pd.read_parquet(p)
    raise FileNotFoundError(f"V42 trajectory manifest not found: {p}")


def _safe_get(row: Any, col: str, default: Any = None) -> Any:
    """安全获取行（Series 或 dict）中的列值。"""
    if isinstance(row, dict):
        return row.get(col, default)
    if hasattr(row, 'index'):
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                return default
            return val
    return default


def _serialize_array(val: Any) -> Any:
    """将 trajectory/action 值转为 parquet 友好格式。

    如果值已经是字符串（manifest 中的存储格式），直接透传以避免
    ast.literal_eval 的性能开销（每个 trajectory 字符串 ~200KB）。
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        # 直接透传字符串，不做反序列化
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return val


def _extract_pfv_core8(
    trajectory: Any,
    node_ids: List[str],
) -> Any:
    """从全节点 trajectory 中提取 8 个 PFV 核心节点的列。

    trajectory shape: (T, n_nodes) → 返回 (T, 8)。
    若 trajectory 为字符串（未解析），直接透传。
    """
    if trajectory is None or (isinstance(trajectory, float) and np.isnan(trajectory)):
        return None
    if isinstance(trajectory, str):
        # 透传原始字符串，避免反序列化开销
        return trajectory
    try:
        a = np.asarray(trajectory, dtype=np.float64)
        if a.ndim == 2 and a.shape[1] >= 8:
            return a[:, :8].tolist()
    except (ValueError, TypeError):
        pass
    return trajectory


def _merge_classified_with_manifest(
    classified_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
) -> pd.DataFrame:
    """将分类结果与 trajectory manifest 和 lineage 合并。

    使用位置对齐（而非键合并），因为 classified/manifest/lineage 都是
    同一批样本按相同顺序排列（各 1200 行，sample_idx 0..1199）。
    使用键合并会在 (event_id, checkpoint_id) 上产生多对多笛卡尔积。
    """
    if manifest_df.empty:
        return classified_df.copy()

    # 位置对齐：确保长度一致
    n = len(classified_df)
    if len(manifest_df) != n:
        logger.warning(
            "Classified (%d) and manifest (%d) length mismatch; "
            "truncating to shorter length",
            n, len(manifest_df),
        )
        n = min(n, len(manifest_df))

    cls_sub = classified_df.iloc[:n].reset_index(drop=True)
    man_sub = manifest_df.iloc[:n].reset_index(drop=True)

    # 去除 manifest 中与 classified 重复的列
    overlap_cols = set(cls_sub.columns) & set(man_sub.columns)
    drop_from_man = overlap_cols - {"sample_idx"}
    man_clean = man_sub.drop(columns=[c for c in drop_from_man if c in man_sub.columns], errors="ignore")

    merged = pd.concat([cls_sub, man_clean], axis=1)

    # 合并 lineage（同样位置对齐）
    if not lineage_df.empty and len(lineage_df) >= n:
        lin_sub = lineage_df.iloc[:n].reset_index(drop=True)
        lin_cols = ["rainfall_fingerprint", "prefix_state_hash",
                    "actual_schedule_sha", "source_round"]
        lin_available = [c for c in lin_cols if c in lin_sub.columns]
        # 避免列名冲突
        for c in lin_available:
            if c in merged.columns:
                merged[f"{c}_lin"] = lin_sub[c]
            else:
                merged[c] = lin_sub[c].values

    return merged


# ---------------------------------------------------------------------------
# 各数据集构建函数
# ---------------------------------------------------------------------------

def build_target_full_supervision(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build target_no_dwf_full_supervision.parquet。

    筛选 grade == TARGET_FULL_SUPERVISION 且非 DWF 的样本。

    Returns
    -------
    int
        样本数。
    """
    grade = AdmissionGrade.TARGET_FULL_SUPERVISION.value
    mask = classified_df["grade"] == grade
    subset = classified_df.loc[mask].copy()

    schema = _full_supervision_schema()
    if subset.empty:
        # 创建空文件但 schema 正确
        empty = pd.DataFrame(columns=schema)
        empty.to_parquet(output_path, index=False)
        logger.info("target_no_dwf_full_supervision: 0 samples (empty schema)")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    rows = _build_full_supervision_rows(merged, dwf_flag=False)
    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("target_no_dwf_full_supervision: %d samples", len(df))
    return len(df)


def build_source_dwf_full_supervision(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build source_dwf_full_supervision.parquet。

    筛选 grade == SOURCE_DWF_FULL_SUPERVISION 的样本。

    Returns
    -------
    int
        样本数。
    """
    grade = AdmissionGrade.SOURCE_DWF_FULL_SUPERVISION.value
    mask = classified_df["grade"] == grade
    subset = classified_df.loc[mask].copy()

    schema = _full_supervision_schema()
    if subset.empty:
        empty = pd.DataFrame(columns=schema)
        empty.to_parquet(output_path, index=False)
        logger.info("source_dwf_full_supervision: 0 samples (empty schema)")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    rows = _build_full_supervision_rows(merged, dwf_flag=True)
    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("source_dwf_full_supervision: %d samples", len(df))
    return len(df)


def _build_full_supervision_rows(
    merged: pd.DataFrame,
    dwf_flag: bool,
) -> List[dict]:
    """为全监督数据集构建行数据。"""
    rows = []
    frozen_ids = list(PFV_CORE_8_IDS)
    for _, row in merged.iterrows():
        rows.append({
            "sample_id": _safe_get(row, "sample_id", ""),
            "sample_idx": _safe_get(row, "sample_idx"),
            "event_id": _safe_get(row, "event_id", ""),
            "checkpoint_id": _safe_get(row, "checkpoint_id", ""),
            "state_key": _safe_get(row, "state_key", ""),
            "split": _safe_get(row, "split", ""),
            "source_round": _safe_get(row, "source_round", ""),
            "grade": _safe_get(row, "grade", ""),
            "priority_contract_id": _PRIORITY_CONTRACT_ID,
            "rainfall_sha": _safe_get(row, "rainfall_fingerprint", ""),
            "prefix_state_hash": _safe_get(row, "prefix_state_hash", ""),
            "actual_schedule_sha": _safe_get(row, "actual_schedule_sha", ""),
            "history_frames": _N_HISTORY_FRAMES,
            "history_interval_min": 5,
            "horizon_steps": _N_HORIZON_STEPS,
            "horizon_interval_min": 10,
            "actual_actions": _serialize_array(
                _safe_get(row, "candidate_action_seq")
            ),
            "trajectory_candidate": _serialize_array(
                _safe_get(row, "trajectory_depth_candidate")
            ),
            "trajectory_no_control": _serialize_array(
                _safe_get(row, "trajectory_depth_no_control")
            ),
            "trajectory_dynamic_internal": _serialize_array(
                _safe_get(row, "trajectory_depth_dynamic_internal")
            ),
            "trajectory_hold_previous": _serialize_array(
                _safe_get(row, "trajectory_depth_hold_previous")
            ),
            "pfv_delta": _safe_get(row, "pfv_delta"),
            "pfv_safe_label": _safe_get(row, "pfv_safe_label"),
            "tfv_delta": _safe_get(row, "tfv_delta"),
            "tfv_improved_label": _safe_get(row, "tfv_improved_label"),
            "peak_delta": _safe_get(row, "peak_delta"),
            "peak_noninferior_label": _safe_get(row, "peak_noninferior_label"),
            "priority_node_ids": frozen_ids,
            "candidate_action_seq": _serialize_array(
                _safe_get(row, "candidate_action_seq")
            ),
            "ref_no_control_action_seq": _serialize_array(
                _safe_get(row, "ref_no_control_action_seq")
            ),
            "ref_dynamic_internal_action_seq": _serialize_array(
                _safe_get(row, "ref_dynamic_internal_action_seq")
            ),
            "ref_hold_previous_action_seq": _serialize_array(
                _safe_get(row, "ref_hold_previous_action_seq")
            ),
            "rainfall_forecast": _serialize_array(
                _safe_get(row, "rainfall_forecast")
            ),
            "dwf_flag": dwf_flag,
        })
    return rows


def build_dynamics_pretrain(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build dynamics_pretrain.parquet。

    包含 DYNAMICS_PRETRAIN_ONLY 及以上等级的样本。

    Returns
    -------
    int
        样本数。
    """
    allowed_grades = {
        AdmissionGrade.TARGET_FULL_SUPERVISION.value,
        AdmissionGrade.TARGET_RECOMPUTABLE.value,
        AdmissionGrade.SOURCE_DWF_FULL_SUPERVISION.value,
        AdmissionGrade.DYNAMICS_PRETRAIN_ONLY.value,
    }
    mask = classified_df["grade"].isin(allowed_grades)
    subset = classified_df.loc[mask].copy()

    schema = _dynamics_pretrain_schema()
    if subset.empty:
        pd.DataFrame(columns=schema).to_parquet(output_path, index=False)
        logger.info("dynamics_pretrain: 0 samples")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    frozen_ids = list(PFV_CORE_8_IDS)
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "sample_id": _safe_get(row, "sample_id", ""),
            "sample_idx": _safe_get(row, "sample_idx"),
            "event_id": _safe_get(row, "event_id", ""),
            "checkpoint_id": _safe_get(row, "checkpoint_id", ""),
            "state_key": _safe_get(row, "state_key", ""),
            "split": _safe_get(row, "split", ""),
            "source_round": _safe_get(row, "source_round", ""),
            "grade": _safe_get(row, "grade", ""),
            "priority_contract_id": _PRIORITY_CONTRACT_ID,
            "rainfall_sha": _safe_get(row, "rainfall_fingerprint", ""),
            "prefix_state_hash": _safe_get(row, "prefix_state_hash", ""),
            "history_frames": _N_HISTORY_FRAMES,
            "history_interval_min": 5,
            "horizon_steps": _N_HORIZON_STEPS,
            "horizon_interval_min": 10,
            "trajectory_candidate": _serialize_array(
                _safe_get(row, "trajectory_depth_candidate")),
            "trajectory_no_control": _serialize_array(
                _safe_get(row, "trajectory_depth_no_control")),
            "trajectory_dynamic_internal": _serialize_array(
                _safe_get(row, "trajectory_depth_dynamic_internal")),
            "trajectory_hold_previous": _serialize_array(
                _safe_get(row, "trajectory_depth_hold_previous")),
            "candidate_action_seq": _serialize_array(
                _safe_get(row, "candidate_action_seq")),
            "ref_no_control_action_seq": _serialize_array(
                _safe_get(row, "ref_no_control_action_seq")),
            "ref_dynamic_internal_action_seq": _serialize_array(
                _safe_get(row, "ref_dynamic_internal_action_seq")),
            "ref_hold_previous_action_seq": _serialize_array(
                _safe_get(row, "ref_hold_previous_action_seq")),
            "rainfall_forecast": _serialize_array(
                _safe_get(row, "rainfall_forecast")),
            "pfv_delta": _safe_get(row, "pfv_delta"),
            "tfv_delta": _safe_get(row, "tfv_delta"),
            "peak_delta": _safe_get(row, "peak_delta"),
            "pfv_safe_label": _safe_get(row, "pfv_safe_label"),
            "tfv_improved_label": _safe_get(row, "tfv_improved_label"),
            "peak_noninferior_label": _safe_get(row, "peak_noninferior_label"),
            "priority_node_ids": frozen_ids,
        })

    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("dynamics_pretrain: %d samples", len(df))
    return len(df)


def build_actuator_effect(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build actuator_effect.parquet。

    包含 ACTUATOR_EFFECT_ONLY 及以上等级。

    Returns
    -------
    int
        样本数。
    """
    allowed_grades = {
        AdmissionGrade.TARGET_FULL_SUPERVISION.value,
        AdmissionGrade.TARGET_RECOMPUTABLE.value,
        AdmissionGrade.SOURCE_DWF_FULL_SUPERVISION.value,
        AdmissionGrade.DYNAMICS_PRETRAIN_ONLY.value,
        AdmissionGrade.ACTUATOR_EFFECT_ONLY.value,
    }
    mask = classified_df["grade"].isin(allowed_grades)
    subset = classified_df.loc[mask].copy()

    schema = _actuator_effect_schema()
    if subset.empty:
        pd.DataFrame(columns=schema).to_parquet(output_path, index=False)
        logger.info("actuator_effect: 0 samples")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    frozen_ids = list(PFV_CORE_8_IDS)
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "sample_id": _safe_get(row, "sample_id", ""),
            "sample_idx": _safe_get(row, "sample_idx"),
            "event_id": _safe_get(row, "event_id", ""),
            "checkpoint_id": _safe_get(row, "checkpoint_id", ""),
            "state_key": _safe_get(row, "state_key", ""),
            "split": _safe_get(row, "split", ""),
            "source_round": _safe_get(row, "source_round", ""),
            "grade": _safe_get(row, "grade", ""),
            "priority_contract_id": _PRIORITY_CONTRACT_ID,
            "actual_schedule_sha": _safe_get(row, "actual_schedule_sha", ""),
            "actual_actions": _serialize_array(
                _safe_get(row, "candidate_action_seq")),
            "candidate_action_seq": _serialize_array(
                _safe_get(row, "candidate_action_seq")),
            "trajectory_candidate": _serialize_array(
                _safe_get(row, "trajectory_depth_candidate")),
            "trajectory_no_control": _serialize_array(
                _safe_get(row, "trajectory_depth_no_control")),
            "pfv_delta": _safe_get(row, "pfv_delta"),
            "tfv_delta": _safe_get(row, "tfv_delta"),
            "peak_delta": _safe_get(row, "peak_delta"),
            "priority_node_ids": frozen_ids,
        })

    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("actuator_effect: %d samples", len(df))
    return len(df)


def build_pfv_constraint_core8(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build pfv_constraint_core8.parquet。

    仅使用正式 8 节点 PFV，包含安全/边界/不安全支撑。

    Returns
    -------
    int
        样本数。
    """
    # 所有非 REJECT / RESERVED 的样本都可用于 PFV 约束
    excluded = {
        AdmissionGrade.REJECT.value,
        AdmissionGrade.RESERVED_EVALUATION.value,
    }
    mask = ~classified_df["grade"].isin(excluded)
    subset = classified_df.loc[mask].copy()

    schema = _pfv_constraint_schema()
    if subset.empty:
        pd.DataFrame(columns=schema).to_parquet(output_path, index=False)
        logger.info("pfv_constraint_core8: 0 samples")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    frozen_ids = list(PFV_CORE_8_IDS)
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "sample_id": _safe_get(row, "sample_id", ""),
            "sample_idx": _safe_get(row, "sample_idx"),
            "event_id": _safe_get(row, "event_id", ""),
            "checkpoint_id": _safe_get(row, "checkpoint_id", ""),
            "state_key": _safe_get(row, "state_key", ""),
            "split": _safe_get(row, "split", ""),
            "source_round": _safe_get(row, "source_round", ""),
            "grade": _safe_get(row, "grade", ""),
            "priority_contract_id": _PRIORITY_CONTRACT_ID,
            "pfv_delta": _safe_get(row, "pfv_delta"),
            "pfv_safe_label": _safe_get(row, "pfv_safe_label"),
            "trajectory_candidate_pfv_core8": _extract_pfv_core8(
                _safe_get(row, "trajectory_depth_candidate"), frozen_ids),
            "trajectory_no_control_pfv_core8": _extract_pfv_core8(
                _safe_get(row, "trajectory_depth_no_control"), frozen_ids),
            "priority_node_ids": frozen_ids,
        })

    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("pfv_constraint_core8: %d samples", len(df))
    return len(df)


def build_tfv_objective(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build tfv_objective.parquet。

    Candidate vs DI 完整 TFV 轨迹。

    Returns
    -------
    int
        样本数。
    """
    excluded = {
        AdmissionGrade.REJECT.value,
        AdmissionGrade.RESERVED_EVALUATION.value,
    }
    mask = ~classified_df["grade"].isin(excluded)
    subset = classified_df.loc[mask].copy()

    schema = _tfv_objective_schema()
    if subset.empty:
        pd.DataFrame(columns=schema).to_parquet(output_path, index=False)
        logger.info("tfv_objective: 0 samples")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    frozen_ids = list(PFV_CORE_8_IDS)
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "sample_id": _safe_get(row, "sample_id", ""),
            "sample_idx": _safe_get(row, "sample_idx"),
            "event_id": _safe_get(row, "event_id", ""),
            "checkpoint_id": _safe_get(row, "checkpoint_id", ""),
            "state_key": _safe_get(row, "state_key", ""),
            "split": _safe_get(row, "split", ""),
            "source_round": _safe_get(row, "source_round", ""),
            "grade": _safe_get(row, "grade", ""),
            "priority_contract_id": _PRIORITY_CONTRACT_ID,
            "tfv_delta": _safe_get(row, "tfv_delta"),
            "tfv_improved_label": _safe_get(row, "tfv_improved_label"),
            "trajectory_candidate": _serialize_array(
                _safe_get(row, "trajectory_depth_candidate")),
            "trajectory_dynamic_internal": _serialize_array(
                _safe_get(row, "trajectory_depth_dynamic_internal")),
            "priority_node_ids": frozen_ids,
        })

    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("tfv_objective: %d samples", len(df))
    return len(df)


def build_peak_constraint(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build peak_constraint.parquet。

    Candidate vs DI 完整 flood-rate 轨迹。

    Returns
    -------
    int
        样本数。
    """
    excluded = {
        AdmissionGrade.REJECT.value,
        AdmissionGrade.RESERVED_EVALUATION.value,
    }
    mask = ~classified_df["grade"].isin(excluded)
    subset = classified_df.loc[mask].copy()

    schema = _peak_constraint_schema()
    if subset.empty:
        pd.DataFrame(columns=schema).to_parquet(output_path, index=False)
        logger.info("peak_constraint: 0 samples")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    frozen_ids = list(PFV_CORE_8_IDS)
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "sample_id": _safe_get(row, "sample_id", ""),
            "sample_idx": _safe_get(row, "sample_idx"),
            "event_id": _safe_get(row, "event_id", ""),
            "checkpoint_id": _safe_get(row, "checkpoint_id", ""),
            "state_key": _safe_get(row, "state_key", ""),
            "split": _safe_get(row, "split", ""),
            "source_round": _safe_get(row, "source_round", ""),
            "grade": _safe_get(row, "grade", ""),
            "priority_contract_id": _PRIORITY_CONTRACT_ID,
            "peak_delta": _safe_get(row, "peak_delta"),
            "peak_noninferior_label": _safe_get(row, "peak_noninferior_label"),
            "trajectory_candidate": _serialize_array(
                _safe_get(row, "trajectory_depth_candidate")),
            "trajectory_dynamic_internal": _serialize_array(
                _safe_get(row, "trajectory_depth_dynamic_internal")),
            "priority_node_ids": frozen_ids,
        })

    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("peak_constraint: %d samples", len(df))
    return len(df)


def build_within_state_ranking_pairs(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build within_state_ranking_pairs.parquet。

    同状态真实 Candidate 对，按 PFV → Peak → TFV 排序。

    Returns
    -------
    int
        对数。
    """
    schema = _ranking_pairs_schema()

    # 仅使用有 ranking 信息的样本（RANKING_ONLY 及以上）
    allowed_grades = {
        AdmissionGrade.TARGET_FULL_SUPERVISION.value,
        AdmissionGrade.TARGET_RECOMPUTABLE.value,
        AdmissionGrade.SOURCE_DWF_FULL_SUPERVISION.value,
        AdmissionGrade.DYNAMICS_PRETRAIN_ONLY.value,
        AdmissionGrade.ACTUATOR_EFFECT_ONLY.value,
        AdmissionGrade.RANKING_ONLY.value,
    }
    mask = classified_df["grade"].isin(allowed_grades)
    subset = classified_df.loc[mask].copy()

    if subset.empty or lineage_df.empty:
        pd.DataFrame(columns=schema).to_parquet(output_path, index=False)
        logger.info("within_state_ranking_pairs: 0 pairs")
        return 0

    # 合并 lineage 获取 state_key 和 KPI 列（位置对齐）
    n = len(subset)
    # 找到 subset 中各行在 lineage 中的位置
    # classified/lineage 都是按 sample_idx 排列的 1200 行
    kpi_cols = ["state_key", "pfv_delta", "tfv_delta", "peak_delta",
                "pfv_safe_label", "tfv_improved_label", "peak_noninferior_label"]
    lin_available = [c for c in kpi_cols if c in lineage_df.columns]
    if lin_available:
        # 使用 sample_idx 对齐
        lin_indexed = lineage_df.set_index("sample_idx")
        for c in lin_available:
            if c not in subset.columns:
                subset[c] = subset["sample_idx"].map(
                    lin_indexed[c].to_dict()
                )

    # 按 state_key 分组，生成同状态对
    pairs = []
    pair_id = 0
    for state_key, group in subset.groupby("state_key"):
        if len(group) < 2:
            continue
        group_sorted = group.sort_values(
            ["pfv_delta", "peak_delta", "tfv_delta"],
            ascending=[True, True, True],
            na_position="last",
        )
        items = group_sorted.to_dict("records")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                pairs.append({
                    "pair_id": pair_id,
                    "state_key": state_key,
                    "sample_id_a": _safe_get(a, "sample_id", ""),
                    "sample_id_b": _safe_get(b, "sample_id", ""),
                    "pfv_delta_a": _safe_get(a, "pfv_delta"),
                    "pfv_delta_b": _safe_get(b, "pfv_delta"),
                    "peak_delta_a": _safe_get(a, "peak_delta"),
                    "peak_delta_b": _safe_get(b, "peak_delta"),
                    "tfv_delta_a": _safe_get(a, "tfv_delta"),
                    "tfv_delta_b": _safe_get(b, "tfv_delta"),
                    "pfv_safe_a": _safe_get(a, "pfv_safe_label"),
                    "pfv_safe_b": _safe_get(b, "pfv_safe_label"),
                    "tfv_improved_a": _safe_get(a, "tfv_improved_label"),
                    "tfv_improved_b": _safe_get(b, "tfv_improved_label"),
                    "peak_noninferior_a": _safe_get(a, "peak_noninferior_label"),
                    "peak_noninferior_b": _safe_get(b, "peak_noninferior_label"),
                    "event_id": _safe_get(a, "event_id", ""),
                    "checkpoint_id": _safe_get(a, "checkpoint_id", ""),
                    "priority_contract_id": _PRIORITY_CONTRACT_ID,
                })
                pair_id += 1

    if pairs:
        df = pd.DataFrame(pairs, columns=schema)
    else:
        df = pd.DataFrame(columns=schema)

    df.to_parquet(output_path, index=False)
    logger.info("within_state_ranking_pairs: %d pairs", len(df))
    return len(df)


def build_consumed_development(
    classified_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build consumed_development.parquet。

    旧评估结果已被查看，允许作为开发数据。

    Returns
    -------
    int
        样本数。
    """
    grade = AdmissionGrade.CONSUMED_DEVELOPMENT.value
    mask = classified_df["grade"] == grade
    subset = classified_df.loc[mask].copy()

    schema = _base_schema_columns() + [
        "pfv_delta", "tfv_delta", "peak_delta",
        "pfv_safe_label", "tfv_improved_label", "peak_noninferior_label",
    ]
    if subset.empty:
        pd.DataFrame(columns=schema).to_parquet(output_path, index=False)
        logger.info("consumed_development: 0 samples")
        return 0

    merged = _merge_classified_with_manifest(subset, trajectory_manifest, lineage_df)
    frozen_ids = list(PFV_CORE_8_IDS)
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "sample_id": _safe_get(row, "sample_id", ""),
            "sample_idx": _safe_get(row, "sample_idx"),
            "event_id": _safe_get(row, "event_id", ""),
            "checkpoint_id": _safe_get(row, "checkpoint_id", ""),
            "state_key": _safe_get(row, "state_key", ""),
            "split": _safe_get(row, "split", ""),
            "source_round": _safe_get(row, "source_round", ""),
            "grade": _safe_get(row, "grade", ""),
            "priority_contract_id": _PRIORITY_CONTRACT_ID,
            "pfv_delta": _safe_get(row, "pfv_delta"),
            "tfv_delta": _safe_get(row, "tfv_delta"),
            "peak_delta": _safe_get(row, "peak_delta"),
            "pfv_safe_label": _safe_get(row, "pfv_safe_label"),
            "tfv_improved_label": _safe_get(row, "tfv_improved_label"),
            "peak_noninferior_label": _safe_get(row, "peak_noninferior_label"),
        })

    df = pd.DataFrame(rows, columns=schema)
    df.to_parquet(output_path, index=False)
    logger.info("consumed_development: %d samples", len(df))
    return len(df)


def build_reserved_evaluation_manifest(
    classified_df: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build reserved_evaluation_manifest.csv。

    严格隔离的 Calibration/Locked/Formal/Challenge 样本。

    Returns
    -------
    int
        样本数。
    """
    grade = AdmissionGrade.RESERVED_EVALUATION.value
    mask = classified_df["grade"] == grade
    subset = classified_df.loc[mask].copy()

    cols = ["sample_id", "sample_idx", "event_id", "checkpoint_id",
            "state_key", "split", "grade"]
    if subset.empty:
        pd.DataFrame(columns=cols).to_csv(output_path, index=False)
        logger.info("reserved_evaluation_manifest: 0 samples")
        return 0

    out = subset[[c for c in cols if c in subset.columns]].copy()
    out.to_csv(output_path, index=False)
    logger.info("reserved_evaluation_manifest: %d samples", len(out))
    return len(out)


def build_rejected_samples(
    classified_df: pd.DataFrame,
    output_path: Path,
) -> int:
    """Build rejected_samples.csv。

    逐样本拒绝原因。

    Returns
    -------
    int
        被拒绝样本数。
    """
    grade = AdmissionGrade.REJECT.value
    mask = classified_df["grade"] == grade
    subset = classified_df.loc[mask].copy()

    cols = ["sample_id", "sample_idx", "event_id", "checkpoint_id",
            "grade", "reason_codes", "details_json"]
    if subset.empty:
        pd.DataFrame(columns=cols).to_csv(output_path, index=False)
        logger.info("rejected_samples: 0 samples")
        return 0

    out = subset[[c for c in cols if c in subset.columns]].copy()
    out.to_csv(output_path, index=False)
    logger.info("rejected_samples: %d samples", len(out))
    return len(out)


def copy_sample_lineage(
    lineage_df: pd.DataFrame,
    output_path: Path,
) -> int:
    """复制 sample_lineage.parquet 到统一数据集目录。

    Returns
    -------
    int
        样本数。
    """
    if lineage_df.empty:
        pd.DataFrame(columns=["sample_idx", "event_id", "checkpoint_id"]).to_parquet(
            output_path, index=False
        )
        return 0

    lineage_df.to_parquet(output_path, index=False)
    logger.info("sample_lineage copy: %d samples", len(lineage_df))
    return len(lineage_df)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_final_unified_datasets(
    project_root: Path,
    output_root: Path,
    classification_path: Path | None = None,
) -> dict:
    """构建全部 12 个最终任务数据集。

    Parameters
    ----------
    project_root : Path
        项目根目录。
    output_root : Path
        输出根目录。
    classification_path : Path | None
        分类结果路径。若 None 则使用默认位置
        ``audits/v42_final_pool/sample_classification.parquet``。

    Returns
    -------
    dict
        每个数据集的样本数和 schema 信息。
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # 输出目录
    out_dir = output_root / "data" / _OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 加载上游数据
    # ------------------------------------------------------------------
    # 分类结果
    if classification_path is None:
        classification_path = project_root / _AUDIT_DIR / "sample_classification.parquet"
    classified_df = pd.read_parquet(classification_path)
    logger.info("Loaded classification: %d samples", len(classified_df))

    # Lineage
    lineage_df = _load_audit_parquet(project_root, "sample_lineage.parquet")
    logger.info("Loaded lineage: %d samples", len(lineage_df))

    # Trajectory manifest
    try:
        trajectory_manifest = _load_trajectory_manifest(project_root)
        logger.info("Loaded trajectory manifest: %d samples", len(trajectory_manifest))
    except FileNotFoundError:
        logger.warning("Trajectory manifest not found; using empty manifest")
        trajectory_manifest = pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. 构建各数据集
    # ------------------------------------------------------------------
    manifest_records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    # 1) target_no_dwf_full_supervision
    p1 = out_dir / "target_no_dwf_full_supervision.parquet"
    c1 = build_target_full_supervision(
        classified_df, lineage_df, trajectory_manifest, p1)
    counts["target_no_dwf_full_supervision"] = c1
    manifest_records.append({
        "dataset": "target_no_dwf_full_supervision",
        "path": str(p1.relative_to(output_root)),
        "sample_count": c1,
        "schema": _full_supervision_schema(),
        "grade_filter": AdmissionGrade.TARGET_FULL_SUPERVISION.value,
    })

    # 2) source_dwf_full_supervision
    p2 = out_dir / "source_dwf_full_supervision.parquet"
    c2 = build_source_dwf_full_supervision(
        classified_df, lineage_df, trajectory_manifest, p2)
    counts["source_dwf_full_supervision"] = c2
    manifest_records.append({
        "dataset": "source_dwf_full_supervision",
        "path": str(p2.relative_to(output_root)),
        "sample_count": c2,
        "schema": _full_supervision_schema(),
        "grade_filter": AdmissionGrade.SOURCE_DWF_FULL_SUPERVISION.value,
    })

    # 3) dynamics_pretrain
    p3 = out_dir / "dynamics_pretrain.parquet"
    c3 = build_dynamics_pretrain(
        classified_df, lineage_df, trajectory_manifest, p3)
    counts["dynamics_pretrain"] = c3
    manifest_records.append({
        "dataset": "dynamics_pretrain",
        "path": str(p3.relative_to(output_root)),
        "sample_count": c3,
        "schema": _dynamics_pretrain_schema(),
        "grade_filter": "DYNAMICS_PRETRAIN_ONLY and above",
    })

    # 4) actuator_effect
    p4 = out_dir / "actuator_effect.parquet"
    c4 = build_actuator_effect(
        classified_df, lineage_df, trajectory_manifest, p4)
    counts["actuator_effect"] = c4
    manifest_records.append({
        "dataset": "actuator_effect",
        "path": str(p4.relative_to(output_root)),
        "sample_count": c4,
        "schema": _actuator_effect_schema(),
        "grade_filter": "ACTUATOR_EFFECT_ONLY and above",
    })

    # 5) pfv_constraint_core8
    p5 = out_dir / "pfv_constraint_core8.parquet"
    c5 = build_pfv_constraint_core8(
        classified_df, lineage_df, trajectory_manifest, p5)
    counts["pfv_constraint_core8"] = c5
    manifest_records.append({
        "dataset": "pfv_constraint_core8",
        "path": str(p5.relative_to(output_root)),
        "sample_count": c5,
        "schema": _pfv_constraint_schema(),
        "grade_filter": "All non-REJECT non-RESERVED",
    })

    # 6) tfv_objective
    p6 = out_dir / "tfv_objective.parquet"
    c6 = build_tfv_objective(
        classified_df, lineage_df, trajectory_manifest, p6)
    counts["tfv_objective"] = c6
    manifest_records.append({
        "dataset": "tfv_objective",
        "path": str(p6.relative_to(output_root)),
        "sample_count": c6,
        "schema": _tfv_objective_schema(),
        "grade_filter": "All non-REJECT non-RESERVED",
    })

    # 7) peak_constraint
    p7 = out_dir / "peak_constraint.parquet"
    c7 = build_peak_constraint(
        classified_df, lineage_df, trajectory_manifest, p7)
    counts["peak_constraint"] = c7
    manifest_records.append({
        "dataset": "peak_constraint",
        "path": str(p7.relative_to(output_root)),
        "sample_count": c7,
        "schema": _peak_constraint_schema(),
        "grade_filter": "All non-REJECT non-RESERVED",
    })

    # 8) within_state_ranking_pairs
    p8 = out_dir / "within_state_ranking_pairs.parquet"
    c8 = build_within_state_ranking_pairs(
        classified_df, lineage_df, p8)
    counts["within_state_ranking_pairs"] = c8
    manifest_records.append({
        "dataset": "within_state_ranking_pairs",
        "path": str(p8.relative_to(output_root)),
        "sample_count": c8,
        "schema": _ranking_pairs_schema(),
        "grade_filter": "RANKING_ONLY and above",
    })

    # 9) consumed_development
    p9 = out_dir / "consumed_development.parquet"
    c9 = build_consumed_development(
        classified_df, lineage_df, trajectory_manifest, p9)
    counts["consumed_development"] = c9
    manifest_records.append({
        "dataset": "consumed_development",
        "path": str(p9.relative_to(output_root)),
        "sample_count": c9,
        "schema": _base_schema_columns() + [
            "pfv_delta", "tfv_delta", "peak_delta",
            "pfv_safe_label", "tfv_improved_label", "peak_noninferior_label",
        ],
        "grade_filter": AdmissionGrade.CONSUMED_DEVELOPMENT.value,
    })

    # 10) reserved_evaluation_manifest
    p10 = out_dir / "reserved_evaluation_manifest.csv"
    c10 = build_reserved_evaluation_manifest(classified_df, p10)
    counts["reserved_evaluation_manifest"] = c10
    manifest_records.append({
        "dataset": "reserved_evaluation_manifest",
        "path": str(p10.relative_to(output_root)),
        "sample_count": c10,
        "schema": ["sample_id", "sample_idx", "event_id", "checkpoint_id",
                    "state_key", "split", "grade"],
        "grade_filter": AdmissionGrade.RESERVED_EVALUATION.value,
    })

    # 11) rejected_samples
    p11 = out_dir / "rejected_samples.csv"
    c11 = build_rejected_samples(classified_df, p11)
    counts["rejected_samples"] = c11
    manifest_records.append({
        "dataset": "rejected_samples",
        "path": str(p11.relative_to(output_root)),
        "sample_count": c11,
        "schema": ["sample_id", "sample_idx", "event_id", "checkpoint_id",
                    "grade", "reason_codes", "details_json"],
        "grade_filter": AdmissionGrade.REJECT.value,
    })

    # 12) sample_lineage
    p12 = out_dir / "sample_lineage.parquet"
    c12 = copy_sample_lineage(lineage_df, p12)
    counts["sample_lineage"] = c12
    manifest_records.append({
        "dataset": "sample_lineage",
        "path": str(p12.relative_to(output_root)),
        "sample_count": c12,
        "schema": list(lineage_df.columns) if not lineage_df.empty else [],
        "grade_filter": "ALL",
    })

    # ------------------------------------------------------------------
    # 3. 写出 dataset_manifest.json
    # ------------------------------------------------------------------
    dataset_manifest = {
        "project": "Project6 V4.2",
        "contract_id": _PRIORITY_CONTRACT_ID,
        "priority_node_ids": list(PFV_CORE_8_IDS),
        "n_history_frames": _N_HISTORY_FRAMES,
        "n_horizon_steps": _N_HORIZON_STEPS,
        "n_facilities": _N_FACILITIES,
        "total_classified_samples": len(classified_df),
        "grade_distribution": classified_df["grade"].value_counts().to_dict()
            if "grade" in classified_df.columns else {},
        "datasets": manifest_records,
        "per_dataset_counts": counts,
    }

    manifest_json_path = out_dir / "dataset_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote dataset_manifest.json → %s", manifest_json_path)

    logger.info(
        "Final unified datasets built: %d datasets → %s\nCounts: %s",
        len(manifest_records), out_dir, counts,
    )

    return dataset_manifest


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

    result = build_final_unified_datasets(project_root, project_root)
    print(json.dumps(
        {k: v for k, v in result.items() if k != "datasets"},
        indent=2, ensure_ascii=False,
    ))
    print(f"\nDataset details:")
    for ds in result.get("datasets", []):
        print(f"  {ds['dataset']}: {ds['sample_count']} samples → {ds['path']}")


if __name__ == "__main__":
    main()
