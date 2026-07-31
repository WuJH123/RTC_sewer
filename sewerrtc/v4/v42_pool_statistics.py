"""V4.2 数据池综合统计报告模块。

从 ``audits/v42_final_pool/`` 和 ``data/v42_final_unified/`` 中聚合数据，
生成完整的数据池统计报告。

报告指标:
  - 原始文件数 / manifest 行数
  - 物理唯一样本数 / 去重数
  - 独立降雨数 / 事件数 / 状态数 / Candidate 数
  - DWF / no-DWF 拆分
  - 各任务池样本数
  - 各设施有效响应事件数
  - PFV safe / boundary / unsafe 计数
  - TFV improved / degraded 计数
  - Peak safe / degraded 计数
  - 有效排序对数
  - 每状态 joint-safe-improved Candidate 数
  - ICC（组内相关系数）
  - 有效样本量

Output → audits/v42_final_pool/pool_statistics.json
       → audits/v42_final_pool/pool_statistics_summary.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sewerrtc._project_root import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_AUDIT_DIR = "audits/v42_final_pool"
_DATA_DIR = "data/v42_final_unified"
_V42_MANIFEST_PARQUET = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_dataset_v42.parquet"
)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _load_json(project_root: Path, rel_path: str) -> dict:
    """加载 JSON 文件，不存在则返回空 dict。"""
    p = project_root / rel_path
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("JSON not found: %s", p)
    return {}


def _load_parquet(project_root: Path, rel_path: str) -> pd.DataFrame:
    """加载 Parquet 文件，不存在则返回空 DataFrame。"""
    p = project_root / rel_path
    if p.exists():
        return pd.read_parquet(p)
    logger.warning("Parquet not found: %s", p)
    return pd.DataFrame()


def _load_csv(project_root: Path, rel_path: str) -> pd.DataFrame:
    """加载 CSV 文件，不存在则返回空 DataFrame。"""
    p = project_root / rel_path
    if p.exists():
        return pd.read_csv(p)
    logger.warning("CSV not found: %s", p)
    return pd.DataFrame()


def _count_files_in_dir(project_root: Path, rel_path: str, pattern: str = "*") -> int:
    """统计目录下匹配 pattern 的文件数。"""
    d = project_root / rel_path
    if d.exists() and d.is_dir():
        return len(list(d.glob(pattern)))
    return 0


def _compute_icc(values: np.ndarray, groups: np.ndarray) -> float:
    """计算单因素随机效应模型的 ICC(1,1)。

    ICC = (MSB - MSW) / (MSB + (k-1)*MSW)

    其中 MSB = 组间均方，MSW = 组内均方，k = 每组观测数。
    若组数 < 2 或方差为零则返回 0.0。
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)

    # 去除 NaN
    mask = np.isfinite(values)
    values = values[mask]
    groups = groups[mask]

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < 2:
        return 0.0

    # 计算组间/组内平方和
    grand_mean = values.mean()
    ss_between = 0.0
    ss_within = 0.0
    n_total = len(values)

    for g in unique_groups:
        g_mask = groups == g
        g_vals = values[g_mask]
        g_mean = g_vals.mean()
        n_k = len(g_vals)
        ss_between += n_k * (g_mean - grand_mean) ** 2
        ss_within += np.sum((g_vals - g_mean) ** 2)

    df_between = n_groups - 1
    df_within = n_total - n_groups

    if df_between == 0 or df_within == 0:
        return 0.0

    msb = ss_between / df_between
    msw = ss_within / df_within

    if msb + msw == 0:
        return 0.0

    # 平均组大小
    k_bar = n_total / n_groups
    icc = (msb - msw) / (msb + (k_bar - 1) * msw)
    return float(icc)


def _effective_sample_size(
    values: np.ndarray,
    groups: np.ndarray,
    icc: float,
) -> int:
    """基于 ICC 和设计效应计算有效样本量。

    ESS = N / (1 + (k_bar - 1) * ICC)

    其中 N = 总样本数，k_bar = 平均组大小。
    """
    n_total = len(values)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups == 0 or icc <= 0:
        return n_total

    k_bar = n_total / n_groups
    design_effect = 1.0 + (k_bar - 1.0) * max(icc, 0.0)
    if design_effect <= 0:
        return n_total
    return int(np.floor(n_total / design_effect))


# ---------------------------------------------------------------------------
# 核心 API
# ---------------------------------------------------------------------------

def compute_pool_statistics(
    project_root: Path,
    output_root: Path,
) -> dict:
    """Compute comprehensive pool statistics. Returns full statistics dict.

    Parameters
    ----------
    project_root : Path
        项目根目录。
    output_root : Path
        输出根目录。

    Returns
    -------
    dict
        完整统计字典，包含所有指标。
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    stats: Dict[str, Any] = {}

    # ==================================================================
    # 1. 原始文件与 manifest 统计
    # ==================================================================
    # 原始 trajectory 文件数
    original_file_count = _count_files_in_dir(
        project_root,
        "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset",
        "*.parquet",
    )
    # 如果找不到 trajectory 目录，尝试从 manifest 推断
    if original_file_count == 0:
        original_file_count = 1  # 至少有 1 个主 manifest

    stats["original_file_count"] = original_file_count

    # 原始 manifest 行数
    manifest_df = _load_parquet(project_root, _V42_MANIFEST_PARQUET)
    if manifest_df.empty:
        # 尝试 test_small_manifest
        manifest_df = _load_parquet(
            project_root,
            "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
            "_test_small_manifest.parquet",
        )
    original_manifest_row_count = len(manifest_df) if not manifest_df.empty else 0
    stats["original_manifest_row_count"] = original_manifest_row_count

    # ==================================================================
    # 2. 去重统计
    # ==================================================================
    dedup_audit = _load_json(project_root, f"{_AUDIT_DIR}/deduplication_audit.json")
    total_samples = dedup_audit.get("total_samples", 0)
    unique_physical = dedup_audit.get("unique_physical_samples", 0)
    dup_groups = dedup_audit.get("duplicate_group_count", 0)
    dup_count = dedup_audit.get("duplicate_sample_count", 0)

    stats["total_samples"] = total_samples
    stats["physical_unique_sample_count"] = unique_physical
    stats["duplicate_removal_count"] = dup_count
    stats["duplicate_group_count"] = dup_groups

    # ==================================================================
    # 3. 加载 lineage 获取详细统计
    # ==================================================================
    lineage = _load_parquet(project_root, f"{_DATA_DIR}/sample_lineage.parquet")
    if lineage.empty:
        lineage = _load_parquet(project_root, f"{_AUDIT_DIR}/sample_lineage.parquet")

    classification = _load_parquet(
        project_root, f"{_AUDIT_DIR}/sample_classification.parquet"
    )

    # ==================================================================
    # 4. 独立降雨数 / 事件数 / 状态数 / Candidate 数
    # ==================================================================
    if not lineage.empty:
        # 独立降雨数
        if "rainfall_fingerprint" in lineage.columns:
            stats["independent_rainfall_count"] = int(
                lineage["rainfall_fingerprint"].nunique()
            )
        else:
            stats["independent_rainfall_count"] = 0

        # 事件数
        if "event_id" in lineage.columns:
            stats["event_count"] = int(lineage["event_id"].nunique())
        else:
            stats["event_count"] = 0

        # 状态数
        if "state_key" in lineage.columns:
            stats["state_count"] = int(lineage["state_key"].nunique())
        else:
            stats["state_count"] = 0

        # Candidate 数（唯一 candidate_id）
        if "candidate_id" in lineage.columns:
            stats["candidate_count"] = int(lineage["candidate_id"].nunique())
        elif "candidate_family" in lineage.columns:
            stats["candidate_count"] = int(lineage["candidate_family"].nunique())
        else:
            stats["candidate_count"] = 0

        # Candidate family 分布
        if "candidate_family" in lineage.columns:
            stats["candidate_family_distribution"] = (
                lineage["candidate_family"].value_counts().to_dict()
            )
        else:
            stats["candidate_family_distribution"] = {}

        # 候选角色分布
        if "candidate_role" in lineage.columns:
            stats["candidate_role_distribution"] = (
                lineage["candidate_role"].value_counts().to_dict()
            )
        else:
            stats["candidate_role_distribution"] = {}
    else:
        # 从 classification summary 获取事件数
        cls_summary = _load_json(
            project_root, f"{_AUDIT_DIR}/sample_classification_summary.json"
        )
        event_counts = cls_summary.get("event_counts", {})
        stats["event_count"] = len(event_counts)
        stats["independent_rainfall_count"] = 0
        stats["state_count"] = 0
        stats["candidate_count"] = 0
        stats["candidate_family_distribution"] = {}
        stats["candidate_role_distribution"] = {}

    # ==================================================================
    # 5. DWF / no-DWF 拆分
    # ==================================================================
    dwf_audit = _load_json(project_root, f"{_AUDIT_DIR}/dwf_audit_summary.json")
    dwf_total = dwf_audit.get("total_samples", 0)
    dwf_class = dwf_audit.get("classification_counts", {})
    stats["dwf_split"] = {
        "total_samples": dwf_total,
        "classification_counts": dwf_class,
        "unique_events": dwf_audit.get("unique_events", 0),
        "unique_base_rainfall_fingerprints": dwf_audit.get(
            "unique_base_rainfall_fingerprints", 0
        ),
    }

    # 从 lineage 中计算 DWF flag（如果有）
    if not lineage.empty and "candidate_role" in lineage.columns:
        role_counts = lineage["candidate_role"].value_counts().to_dict()
        stats["dwf_split"]["candidate_role_counts"] = role_counts

    # ==================================================================
    # 6. 各任务池样本数
    # ==================================================================
    dataset_manifest = _load_json(
        project_root, f"{_DATA_DIR}/dataset_manifest.json"
    )
    per_dataset_counts = dataset_manifest.get("per_dataset_counts", {})
    stats["per_task_pool_sample_count"] = per_dataset_counts

    # ==================================================================
    # 7. PFV / TFV / Peak 标签统计
    # ==================================================================
    if not lineage.empty:
        # PFV safe / boundary / unsafe
        if "pfv_safe_label" in lineage.columns:
            pfv_counts = lineage["pfv_safe_label"].value_counts().to_dict()
            stats["pfv_label_counts"] = {
                "safe": int(pfv_counts.get(True, 0) + pfv_counts.get("True", 0)),
                "boundary": int(pfv_counts.get("boundary", 0)),
                "unsafe": int(pfv_counts.get(False, 0) + pfv_counts.get("False", 0)),
                "raw_distribution": {str(k): int(v) for k, v in pfv_counts.items()},
            }
        else:
            stats["pfv_label_counts"] = {"safe": 0, "boundary": 0, "unsafe": 0}

        # TFV improved / degraded
        if "tfv_improved_label" in lineage.columns:
            tfv_counts = lineage["tfv_improved_label"].value_counts().to_dict()
            stats["tfv_label_counts"] = {
                "improved": int(tfv_counts.get(True, 0) + tfv_counts.get("True", 0)),
                "degraded": int(tfv_counts.get(False, 0) + tfv_counts.get("False", 0)),
                "raw_distribution": {str(k): int(v) for k, v in tfv_counts.items()},
            }
        else:
            stats["tfv_label_counts"] = {"improved": 0, "degraded": 0}

        # Peak safe / degraded
        if "peak_noninferior_label" in lineage.columns:
            peak_counts = lineage["peak_noninferior_label"].value_counts().to_dict()
            stats["peak_label_counts"] = {
                "safe": int(peak_counts.get(True, 0) + peak_counts.get("True", 0)),
                "degraded": int(
                    peak_counts.get(False, 0) + peak_counts.get("False", 0)
                ),
                "raw_distribution": {str(k): int(v) for k, v in peak_counts.items()},
            }
        else:
            stats["peak_label_counts"] = {"safe": 0, "degraded": 0}
    else:
        stats["pfv_label_counts"] = {"safe": 0, "boundary": 0, "unsafe": 0}
        stats["tfv_label_counts"] = {"improved": 0, "degraded": 0}
        stats["peak_label_counts"] = {"safe": 0, "degraded": 0}

    # ==================================================================
    # 8. 有效排序对数
    # ==================================================================
    stats["valid_ranking_pair_count"] = per_dataset_counts.get(
        "within_state_ranking_pairs", 0
    )

    # ==================================================================
    # 9. 每状态 joint-safe-improved Candidate 数
    # ==================================================================
    if not lineage.empty and all(
        c in lineage.columns
        for c in ["state_key", "pfv_safe_label", "tfv_improved_label"]
    ):
        # joint-safe-improved: pfv_safe AND tfv_improved
        pfv_safe = lineage["pfv_safe_label"].isin([True, "True"])
        tfv_imp = lineage["tfv_improved_label"].isin([True, "True"])
        lineage["_joint_safe_improved"] = pfv_safe & tfv_imp

        per_state_joint = (
            lineage.groupby("state_key")["_joint_safe_improved"]
            .sum()
            .astype(int)
            .to_dict()
        )
        stats["per_state_joint_safe_improved_count"] = {
            str(k): int(v) for k, v in per_state_joint.items()
        }
        stats["n_states_with_joint_safe_improved"] = int(
            sum(1 for v in per_state_joint.values() if v > 0)
        )
        stats["total_joint_safe_improved"] = int((pfv_safe & tfv_imp).sum())
    else:
        stats["per_state_joint_safe_improved_count"] = {}
        stats["n_states_with_joint_safe_improved"] = 0
        stats["total_joint_safe_improved"] = 0

    # ==================================================================
    # 10. 各设施有效响应事件数
    # ==================================================================
    # 从 lineage 中按 candidate_family 统计每个设施的有效响应事件
    if not lineage.empty and "event_id" in lineage.columns:
        # 每个事件有 36 个设施，统计有有效响应（非零 delta）的事件数
        if "pfv_delta" in lineage.columns:
            # 按事件统计有有效 PFV 响应的事件数
            events_with_pfv_response = int(
                lineage.groupby("event_id")["pfv_delta"]
                .apply(lambda x: (x.abs() > 0).any())
                .sum()
            )
            stats["events_with_effective_pfv_response"] = events_with_pfv_response
        else:
            stats["events_with_effective_pfv_response"] = 0

        # 按事件统计样本数
        if "event_id" in lineage.columns:
            event_sample_counts = (
                lineage["event_id"].value_counts().to_dict()
            )
            stats["per_event_sample_count"] = {
                str(k): int(v) for k, v in event_sample_counts.items()
            }
        else:
            stats["per_event_sample_count"] = {}
    else:
        stats["events_with_effective_pfv_response"] = 0
        stats["per_event_sample_count"] = {}

    # ==================================================================
    # 11. ICC（组内相关系数）
    # ==================================================================
    if not lineage.empty and "pfv_delta" in lineage.columns:
        if "state_key" in lineage.columns:
            pfv_vals = lineage["pfv_delta"].values
            state_groups = lineage["state_key"].values
            icc_val = _compute_icc(pfv_vals, state_groups)
            stats["icc_pfv_delta_by_state"] = round(icc_val, 6)
        else:
            stats["icc_pfv_delta_by_state"] = 0.0

        if "event_id" in lineage.columns:
            pfv_vals = lineage["pfv_delta"].values
            event_groups = lineage["event_id"].values
            icc_event = _compute_icc(pfv_vals, event_groups)
            stats["icc_pfv_delta_by_event"] = round(icc_event, 6)
        else:
            stats["icc_pfv_delta_by_event"] = 0.0
    else:
        stats["icc_pfv_delta_by_state"] = 0.0
        stats["icc_pfv_delta_by_event"] = 0.0

    # TFV ICC
    if not lineage.empty and "tfv_delta" in lineage.columns:
        if "state_key" in lineage.columns:
            tfv_vals = lineage["tfv_delta"].values
            state_groups = lineage["state_key"].values
            stats["icc_tfv_delta_by_state"] = round(
                _compute_icc(tfv_vals, state_groups), 6
            )
        else:
            stats["icc_tfv_delta_by_state"] = 0.0
    else:
        stats["icc_tfv_delta_by_state"] = 0.0

    # ==================================================================
    # 12. 有效样本量
    # ==================================================================
    n_total = total_samples if total_samples > 0 else len(lineage)
    icc_state = stats.get("icc_pfv_delta_by_state", 0.0)

    if not lineage.empty and "state_key" in lineage.columns:
        state_vals = lineage["pfv_delta"].values if "pfv_delta" in lineage.columns else np.zeros(n_total)
        state_grps = lineage["state_key"].values
        stats["effective_sample_size_by_state"] = _effective_sample_size(
            state_vals, state_grps, icc_state
        )
    else:
        stats["effective_sample_size_by_state"] = n_total

    icc_event = stats.get("icc_pfv_delta_by_event", 0.0)
    if not lineage.empty and "event_id" in lineage.columns:
        event_vals = lineage["pfv_delta"].values if "pfv_delta" in lineage.columns else np.zeros(n_total)
        event_grps = lineage["event_id"].values
        stats["effective_sample_size_by_event"] = _effective_sample_size(
            event_vals, event_grps, icc_event
        )
    else:
        stats["effective_sample_size_by_event"] = n_total

    # ==================================================================
    # 13. 等级分布
    # ==================================================================
    if not classification.empty and "grade" in classification.columns:
        stats["grade_distribution"] = (
            classification["grade"].value_counts().to_dict()
        )
    else:
        cls_summary = _load_json(
            project_root, f"{_AUDIT_DIR}/sample_classification_summary.json"
        )
        stats["grade_distribution"] = cls_summary.get("grade_counts", {})

    # ==================================================================
    # 14. 元信息
    # ==================================================================
    stats["metadata"] = {
        "project": "Project6 V4.2",
        "contract_id": dataset_manifest.get("contract_id", "PFV_CORE8_V1"),
        "n_facilities": dataset_manifest.get("n_facilities", 36),
        "n_history_frames": dataset_manifest.get("n_history_frames", 13),
        "n_horizon_steps": dataset_manifest.get("n_horizon_steps", 12),
        "priority_node_ids": dataset_manifest.get("priority_node_ids", []),
    }

    # ==================================================================
    # 写出结果
    # ==================================================================
    out_dir = output_root / _AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "pool_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote pool_statistics.json → %s", json_path)

    # CSV summary
    csv_path = out_dir / "pool_statistics_summary.csv"
    summary_rows = _flatten_stats_to_rows(stats)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)
    logger.info("Wrote pool_statistics_summary.csv → %s", csv_path)

    logger.info("Pool statistics complete: %d top-level keys", len(stats))
    return stats


def _flatten_stats_to_rows(stats: dict) -> List[Dict[str, Any]]:
    """将统计字典展平为 CSV 行列表。

    嵌套 dict 展平为 key.subkey = value 格式。
    """
    rows: List[Dict[str, Any]] = []

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)) and not isinstance(v, list):
                    _flatten(v, new_key)
                elif isinstance(v, list) and len(v) > 20:
                    # 长列表只记录长度
                    rows.append({"metric": new_key, "value": f"[list len={len(v)}]"})
                elif isinstance(v, list):
                    rows.append({"metric": new_key, "value": str(v)})
                else:
                    rows.append({"metric": new_key, "value": v})
        else:
            rows.append({"metric": prefix, "value": obj})

    # 跳过 metadata 和大型分布 dict 的深层展开
    top_level_skip = {"metadata", "per_state_joint_safe_improved_count",
                      "per_event_sample_count", "candidate_family_distribution"}
    for k, v in stats.items():
        if k in top_level_skip and isinstance(v, dict):
            rows.append({
                "metric": k,
                "value": f"[dict len={len(v)}]",
            })
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (dict, list)) and len(str(sv)) > 200:
                    rows.append({"metric": f"{k}.{sk}", "value": f"[complex len={len(sv) if isinstance(sv, (dict, list)) else '?'}]"})
                else:
                    rows.append({"metric": f"{k}.{sk}", "value": sv})
        else:
            rows.append({"metric": k, "value": v})

    return rows


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

    result = compute_pool_statistics(project_root, project_root)
    # 打印精简摘要
    for k, v in result.items():
        if isinstance(v, dict) and len(v) > 10:
            print(f"  {k}: [dict with {len(v)} entries]")
        elif isinstance(v, list) and len(v) > 10:
            print(f"  {k}: [list with {len(v)} items]")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
