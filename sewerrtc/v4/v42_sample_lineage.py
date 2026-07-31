"""V4.2 样本血缘追溯与物理去重模块。

追溯每个 V4 样本到其 Round0-5 来源，综合比较 rainfall / checkpoint / state /
action / trajectories / KPI 进行物理去重，防止跨 Round 重复计数。

Output → audits/v42_final_pool/
  - sample_lineage.parquet
  - physical_duplicate_groups.csv
  - canonical_sample_selection.csv
  - deduplication_audit.json
"""

from __future__ import annotations

import hashlib
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
_ROUND_NAMES = ["round0", "round1", "round2", "calibration", "locked_validation"]
_V42_MANIFEST_PARQUET = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_manifest_v42.parquet"
)
_V42_MANIFEST_CSV = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_manifest_v42.csv"
)
_ROUND_SAMPLE_MANIFEST = "dataset/round_sample_manifest.csv"
_EVENT_INVENTORY = "outputs/project6_dual_reference_v4/final_v4/inventory/event_inventory.csv"
_EVENT_USAGE_LEDGER = (
    "outputs/project6_dual_reference_v4/final_v4/inventory/event_usage_ledger.csv"
)
_OUTPUT_DIR = "audits/v42_final_pool"

# 物理去重列 — 8 维签名
_DEDUP_DIMS = [
    "rainfall_fingerprint",
    "checkpoint_timestamp",
    "prefix_state_hash",
    "actual_schedule_sha",
    "candidate_trajectory_sha",
    "ref_nc_trajectory_sha",
    "ref_di_trajectory_sha",
    "ref_hold_trajectory_sha",
    "kpi_label_tuple",
    "original_case_lineage",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _sha_series(values: List[float] | np.ndarray) -> str:
    """对浮点序列计算 SHA-256 指纹。"""
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _safe_sha(val: Any) -> str:
    """对任意值计算 SHA-256；None/NaN → empty string hash。"""
    if val is None:
        return hashlib.sha256(b"").hexdigest()
    if isinstance(val, float) and np.isnan(val):
        return hashlib.sha256(b"").hexdigest()
    raw = str(val).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _action_seq_sha(seq: Any) -> str:
    """对 action sequence（list-of-lists 或 JSON 字符串）计算 SHA。"""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return _sha_series([])
    if isinstance(seq, str):
        try:
            seq = json.loads(seq)
        except (json.JSONDecodeError, TypeError):
            return _safe_sha(seq)
    arr = np.asarray(seq, dtype=np.float64).ravel()
    return _sha_series(arr)


def _load_v42_manifest(project_root: Path) -> pd.DataFrame:
    """加载 V4.2 trajectory manifest。"""
    pq_path = project_root / _V42_MANIFEST_PARQUET
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    csv_path = project_root / _V42_MANIFEST_CSV
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"V42 manifest not found under {project_root}")


def _load_round_manifests(project_root: Path) -> Dict[str, pd.DataFrame]:
    """加载所有 Round 的 round_sample_manifest.csv。"""
    train_root = project_root / "outputs" / "project6_dual_reference_v4" / "final_v4" / "train1600_v3"
    result: Dict[str, pd.DataFrame] = {}
    for rnd in _ROUND_NAMES:
        manifest_path = train_root / rnd / _ROUND_SAMPLE_MANIFEST
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            result[rnd] = df
            logger.info("Loaded %s: %d samples", rnd, len(df))
        else:
            logger.warning("Round manifest missing: %s", manifest_path)
    return result


def _load_event_inventory(project_root: Path) -> Optional[pd.DataFrame]:
    p = project_root / _EVENT_INVENTORY
    if p.exists():
        return pd.read_csv(p)
    logger.warning("Event inventory not found: %s", p)
    return None


def _load_event_ledger(project_root: Path) -> Optional[pd.DataFrame]:
    p = project_root / _EVENT_USAGE_LEDGER
    if p.exists():
        return pd.read_csv(p)
    logger.warning("Event usage ledger not found: %s", p)
    return None


# ---------------------------------------------------------------------------
# 核心：构建样本血缘
# ---------------------------------------------------------------------------

def build_sample_lineage(
    project_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    """Build lineage for all samples across all manifests.

    对 V42 trajectory manifest 中的每个样本，追溯其来源 Round、原始 case_id、
    rainfall fingerprint、checkpoint 信息、action schedule、branch trajectory
    路径、label source 等完整血缘。

    Parameters
    ----------
    project_root : Path
        项目根目录 (E:\\RTC_sewer\\Project6)
    output_root : Path
        输出根目录（通常同 project_root）

    Returns
    -------
    pd.DataFrame
        每个样本一行，包含完整血缘字段
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # 1. 加载 V42 manifest
    v42_df = _load_v42_manifest(project_root)
    logger.info("V42 manifest: %d samples, %d unique events",
                len(v42_df), v42_df["event_id"].nunique())

    # 2. 加载 Round manifests 并构建查找索引
    round_manifests = _load_round_manifests(project_root)

    # 构建 (event_id, checkpoint_id) → round info 的索引
    round_index: Dict[tuple, Dict[str, Any]] = {}
    for rnd_name, rnd_df in round_manifests.items():
        for _, row in rnd_df.iterrows():
            key = (row.get("event_id"), row.get("checkpoint_id"))
            if key not in round_index:
                round_index[key] = {
                    "source_round": rnd_name,
                    "original_case_id": row.get("case_id", ""),
                    "candidate_family": row.get("candidate_family", ""),
                    "candidate_role": row.get("candidate_role", ""),
                    "source_anchor_id": row.get("source_anchor_id", ""),
                    "source_anchor_role": row.get("source_anchor_role", ""),
                    "k_actual": row.get("k_actual"),
                    "k_target": row.get("k_target"),
                    "actual_schedule_sha": row.get("actual_schedule_sha256", ""),
                    "checkpoint_state_sha256": row.get("checkpoint_state_sha256", ""),
                    "rainfall_sha256": row.get("rainfall_sha256", ""),
                    "network_sha256": row.get("network_sha256", ""),
                    "split": row.get("split", ""),
                    "status": row.get("status", ""),
                    "pfv_safe": row.get("pfv_safe"),
                    "tfv_improved": row.get("tfv_improved"),
                    "peak_noninferior": row.get("peak_noninferior"),
                }

    # 3. 加载 event inventory 获取 rainfall 指纹
    event_inv = _load_event_inventory(project_root)
    rainfall_map: Dict[str, str] = {}
    if event_inv is not None:
        for _, row in event_inv.iterrows():
            eid = row.get("event_id", "")
            rsha = row.get("rainfall_series_sha256", row.get("rainfall_sha256", ""))
            if eid and rsha:
                rainfall_map[eid] = str(rsha)

    # 4. 构建血缘 DataFrame
    lineage_rows: List[Dict[str, Any]] = []
    for idx, row in v42_df.iterrows():
        event_id = row.get("event_id", "")
        checkpoint_id = row.get("checkpoint_id", "")
        state_key = row.get("state_key", "")
        key = (event_id, checkpoint_id)

        # 从 Round 索引中查找来源
        round_info = round_index.get(key, {})
        source_round = round_info.get("source_round", "v4_final")
        original_case_id = round_info.get("original_case_id", state_key)

        # rainfall fingerprint
        rainfall_fp = rainfall_map.get(event_id, round_info.get("rainfall_sha256", ""))

        # action sequences SHA
        candidate_act_sha = _action_seq_sha(row.get("candidate_action_seq"))
        ref_nc_sha = _action_seq_sha(row.get("ref_no_control_action_seq"))
        ref_di_sha = _action_seq_sha(row.get("ref_dynamic_internal_action_seq"))
        ref_hold_sha = _action_seq_sha(row.get("ref_hold_previous_action_seq"))

        # KPI label tuple
        kpi_tuple = (
            int(row.get("pfv_safe_label", 0)),
            int(row.get("tfv_improved_label", 0)),
            int(row.get("peak_noninferior_label", 0)),
        )

        # 从 Round manifest 获取 trajectory SHA（如果有）
        candidate_traj_sha = round_info.get("actual_schedule_sha", "")

        lineage_rows.append({
            "sample_idx": idx,
            "event_id": event_id,
            "checkpoint_id": checkpoint_id,
            "state_key": state_key,
            "split": row.get("split", ""),
            "source_manifest": f"trajectory_manifest_v42.parquet",
            "source_round": source_round,
            "original_case_id": original_case_id,
            "rainfall_fingerprint": rainfall_fp,
            "checkpoint_timestamp": checkpoint_id,
            "prefix_state_hash": round_info.get("checkpoint_state_sha256", ""),
            "state_id": state_key,
            "candidate_id": original_case_id,
            "candidate_family": round_info.get("candidate_family", ""),
            "actual_schedule_sha": round_info.get("actual_schedule_sha", ""),
            "candidate_action_sha": candidate_act_sha,
            "ref_nc_action_sha": ref_nc_sha,
            "ref_di_action_sha": ref_di_sha,
            "ref_hold_action_sha": ref_hold_sha,
            "candidate_trajectory_sha": candidate_traj_sha,
            "ref_nc_trajectory_sha": ref_nc_sha,
            "ref_di_trajectory_sha": ref_di_sha,
            "ref_hold_trajectory_sha": ref_hold_sha,
            "kpi_label_tuple": str(kpi_tuple),
            "label_source": "v42_trajectory_manifest",
            "derived_manifest_ids": f"{source_round}:{original_case_id}",
            "pfv_delta": row.get("pfv_delta"),
            "tfv_delta": row.get("tfv_delta"),
            "peak_delta": row.get("peak_delta"),
            "pfv_safe_label": row.get("pfv_safe_label"),
            "tfv_improved_label": row.get("tfv_improved_label"),
            "peak_noninferior_label": row.get("peak_noninferior_label"),
            "network_sha256": round_info.get("network_sha256", ""),
            "candidate_role": round_info.get("candidate_role", ""),
            "source_anchor_id": round_info.get("source_anchor_id", ""),
            "k_actual": round_info.get("k_actual"),
            "k_target": round_info.get("k_target"),
        })

    lineage_df = pd.DataFrame(lineage_rows)
    logger.info("Built lineage for %d samples", len(lineage_df))

    # 5. 写出
    out_dir = output_root / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage_df.to_parquet(out_dir / "sample_lineage.parquet", index=False)
    logger.info("Wrote sample_lineage.parquet → %s", out_dir)

    return lineage_df


# ---------------------------------------------------------------------------
# 核心：物理去重审计
# ---------------------------------------------------------------------------

def audit_physical_deduplication(
    lineage_df: pd.DataFrame,
) -> dict:
    """Identify physical duplicates and select canonical samples.

    使用 8 维物理签名比较，而非单一文件 SHA。维度包括：
    1. rainfall real time series
    2. checkpoint timestamp
    3. numeric prefix state
    4. actual action sequence
    5. Candidate branch trajectory
    6. Reference branch trajectory (NC / DI / Hold)
    7. KPI labels
    8. Original case lineage

    Parameters
    ----------
    lineage_df : pd.DataFrame
        build_sample_lineage 的输出

    Returns
    -------
    dict
        去重审计摘要，包含：
        - total_samples
        - unique_physical_samples
        - duplicate_group_count
        - duplicate_sample_count
        - duplicate_groups (list of dicts)
    """
    if lineage_df.empty:
        return {
            "total_samples": 0,
            "unique_physical_samples": 0,
            "duplicate_group_count": 0,
            "duplicate_sample_count": 0,
            "duplicate_groups": [],
        }

    # 构建去重 key：所有 8 维签名组合
    dedup_cols = [
        "rainfall_fingerprint",
        "checkpoint_timestamp",
        "prefix_state_hash",
        "actual_schedule_sha",
        "candidate_trajectory_sha",
        "ref_nc_trajectory_sha",
        "ref_di_trajectory_sha",
        "ref_hold_trajectory_sha",
        "kpi_label_tuple",
        "original_case_lineage",
    ]

    # 对于缺失值用空字符串填充
    df = lineage_df.copy()
    for col in dedup_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # 计算物理签名复合 key
    df["physical_dedup_key"] = (
        df["rainfall_fingerprint"]
        + "|" + df["checkpoint_timestamp"]
        + "|" + df["prefix_state_hash"]
        + "|" + df["actual_schedule_sha"]
        + "|" + df["candidate_trajectory_sha"]
        + "|" + df["ref_nc_trajectory_sha"]
        + "|" + df["ref_di_trajectory_sha"]
        + "|" + df["ref_hold_trajectory_sha"]
        + "|" + df["kpi_label_tuple"]
        + "|" + df["derived_manifest_ids"]
    )

    # 分组找重复
    groups = df.groupby("physical_dedup_key")
    dup_groups = {k: v for k, v in groups if len(v) > 1}

    # 为每个重复组选择 canonical sample（优先选 source_round 最早的）
    round_priority = {
        "round0": 0, "round1": 1, "round2": 2,
        "calibration": 3, "locked_validation": 4, "v4_final": 5,
    }

    dup_group_records: List[Dict] = []
    canonical_indices: List[int] = []
    duplicate_indices: List[int] = []

    for dedup_key, group_df in dup_groups.items():
        # 选择 canonical：round 优先级最高，其次 sample_idx 最小
        sorted_group = group_df.copy()
        sorted_group["_priority"] = sorted_group["source_round"].map(
            lambda x: round_priority.get(x, 99)
        )
        sorted_group = sorted_group.sort_values(["_priority", "sample_idx"])
        canonical_idx = sorted_group.iloc[0]["sample_idx"]

        canonical_indices.append(int(canonical_idx))
        dup_indices_in_group = [
            int(i) for i in sorted_group["sample_idx"].tolist() if i != canonical_idx
        ]
        duplicate_indices.extend(dup_indices_in_group)

        dup_group_records.append({
            "dedup_key": dedup_key,
            "group_size": len(group_df),
            "canonical_sample_idx": int(canonical_idx),
            "duplicate_indices": str(dup_indices_in_group),
            "event_id": group_df.iloc[0]["event_id"],
            "checkpoint_id": group_df.iloc[0]["checkpoint_id"],
            "source_round": group_df.iloc[0]["source_round"],
        })

    # 标记 canonical / duplicate
    df["is_canonical"] = df["sample_idx"].isin(canonical_indices)
    df["is_duplicate"] = df["sample_idx"].isin(duplicate_indices)

    # 写出
    out_dir = Path(lineage_df.attrs.get("output_root", ".")) / _OUTPUT_DIR
    if not out_dir.exists():
        out_dir = Path("audits/v42_final_pool")
    out_dir.mkdir(parents=True, exist_ok=True)

    # physical_duplicate_groups.csv
    if dup_group_records:
        pd.DataFrame(dup_group_records).to_csv(
            out_dir / "physical_duplicate_groups.csv", index=False
        )
    else:
        pd.DataFrame(columns=["dedup_key", "group_size", "canonical_sample_idx",
                               "duplicate_indices", "event_id", "checkpoint_id",
                               "source_round"]).to_csv(
            out_dir / "physical_duplicate_groups.csv", index=False
        )

    # canonical_sample_selection.csv
    df[["sample_idx", "event_id", "checkpoint_id", "source_round",
        "original_case_id", "is_canonical", "is_duplicate"]].to_csv(
        out_dir / "canonical_sample_selection.csv", index=False
    )

    # deduplication_audit.json
    audit_summary = {
        "total_samples": len(df),
        "unique_physical_samples": len(df) - len(duplicate_indices),
        "duplicate_group_count": len(dup_groups),
        "duplicate_sample_count": len(duplicate_indices),
        "duplicate_groups": dup_group_records,
        "dedup_dimensions": _DEDUP_DIMS,
    }
    with open(out_dir / "deduplication_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Dedup audit: %d total → %d unique, %d duplicate groups, %d duplicates",
        len(df),
        audit_summary["unique_physical_samples"],
        audit_summary["duplicate_group_count"],
        audit_summary["duplicate_sample_count"],
    )

    return audit_summary


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

    lineage_df = build_sample_lineage(project_root, project_root)
    lineage_df.attrs["output_root"] = str(project_root)
    audit = audit_physical_deduplication(lineage_df)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
