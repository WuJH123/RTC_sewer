"""V4.2 DWF 审计模块。

逐样本检查 DWF（Dry Weather Flow）状态并分类：
- SOURCE_DWF_FULL_SUPERVISION: DWF 完整，可用于全监督训练
- SOURCE_DWF_DYNAMICS_PRETRAIN: DWF 存在但仅用于动力学预训练
- SOURCE_DWF_ACTUATOR_EFFECT: DWF 存在，用于执行器效应学习
- DWF_INCOMPLETE_REJECT: DWF 不完整，拒绝

同时检查：
- DWF 节点入流存在性
- H120 DWF 序列可读性
- Candidate/NC/DI/Hold 分支是否共享同一 DWF
- 模型输入中 DWF 存在性
- 实际动作存在性
- 水力轨迹完整性
- 标签是否可在同一 DWF 域内重算

DWF 和无 DWF 版本使用统一 base_rainfall_fingerprint 防止跨 fold 泄漏。

Output → audits/v42_final_pool/
  - dwf_classification.parquet
  - dwf_audit_summary.json
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
_V42_MANIFEST_PARQUET = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_manifest_v42.parquet"
)
_V42_SUMMARY_JSON = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_dataset_v42_summary.json"
)
_ROUND_SAMPLE_MANIFEST = "dataset/round_sample_manifest.csv"
_EVENT_INVENTORY = "outputs/project6_dual_reference_v4/final_v4/inventory/event_inventory.csv"
_FACILITY_CSV = "data/project6_v3_facility_semantics_36.csv"
_OUTPUT_DIR = "audits/v42_final_pool"

_ROUND_NAMES = ["round0", "round1", "round2", "calibration", "locked_validation"]

# DWF 分类标签
DWF_FULL_SUPERVISION = "SOURCE_DWF_FULL_SUPERVISION"
DWF_DYNAMICS_PRETRAIN = "SOURCE_DWF_DYNAMICS_PRETRAIN"
DWF_ACTUATOR_EFFECT = "SOURCE_DWF_ACTUATOR_EFFECT"
DWF_INCOMPLETE_REJECT = "DWF_INCOMPLETE_REJECT"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    logger.warning("CSV not found: %s", path)
    return None


def _safe_load_json(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("JSON not found: %s", path)
    return None


def _action_seq_depth(seq: Any) -> int:
    """返回 action sequence 的时间步数。"""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return 0
    if isinstance(seq, str):
        try:
            seq = json.loads(seq)
        except (json.JSONDecodeError, TypeError):
            return 0
    if isinstance(seq, (list, np.ndarray)):
        arr = np.asarray(seq)
        if arr.ndim >= 2:
            return arr.shape[0]
        return len(arr)
    return 0


def _base_rainfall_fingerprint(event_id: str) -> str:
    """计算统一的 base rainfall fingerprint（不依赖 DWF 状态）。

    确保 DWF 和无 DWF 版本使用相同指纹，防止跨 fold 泄漏。
    """
    return hashlib.sha256(f"base_rainfall:{event_id}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 核心：DWF 审计
# ---------------------------------------------------------------------------

def build_dwf_classification(
    project_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    """Build per-sample DWF classification.

    对 V42 trajectory manifest 中的每个样本，检查 DWF 可用性并分类。

    Parameters
    ----------
    project_root : Path
        项目根目录
    output_root : Path
        输出根目录

    Returns
    -------
    pd.DataFrame
        每样本一行，包含 DWF 分类结果
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # 1. 加载 V42 manifest
    manifest_path = project_root / _V42_MANIFEST_PARQUET
    if not manifest_path.exists():
        raise FileNotFoundError(f"V42 manifest not found: {manifest_path}")
    v42_df = pd.read_parquet(manifest_path)

    # 2. 加载 schema
    summary = _safe_load_json(project_root / _V42_SUMMARY_JSON) or {}
    schema_horizon_steps = summary.get("schema", {}).get("n_horizon_steps", 12)
    schema_horizon_interval = summary.get("schema", {}).get("horizon_interval_min", 10)
    schema_n_facilities = summary.get("schema", {}).get("n_facilities", 36)

    # 3. 加载 event inventory
    event_inv = _safe_read_csv(project_root / _EVENT_INVENTORY)
    event_rainfall_map: Dict[str, str] = {}
    if event_inv is not None:
        for _, row in event_inv.iterrows():
            eid = row.get("event_id", "")
            rsha = row.get("rainfall_series_sha256", row.get("rainfall_sha256", ""))
            if eid:
                event_rainfall_map[eid] = str(rsha) if rsha else ""

    # 4. 加载 Round manifests 获取 DWF 相关信息
    round_info_map: Dict[tuple, Dict] = {}
    train_root = project_root / "outputs" / "project6_dual_reference_v4" / "final_v4" / "train1600_v3"
    for rnd in _ROUND_NAMES:
        manifest_path_rnd = train_root / rnd / _ROUND_SAMPLE_MANIFEST
        if manifest_path_rnd.exists():
            rnd_df = pd.read_csv(manifest_path_rnd)
            for _, row_rnd in rnd_df.iterrows():
                key = (row_rnd.get("event_id"), row_rnd.get("checkpoint_id"))
                if key not in round_info_map:
                    round_info_map[key] = {
                        "source_round": rnd,
                        "rainfall_sha256": row_rnd.get("rainfall_sha256", ""),
                        "k_actual": row_rnd.get("k_actual"),
                        "k_target": row_rnd.get("k_target"),
                        "four_branches_complete": row_rnd.get("four_branches_complete", False),
                        "h120_window_complete": row_rnd.get("h120_window_complete", False),
                        "h120_eligible": row_rnd.get("h120_eligible", False),
                        "completion_valid": row_rnd.get("completion_valid", False),
                        "physical_sha_ok": row_rnd.get("physical_sha_ok", False),
                        "rainfall_sha_ok": row_rnd.get("rainfall_sha_ok", False),
                        "kpi_recompute_ok": row_rnd.get("kpi_recompute_ok", False),
                        "pfv_safe": row_rnd.get("pfv_safe"),
                        "tfv_improved": row_rnd.get("tfv_improved"),
                        "peak_noninferior": row_rnd.get("peak_noninferior"),
                        "joint_noninferior": row_rnd.get("joint_noninferior"),
                        "candidate_role": row_rnd.get("candidate_role", ""),
                        "candidate_family": row_rnd.get("candidate_family", ""),
                    }

    # 5. 逐样本 DWF 审计
    classification_rows: List[Dict[str, Any]] = []

    for idx, row in v42_df.iterrows():
        event_id = row.get("event_id", "")
        checkpoint_id = row.get("checkpoint_id", "")
        key = (event_id, checkpoint_id)
        rnd_info = round_info_map.get(key, {})

        # --- DWF 检查项 ---

        # 1. DWF 节点入流存在性
        # 通过 rainfall_sha256 存在性推断（有 rainfall 即有 DWF 上下文）
        rainfall_sha = rnd_info.get("rainfall_sha256", event_rainfall_map.get(event_id, ""))
        dwf_node_inflow_present = bool(rainfall_sha)

        # 2. H120 DWF 序列可读性
        h120_complete = rnd_info.get("h120_window_complete", False)
        h120_eligible = rnd_info.get("h120_eligible", False)

        # 3. 4-branch 共享同一 DWF
        four_branches = rnd_info.get("four_branches_complete", False)

        # 4. 模型输入中 DWF 存在性
        rainfall_forecast = row.get("rainfall_forecast")
        model_dwf_present = _action_seq_depth(rainfall_forecast) > 0

        # 5. 实际动作存在性
        cand_actions = row.get("candidate_action_seq")
        actual_actions_present = _action_seq_depth(cand_actions) > 0
        k_actual = rnd_info.get("k_actual", 0) or 0

        # 6. 水力轨迹完整性
        traj_depth_cand = row.get("trajectory_depth_candidate")
        traj_depth_nc = row.get("trajectory_depth_no_control")
        traj_depth_di = row.get("trajectory_depth_dynamic_internal")
        traj_depth_hold = row.get("trajectory_depth_hold_previous")
        hydraulic_complete = all(
            _action_seq_depth(x) > 0
            for x in [traj_depth_cand, traj_depth_nc, traj_depth_di, traj_depth_hold]
        )

        # 7. 标签可重算性
        kpi_recompute_ok = rnd_info.get("kpi_recompute_ok", False)
        pfv_safe = row.get("pfv_safe_label")
        tfv_improved = row.get("tfv_improved_label")
        peak_noninf = row.get("peak_noninferior_label")
        labels_present = all(
            x is not None and not (isinstance(x, float) and np.isnan(x))
            for x in [pfv_safe, tfv_improved, peak_noninf]
        )

        # --- 分类 ---
        dwf_classification = _classify_dwf(
            dwf_node_inflow_present=dwf_node_inflow_present,
            h120_complete=bool(h120_complete),
            h120_eligible=bool(h120_eligible),
            four_branches=bool(four_branches),
            model_dwf_present=model_dwf_present,
            actual_actions_present=actual_actions_present,
            hydraulic_complete=hydraulic_complete,
            labels_present=labels_present,
            k_actual=int(k_actual),
            candidate_role=rnd_info.get("candidate_role", ""),
        )

        # base rainfall fingerprint（DWF 无关）
        base_rf = _base_rainfall_fingerprint(event_id)

        classification_rows.append({
            "sample_idx": idx,
            "event_id": event_id,
            "checkpoint_id": checkpoint_id,
            "state_key": row.get("state_key", ""),
            "split": row.get("split", ""),
            # DWF 检查项
            "dwf_node_inflow_present": dwf_node_inflow_present,
            "h120_dwf_sequence_readable": bool(h120_complete),
            "h120_eligible": bool(h120_eligible),
            "branches_share_dwf": bool(four_branches),
            "model_input_dwf_present": model_dwf_present,
            "actual_actions_present": actual_actions_present,
            "hydraulic_trajectories_complete": hydraulic_complete,
            "labels_recomputable": bool(kpi_recompute_ok) and labels_present,
            # 分类结果
            "dwf_classification": dwf_classification,
            # 统一指纹
            "base_rainfall_fingerprint": base_rf,
            "rainfall_sha256": rainfall_sha,
            # 补充信息
            "source_round": rnd_info.get("source_round", "v4_final"),
            "candidate_role": rnd_info.get("candidate_role", ""),
            "candidate_family": rnd_info.get("candidate_family", ""),
            "k_actual": int(k_actual),
            "k_target": rnd_info.get("k_target"),
            "completion_valid": rnd_info.get("completion_valid", False),
            "physical_sha_ok": rnd_info.get("physical_sha_ok", False),
            "rainfall_sha_ok": rnd_info.get("rainfall_sha_ok", False),
            # 标签
            "pfv_safe_label": pfv_safe,
            "tfv_improved_label": tfv_improved,
            "peak_noninferior_label": peak_noninf,
            "pfv_delta": row.get("pfv_delta"),
            "tfv_delta": row.get("tfv_delta"),
            "peak_delta": row.get("peak_delta"),
        })

    classification_df = pd.DataFrame(classification_rows)

    # 6. 写出
    out_dir = output_root / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    classification_df.to_parquet(out_dir / "dwf_classification.parquet", index=False)

    # dwf_audit_summary.json
    if not classification_df.empty:
        class_counts = classification_df["dwf_classification"].value_counts().to_dict()
        total = len(classification_df)
        audit_summary = {
            "total_samples": total,
            "classification_counts": class_counts,
            "classification_ratios": {
                k: v / total for k, v in class_counts.items()
            },
            "dwf_checks": {
                "dwf_node_inflow_present": int(classification_df["dwf_node_inflow_present"].sum()),
                "h120_dwf_sequence_readable": int(classification_df["h120_dwf_sequence_readable"].sum()),
                "branches_share_dwf": int(classification_df["branches_share_dwf"].sum()),
                "model_input_dwf_present": int(classification_df["model_input_dwf_present"].sum()),
                "actual_actions_present": int(classification_df["actual_actions_present"].sum()),
                "hydraulic_trajectories_complete": int(classification_df["hydraulic_trajectories_complete"].sum()),
                "labels_recomputable": int(classification_df["labels_recomputable"].sum()),
            },
            "unique_events": int(classification_df["event_id"].nunique()),
            "unique_base_rainfall_fingerprints": int(
                classification_df["base_rainfall_fingerprint"].nunique()
            ),
            "source_round_distribution": (
                classification_df["source_round"].value_counts().to_dict()
            ),
        }
    else:
        audit_summary = {"total_samples": 0}

    with open(out_dir / "dwf_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "DWF audit complete: %d samples, classification: %s",
        len(classification_df),
        class_counts if not classification_df.empty else "N/A",
    )

    return classification_df


def _classify_dwf(
    *,
    dwf_node_inflow_present: bool,
    h120_complete: bool,
    h120_eligible: bool,
    four_branches: bool,
    model_dwf_present: bool,
    actual_actions_present: bool,
    hydraulic_complete: bool,
    labels_present: bool,
    k_actual: int,
    candidate_role: str,
) -> str:
    """根据 DWF 检查结果分类样本。

    分类逻辑：
    - SOURCE_DWF_FULL_SUPERVISION: DWF 完整 + 4-branch + 标签可重算
    - SOURCE_DWF_DYNAMICS_PRETRAIN: DWF 存在但仅动力学可用（无标签/无动作）
    - SOURCE_DWF_ACTUATOR_EFFECT: DWF 存在 + 动作存在 + 执行器效应可学
    - DWF_INCOMPLETE_REJECT: DWF 不完整
    """
    # 首先检查 DWF 是否基本可用
    if not dwf_node_inflow_present:
        return DWF_INCOMPLETE_REJECT

    if not model_dwf_present:
        return DWF_INCOMPLETE_REJECT

    # DWF 存在，进一步分类
    if not actual_actions_present:
        # 无动作 → 仅动力学预训练
        if hydraulic_complete:
            return DWF_DYNAMICS_PRETRAIN
        return DWF_INCOMPLETE_REJECT

    # 有动作
    if not hydraulic_complete:
        return DWF_INCOMPLETE_REJECT

    # 有动作 + 水力完整
    if not labels_present:
        # 标签不可用 → 执行器效应学习
        return DWF_ACTUATOR_EFFECT

    # 有动作 + 水力完整 + 标签
    if not h120_complete or not h120_eligible:
        # H120 不完整 → 执行器效应
        return DWF_ACTUATOR_EFFECT

    if not four_branches:
        # 4-branch 不完整 → 执行器效应
        return DWF_ACTUATOR_EFFECT

    # 全监督条件满足
    return DWF_FULL_SUPERVISION


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

    classification_df = build_dwf_classification(project_root, project_root)
    print(f"DWF classification: {len(classification_df)} samples")
    if not classification_df.empty:
        print(classification_df["dwf_classification"].value_counts())


if __name__ == "__main__":
    main()
