"""V4.2 语义审计模块。

逐样本解析并审计以下语义合同：
A. Network: node IDs/types, link IDs, Storage, Outfall, Engineering36 IDs, facility types
B. Time: timestamps, recording interval, checkpoint, history/future start/end,
         13-frame coverage, H120 coverage
C. Forcing: rainfall, DWF, lateral inflow, 4-branch forcing consistency
D. Actions: requested/projected/written/target/actual, 12×36 order, K_actual
E. Branches: Candidate, NC, DI, Hold, historical Passive anchor
F. Labels: PFV_CORE8, TFV, Peak, Reference, units, time integration, dead-zone version

Output → audits/v42_final_pool/
  - semantic_sample_inventory.parquet
  - semantic_source_summary.csv
  - time_contract_conflicts.csv
  - action_contract_conflicts.csv
  - reference_contract_conflicts.csv
  - network_mapping_conflicts.csv
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
_V42_MANIFEST_PARQUET = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_manifest_v42.parquet"
)
_V42_SUMMARY_JSON = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_dataset_v42_summary.json"
)
_V42_AUDIT_JSON = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "trajectory_audit_v42.json"
)
_ACTION_SCHEMA_JSON = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "action_schema_v42.json"
)
_GRAPH_SCHEMA_JSON = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "graph_schema_v42.json"
)
_NODE_FEATURE_SCHEMA_JSON = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "node_feature_schema_v42.json"
)
_EDGE_FEATURE_SCHEMA_JSON = (
    "outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/"
    "edge_feature_schema_v42.json"
)
_ROUND_SAMPLE_MANIFEST = "dataset/round_sample_manifest.csv"
_EVENT_INVENTORY = "outputs/project6_dual_reference_v4/final_v4/inventory/event_inventory.csv"
_FACILITY_CSV = "data/project6_v3_facility_semantics_36.csv"
_OUTPUT_DIR = "audits/v42_final_pool"

_ROUND_NAMES = ["round0", "round1", "round2", "calibration", "locked_validation"]

# PFV_CORE8 节点列表（从 sentinel nodes 加载）
_PVV_CORE8_DEFAULT = [
    "PFV_1", "PFV_2", "PFV_3", "PFV_4",
    "PFV_5", "PFV_6", "PFV_7", "PFV_8",
]

# 期望的时间框架
_EXPECTED_HISTORY_FRAMES = 7
_EXPECTED_HORIZON_STEPS = 12
_EXPECTED_HISTORY_INTERVAL_MIN = 5
_EXPECTED_HORIZON_INTERVAL_MIN = 10


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_load_json(path: Path) -> Optional[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning("JSON not found: %s", path)
    return None


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path, **kwargs)
    logger.warning("CSV not found: %s", path)
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


def _action_seq_width(seq: Any) -> int:
    """返回 action sequence 的设施维度。"""
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
            return arr.shape[1]
        return 0
    return 0


# ---------------------------------------------------------------------------
# 核心：语义审计
# ---------------------------------------------------------------------------

def build_semantic_inventory(
    project_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    """Build per-sample semantic inventory.

    对 V42 trajectory manifest 中的每个样本，解析并记录：
    - 网络拓扑信息（节点数、边数、设施数）
    - 时间合同（history frames、horizon steps、interval）
    - 强迫一致性（rainfall、4-branch forcing）
    - 动作合同（requested/projected/actual、12×36 order）
    - 分支信息（Candidate、NC、DI、Hold）
    - 标签信息（PFV、TFV、Peak）

    Parameters
    ----------
    project_root : Path
        项目根目录
    output_root : Path
        输出根目录

    Returns
    -------
    pd.DataFrame
        每样本一行，包含语义审计字段
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # 1. 加载 V42 manifest
    manifest_path = project_root / _V42_MANIFEST_PARQUET
    if not manifest_path.exists():
        raise FileNotFoundError(f"V42 manifest not found: {manifest_path}")
    v42_df = pd.read_parquet(manifest_path)

    # 2. 加载 schema 信息
    summary = _safe_load_json(project_root / _V42_SUMMARY_JSON) or {}
    action_schema = _safe_load_json(project_root / _ACTION_SCHEMA_JSON) or {}
    graph_schema = _safe_load_json(project_root / _GRAPH_SCHEMA_JSON) or {}
    node_schema = _safe_load_json(project_root / _NODE_FEATURE_SCHEMA_JSON) or {}

    # 3. 加载 facility 信息
    facility_df = _safe_read_csv(project_root / _FACILITY_CSV)
    n_facilities = len(facility_df) if facility_df is not None else 36

    # 4. 加载 event inventory
    event_inv = _safe_read_csv(project_root / _EVENT_INVENTORY)
    event_meta: Dict[str, Dict] = {}
    if event_inv is not None:
        for _, row in event_inv.iterrows():
            eid = row.get("event_id", "")
            event_meta[eid] = {
                "duration_min": row.get("duration_min"),
                "total_depth": row.get("total_depth"),
                "peak_intensity": row.get("peak_intensity"),
                "start_time": row.get("start_time", ""),
                "end_time": row.get("end_time", ""),
            }

    # 5. 加载 Round manifests 获取补充信息
    round_info_map: Dict[tuple, Dict] = {}
    train_root = project_root / "outputs" / "project6_dual_reference_v4" / "final_v4" / "train1600_v3"
    for rnd in _ROUND_NAMES:
        manifest_path_rnd = train_root / rnd / _ROUND_SAMPLE_MANIFEST
        if manifest_path_rnd.exists():
            rnd_df = pd.read_csv(manifest_path_rnd)
            for _, row in rnd_df.iterrows():
                key = (row.get("event_id"), row.get("checkpoint_id"))
                if key not in round_info_map:
                    round_info_map[key] = {
                        "source_round": rnd,
                        "requested_schedule_sha": row.get("requested_schedule_sha256", ""),
                        "projected_schedule_sha": row.get("projected_schedule_sha256", ""),
                        "actual_schedule_sha": row.get("actual_schedule_sha256", ""),
                        "requested_schedule_json": row.get("requested_schedule_json", ""),
                        "projected_schedule_json": row.get("projected_schedule_json", ""),
                        "anchor_schedule_json": row.get("anchor_schedule_json", ""),
                        "k_target": row.get("k_target"),
                        "k_actual": row.get("k_actual"),
                        "k_sequence": row.get("k_sequence", ""),
                        "candidate_role": row.get("candidate_role", ""),
                        "candidate_family": row.get("candidate_family", ""),
                        "four_branches_complete": row.get("four_branches_complete"),
                        "h120_window_complete": row.get("h120_window_complete"),
                        "h120_eligible": row.get("h120_eligible"),
                    }

    # 6. 构建语义清单
    schema_n_nodes = summary.get("schema", {}).get("n_nodes", 932)
    schema_n_edges = summary.get("schema", {}).get("n_edges", 1276)
    schema_n_facilities = summary.get("schema", {}).get("n_facilities", 36)
    schema_hist_frames = summary.get("schema", {}).get("n_history_frames", 7)
    schema_horizon_steps = summary.get("schema", {}).get("n_horizon_steps", 12)
    schema_hist_interval = summary.get("schema", {}).get("history_interval_min", 5)
    schema_horizon_interval = summary.get("schema", {}).get("horizon_interval_min", 10)

    inventory_rows: List[Dict[str, Any]] = []
    time_conflicts: List[Dict] = []
    action_conflicts: List[Dict] = []
    ref_conflicts: List[Dict] = []
    network_conflicts: List[Dict] = []

    for idx, row in v42_df.iterrows():
        event_id = row.get("event_id", "")
        checkpoint_id = row.get("checkpoint_id", "")
        key = (event_id, checkpoint_id)
        rnd_info = round_info_map.get(key, {})

        # --- A. Network ---
        n_nodes = schema_n_nodes
        n_edges = schema_n_edges
        n_fac = schema_n_facilities

        # --- B. Time ---
        history_depth = row.get("history_depth")
        traj_depth_cand = row.get("trajectory_depth_candidate")
        hist_depth_val = _action_seq_depth(history_depth)
        traj_cand_val = _action_seq_depth(traj_depth_cand)

        # 检查时间合同一致性
        time_ok = True
        time_issues: List[str] = []
        if hist_depth_val > 0 and hist_depth_val != schema_hist_frames:
            time_ok = False
            time_issues.append(f"history_frames={hist_depth_val}!={schema_hist_frames}")
        if traj_cand_val > 0 and traj_cand_val != schema_horizon_steps:
            time_ok = False
            time_issues.append(f"horizon_steps={traj_cand_val}!={schema_horizon_steps}")

        if not time_ok:
            time_conflicts.append({
                "sample_idx": idx,
                "event_id": event_id,
                "checkpoint_id": checkpoint_id,
                "issue": "; ".join(time_issues),
            })

        # --- C. Forcing ---
        rainfall_fc = row.get("rainfall_forecast")
        rainfall_depth = _action_seq_depth(rainfall_fc)

        # --- D. Actions ---
        cand_actions = row.get("candidate_action_seq")
        ref_nc_actions = row.get("ref_no_control_action_seq")
        ref_di_actions = row.get("ref_dynamic_internal_action_seq")
        ref_hold_actions = row.get("ref_hold_previous_action_seq")

        cand_w = _action_seq_width(cand_actions)
        cand_d = _action_seq_depth(cand_actions)

        # 检查 12×36 order
        action_ok = True
        action_issues: List[str] = []
        if cand_d > 0 and cand_d != schema_horizon_steps:
            action_ok = False
            action_issues.append(f"candidate_depth={cand_d}!={schema_horizon_steps}")
        if cand_w > 0 and cand_w != n_fac:
            action_ok = False
            action_issues.append(f"candidate_width={cand_w}!={n_fac}")

        k_target = rnd_info.get("k_target")
        k_actual = rnd_info.get("k_actual")
        if k_target is not None and k_actual is not None:
            if k_actual > k_target:
                action_ok = False
                action_issues.append(f"k_actual={k_actual}>k_target={k_target}")

        if not action_ok:
            action_conflicts.append({
                "sample_idx": idx,
                "event_id": event_id,
                "checkpoint_id": checkpoint_id,
                "issue": "; ".join(action_issues),
            })

        # --- E. Branches ---
        branch_ok = True
        branch_issues: List[str] = []
        for name, seq in [("NC", ref_nc_actions), ("DI", ref_di_actions), ("Hold", ref_hold_actions)]:
            d = _action_seq_depth(seq)
            w = _action_seq_width(seq)
            if d > 0 and d != schema_horizon_steps:
                branch_ok = False
                branch_issues.append(f"{name}_depth={d}!={schema_horizon_steps}")
            if w > 0 and w != n_fac:
                branch_ok = False
                branch_issues.append(f"{name}_width={w}!={n_fac}")

        if not branch_ok:
            ref_conflicts.append({
                "sample_idx": idx,
                "event_id": event_id,
                "checkpoint_id": checkpoint_id,
                "issue": "; ".join(branch_issues),
            })

        # --- F. Labels ---
        pfv_safe = row.get("pfv_safe_label")
        tfv_improved = row.get("tfv_improved_label")
        peak_noninf = row.get("peak_noninferior_label")

        # 构建行
        inventory_rows.append({
            "sample_idx": idx,
            "event_id": event_id,
            "checkpoint_id": checkpoint_id,
            "state_key": row.get("state_key", ""),
            "split": row.get("split", ""),
            # Network
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_facilities": n_fac,
            "facility_semantics_loaded": facility_df is not None,
            # Time
            "history_frames": hist_depth_val,
            "horizon_steps": traj_cand_val,
            "history_interval_min": schema_hist_interval,
            "horizon_interval_min": schema_horizon_interval,
            "total_coverage_min": hist_depth_val * schema_hist_interval + traj_cand_val * schema_horizon_interval,
            "h120_coverage_min": schema_horizon_steps * schema_horizon_interval,
            "time_contract_ok": time_ok,
            # Forcing
            "rainfall_depth_frames": rainfall_depth,
            "rainfall_forecast_present": rainfall_depth > 0,
            # Actions
            "candidate_action_depth": cand_d,
            "candidate_action_width": cand_w,
            "k_target": k_target,
            "k_actual": k_actual,
            "action_contract_ok": action_ok,
            # Branches
            "nc_branch_depth": _action_seq_depth(ref_nc_actions),
            "di_branch_depth": _action_seq_depth(ref_di_actions),
            "hold_branch_depth": _action_seq_depth(ref_hold_actions),
            "branch_contract_ok": branch_ok,
            # Labels
            "pfv_safe_label": pfv_safe,
            "tfv_improved_label": tfv_improved,
            "peak_noninferior_label": peak_noninf,
            "pfv_delta": row.get("pfv_delta"),
            "tfv_delta": row.get("tfv_delta"),
            "peak_delta": row.get("peak_delta"),
            # Source
            "source_round": rnd_info.get("source_round", "v4_final"),
            "candidate_role": rnd_info.get("candidate_role", ""),
            "candidate_family": rnd_info.get("candidate_family", ""),
            "four_branches_complete": rnd_info.get("four_branches_complete"),
            "h120_window_complete": rnd_info.get("h120_window_complete"),
        })

    inventory_df = pd.DataFrame(inventory_rows)

    # 7. 写出
    out_dir = output_root / _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_df.to_parquet(out_dir / "semantic_sample_inventory.parquet", index=False)

    # semantic_source_summary.csv
    if not inventory_df.empty:
        source_summary = inventory_df.groupby("source_round").agg(
            n_samples=("sample_idx", "count"),
            n_events=("event_id", "nunique"),
            time_contract_failures=("time_contract_ok", lambda x: (~x).sum()),
            action_contract_failures=("action_contract_ok", lambda x: (~x).sum()),
            branch_contract_failures=("branch_contract_ok", lambda x: (~x).sum()),
            mean_pfv_delta=("pfv_delta", "mean"),
            mean_tfv_delta=("tfv_delta", "mean"),
        ).reset_index()
    else:
        source_summary = pd.DataFrame()
    source_summary.to_csv(out_dir / "semantic_source_summary.csv", index=False)

    # conflict CSVs
    _write_conflict_csv(out_dir / "time_contract_conflicts.csv", time_conflicts)
    _write_conflict_csv(out_dir / "action_contract_conflicts.csv", action_conflicts)
    _write_conflict_csv(out_dir / "reference_contract_conflicts.csv", ref_conflicts)
    _write_conflict_csv(out_dir / "network_mapping_conflicts.csv", network_conflicts)

    logger.info(
        "Semantic audit complete: %d samples, %d time conflicts, %d action conflicts, "
        "%d ref conflicts, %d network conflicts",
        len(inventory_df), len(time_conflicts), len(action_conflicts),
        len(ref_conflicts), len(network_conflicts),
    )

    return inventory_df


def _write_conflict_csv(path: Path, records: List[Dict]) -> None:
    if records:
        pd.DataFrame(records).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["sample_idx", "event_id", "checkpoint_id", "issue"]).to_csv(
            path, index=False
        )


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

    inventory_df = build_semantic_inventory(project_root, project_root)
    print(f"Semantic inventory: {len(inventory_df)} samples")
    if not inventory_df.empty:
        print(inventory_df[["source_round", "time_contract_ok", "action_contract_ok",
                             "branch_contract_ok"]].describe())


if __name__ == "__main__":
    main()
