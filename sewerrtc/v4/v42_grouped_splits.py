"""V4.2 分组 train/val 切分模块。

按 ``base_rainfall_fingerprint`` 分组，生成嵌套交叉验证的 fold 分配。
同一降雨事件的 DWF / no-DWF 变体、所有 Candidate 与 Reference 不得跨 fold。

使用 ``GroupKFold`` 以 ``rainfall_sha256`` 为分组键。
5 outer folds × 3 inner folds（匹配 V42CVPlan）。
所有 scaler / normalizer / feature-selection / sampling 仅 fit 在 train fold 上。
Reserved evaluation pool 不参与切分。

Output → audits/v42_final_pool/grouped_splits.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from sewerrtc._project_root import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_AUDIT_DIR = "audits/v42_final_pool"
_DATA_DIR = "data/v42_final_unified"
_DEFAULT_N_OUTER = 5
_DEFAULT_N_INNER = 3
_DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _load_lineage(project_root: Path) -> pd.DataFrame:
    """加载 sample_lineage.parquet。"""
    p = project_root / _DATA_DIR / "sample_lineage.parquet"
    if not p.exists():
        p = project_root / _AUDIT_DIR / "sample_lineage.parquet"
    if not p.exists():
        raise FileNotFoundError(f"sample_lineage.parquet not found")
    return pd.read_parquet(p)


def _load_classification(project_root: Path) -> pd.DataFrame:
    """加载 sample_classification.parquet。"""
    p = project_root / _AUDIT_DIR / "sample_classification.parquet"
    if p.exists():
        return pd.read_parquet(p)
    raise FileNotFoundError(f"sample_classification.parquet not found at {p}")


def _load_reserved(project_root: Path) -> pd.DataFrame:
    """加载 reserved_evaluation_manifest.csv（可能为空）。"""
    p = project_root / _DATA_DIR / "reserved_evaluation_manifest.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _freeze_seeds(base_seed: int, n: int) -> List[int]:
    """从 base_seed 生成 n 个确定性种子。"""
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(n)]


def _build_event_group_table(
    lineage: pd.DataFrame,
    classification: pd.DataFrame,
    reserved_ids: set,
) -> pd.DataFrame:
    """构建事件级分组表。

    返回 DataFrame，每行一个唯一 event_id，包含：
      - event_id
      - rainfall_sha256 (分组键)
      - n_samples: 该事件的样本数
      - is_reserved: 是否为 reserved evaluation 样本
    """
    # 合并 lineage 和 classification 获取 rainfall_fingerprint
    lin = lineage.copy()
    cls = classification.copy()

    # 从 lineage 获取 event_id → rainfall_fingerprint 映射
    if "rainfall_fingerprint" in lin.columns:
        event_rain = lin[["event_id", "rainfall_fingerprint"]].drop_duplicates(
            subset="event_id", keep="first"
        )
        event_rain = event_rain.rename(
            columns={"rainfall_fingerprint": "rainfall_sha256"}
        )
    else:
        raise ValueError("lineage missing 'rainfall_fingerprint' column")

    # 统计每个事件的样本数
    event_counts = cls.groupby("event_id").size().reset_index(name="n_samples")

    # 合并
    event_df = event_counts.merge(event_rain, on="event_id", how="left")

    # 标记 reserved
    if "event_id" in cls.columns and reserved_ids:
        reserved_events = set(
            cls.loc[cls["sample_id"].isin(reserved_ids), "event_id"]
            if "sample_id" in cls.columns
            else []
        )
        event_df["is_reserved"] = event_df["event_id"].isin(reserved_events)
    else:
        event_df["is_reserved"] = False

    return event_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 核心 API
# ---------------------------------------------------------------------------

def build_grouped_splits(
    project_root: Path,
    output_root: Path,
    dataset_path: Path | None = None,
    n_outer: int = _DEFAULT_N_OUTER,
    n_inner: int = _DEFAULT_N_INNER,
    frozen_seeds: list[int] | None = None,
) -> dict:
    """Build grouped train/val splits. Returns split summary.

    Parameters
    ----------
    project_root : Path
        项目根目录。
    output_root : Path
        输出根目录。
    dataset_path : Path | None
        可选；指定数据集路径以覆盖默认 lineage 位置。
    n_outer : int
        外层 fold 数（默认 5）。
    n_inner : int
        内层 fold 数（默认 3）。
    frozen_seeds : list[int] | None
        可选冻结种子列表。若 None 则使用默认种子生成。

    Returns
    -------
    dict
        包含 fold 分配、事件统计、种子信息的摘要字典。
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # ------------------------------------------------------------------
    # 1. 加载上游数据
    # ------------------------------------------------------------------
    lineage = _load_lineage(project_root)
    classification = _load_classification(project_root)
    reserved_df = _load_reserved(project_root)

    # Reserved evaluation 样本 ID 集合
    reserved_ids: set = set()
    if not reserved_df.empty and "sample_id" in reserved_df.columns:
        reserved_ids = set(reserved_df["sample_id"].tolist())
    elif not reserved_df.empty and "sample_idx" in reserved_df.columns:
        reserved_ids = set(reserved_df["sample_idx"].tolist())

    logger.info(
        "Loaded: lineage=%d, classification=%d, reserved=%d",
        len(lineage), len(classification), len(reserved_ids),
    )

    # ------------------------------------------------------------------
    # 2. 过滤 reserved evaluation 样本
    # ------------------------------------------------------------------
    cls_train = classification.copy()
    if reserved_ids:
        if "sample_id" in cls_train.columns:
            cls_train = cls_train[~cls_train["sample_id"].isin(reserved_ids)]
        elif "sample_idx" in cls_train.columns:
            cls_train = cls_train[~cls_train["sample_idx"].isin(reserved_ids)]
    logger.info("After excluding reserved: %d samples", len(cls_train))

    # ------------------------------------------------------------------
    # 3. 构建事件级分组表
    # ------------------------------------------------------------------
    event_df = _build_event_group_table(lineage, cls_train, reserved_ids)
    # 仅使用非 reserved 事件
    event_df = event_df[~event_df["is_reserved"]].reset_index(drop=True)

    unique_events = event_df["event_id"].values
    groups = event_df["rainfall_sha256"].values

    n_events = len(event_df)
    n_groups = event_df["rainfall_sha256"].nunique()
    logger.info(
        "Event table: %d events, %d unique rainfall groups",
        n_events, n_groups,
    )

    if n_groups < n_outer:
        raise ValueError(
            f"Cannot create {n_outer} folds with only {n_groups} rainfall groups"
        )

    # ------------------------------------------------------------------
    # 4. 外层 GroupKFold
    # ------------------------------------------------------------------
    outer_gkf = GroupKFold(n_splits=n_outer)
    pseudo_y = np.zeros(n_events)
    outer_assignment: Dict[str, int] = {}  # event_id → fold_idx

    for fold_idx, (_train_idx, test_idx) in enumerate(
        outer_gkf.split(np.arange(n_events), pseudo_y, groups)
    ):
        for idx in test_idx:
            eid = str(unique_events[idx])
            if eid in outer_assignment:
                raise RuntimeError(f"Event {eid} assigned to multiple outer folds")
            outer_assignment[eid] = int(fold_idx)

    # 验证所有事件均已分配
    unassigned = set(unique_events) - set(outer_assignment.keys())
    if unassigned:
        raise RuntimeError(f"{len(unassigned)} events not assigned to any fold")

    # ------------------------------------------------------------------
    # 5. 内层 GroupKFold（每个 outer train fold 内部）
    # ------------------------------------------------------------------
    inner_assignments: List[Dict[str, Any]] = []

    for outer_fold in range(n_outer):
        # 找出 outer train 部分的事件
        train_mask = np.array([
            outer_assignment.get(eid, -1) != outer_fold
            for eid in unique_events
        ])
        train_events = unique_events[train_mask]
        train_groups = groups[train_mask]
        n_train = len(train_events)

        if n_train < n_inner:
            logger.warning(
                "Outer fold %d: only %d train events, need >= %d for inner folds",
                outer_fold, n_train, n_inner,
            )
            continue

        inner_gkf = GroupKFold(n_splits=n_inner)
        inner_pseudo_y = np.zeros(n_train)
        inner_fold_map: Dict[str, int] = {}

        for inner_idx, (_tr, te) in enumerate(
            inner_gkf.split(np.arange(n_train), inner_pseudo_y, train_groups)
        ):
            for idx in te:
                eid = str(train_events[idx])
                inner_fold_map[eid] = int(inner_idx)

        for eid in train_events:
            inner_assignments.append({
                "event_id": eid,
                "outer_fold": int(outer_fold),
                "inner_fold": inner_fold_map.get(str(eid), -1),
            })

    # ------------------------------------------------------------------
    # 6. 种子
    # ------------------------------------------------------------------
    if frozen_seeds is None:
        frozen_seeds = _freeze_seeds(_DEFAULT_SEED, n_outer * n_inner)

    # ------------------------------------------------------------------
    # 7. 构建 fold 统计
    # ------------------------------------------------------------------
    outer_fold_stats: List[Dict[str, Any]] = []
    for fold_idx in range(n_outer):
        test_events = [
            eid for eid, f in outer_assignment.items() if f == fold_idx
        ]
        # 该 fold 的样本数
        n_samples = int(
            cls_train.loc[cls_train["event_id"].isin(test_events)].shape[0]
        ) if "event_id" in cls_train.columns else 0
        # 该 fold 的 rainfall 组数
        test_event_df = event_df[event_df["event_id"].isin(test_events)]
        n_rainfall_groups = int(test_event_df["rainfall_sha256"].nunique())

        outer_fold_stats.append({
            "fold": fold_idx,
            "n_test_events": len(test_events),
            "n_test_samples": n_samples,
            "n_rainfall_groups": n_rainfall_groups,
            "test_event_ids": sorted(test_events),
        })

    # ------------------------------------------------------------------
    # 8. 组装输出
    # ------------------------------------------------------------------
    result = {
        "algorithm": "GroupKFold",
        "group_column": "rainfall_sha256",
        "n_outer": n_outer,
        "n_inner": n_inner,
        "frozen_seeds": frozen_seeds,
        "n_events": n_events,
        "n_unique_rainfall_groups": int(n_groups),
        "n_total_train_samples": int(len(cls_train)),
        "n_reserved_excluded": int(len(reserved_ids)),
        "outer_fold_assignment": outer_assignment,
        "outer_fold_stats": outer_fold_stats,
        "inner_fold_assignment": inner_assignments,
    }

    # ------------------------------------------------------------------
    # 9. 写出 JSON
    # ------------------------------------------------------------------
    out_dir = output_root / _AUDIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grouped_splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote grouped_splits.json → %s", out_path)

    # 同时写出可读的 CSV 版本
    outer_csv_path = out_dir / "grouped_splits_outer.csv"
    outer_rows = [
        {"event_id": eid, "outer_fold": fid}
        for eid, fid in sorted(outer_assignment.items())
    ]
    pd.DataFrame(outer_rows).to_csv(outer_csv_path, index=False)

    inner_csv_path = out_dir / "grouped_splits_inner.csv"
    if inner_assignments:
        pd.DataFrame(inner_assignments).to_csv(inner_csv_path, index=False)
    else:
        pd.DataFrame(
            columns=["event_id", "outer_fold", "inner_fold"]
        ).to_csv(inner_csv_path, index=False)

    logger.info(
        "Grouped splits complete: %d outer × %d inner, %d events, %d groups",
        n_outer, n_inner, n_events, n_groups,
    )

    return result


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

    result = build_grouped_splits(project_root, project_root)
    print(json.dumps(
        {k: v for k, v in result.items()
         if k not in ("outer_fold_assignment", "inner_fold_assignment")},
        indent=2, ensure_ascii=False,
    ))
    print(f"\nOuter fold summary:")
    for fs in result.get("outer_fold_stats", []):
        print(
            f"  Fold {fs['fold']}: {fs['n_test_events']} events, "
            f"{fs['n_test_samples']} samples, "
            f"{fs['n_rainfall_groups']} rainfall groups"
        )


if __name__ == "__main__":
    main()
