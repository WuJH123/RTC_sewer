"""Numeric same-state and forcing audit for reusable four-reference cases.

Hash equality is useful lineage evidence but is not sufficient for admission.
This module reopens the real detail files and compares Candidate/NC/DI/Hold
history numerically at the frozen 13 x 5-minute prefix and compares future
rainfall forcing at the H120 timestamps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_trajectory_builder import (
    HISTORY_INTERVAL_MIN,
    HORIZON_INTERVAL_MIN,
    N_HISTORY_FRAMES,
    N_HORIZON_STEPS,
    _load_engineering36_ids,
    _load_graph_topology,
    _parse_inp_topology,
)


FOUR_ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")
TIME_ATOL_MIN = 1.0e-6


@dataclass(frozen=True)
class CaseAlignmentResult:
    case_uid: str
    branch_count: int
    checkpoint_min: float | None
    same_state_numeric_pass: bool
    same_forcing_pass: bool
    max_depth_prefix_diff_m: float | None
    max_storage_prefix_diff_m3: float | None
    max_facility_flow_prefix_diff_m3s: float | None
    max_setting_prefix_diff: float | None
    max_future_rainfall_diff_mm_h: float | None
    error: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _read_json_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parsed = json.loads(str(value))
    return [str(x) for x in parsed]


def _select_times(df: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    if "elapsed_min" not in df.columns:
        raise KeyError("detail missing elapsed_min")
    elapsed = pd.to_numeric(df["elapsed_min"], errors="coerce").to_numpy(float)
    if not np.isfinite(elapsed).all():
        raise ValueError("elapsed_min contains non-finite values")
    rows: list[int] = []
    for target in targets:
        idx = np.flatnonzero(np.isclose(elapsed, target, atol=TIME_ATOL_MIN, rtol=0.0))
        if len(idx) != 1:
            raise ValueError(f"expected one row at {target} min, found {len(idx)}")
        rows.append(int(idx[0]))
    return df.iloc[rows].reset_index(drop=True)


def _columns_by_ids(df: pd.DataFrame, prefix: str, ids: list[str]) -> np.ndarray:
    lookup = {
        str(c)[len(prefix):].casefold(): str(c)
        for c in df.columns
        if str(c).startswith(prefix)
    }
    missing = [x for x in ids if x.casefold() not in lookup]
    if missing:
        raise KeyError(f"missing {prefix} columns: {missing[:10]}")
    cols = [lookup[x.casefold()] for x in ids]
    values = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {prefix} prefix values")
    return values


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def audit_case_alignment(
    *,
    project_root: str | Path,
    physical_inventory: str | Path,
    case_inventory: str | Path,
    output_path: str | Path,
    depth_atol_m: float = 1.0e-6,
    storage_atol_m3: float = 1.0e-6,
    flow_atol_m3s: float = 1.0e-6,
    setting_atol: float = 1.0e-7,
    rainfall_atol_mm_h: float = 1.0e-7,
) -> pd.DataFrame:
    project_root = Path(project_root)
    physical_path = Path(physical_inventory)
    case_path = Path(case_inventory)
    physical = pd.read_parquet(physical_path) if physical_path.suffix.lower() == ".parquet" else pd.read_csv(physical_path)
    cases = pd.read_parquet(case_path) if case_path.suffix.lower() == ".parquet" else pd.read_csv(case_path)
    graph = _load_graph_topology(project_root)
    node_ids = list(graph["node_ids"])
    facility_ids = _load_engineering36_ids(project_root)
    nodes, _ = _parse_inp_topology(project_root / "data" / "wuhan_v8_storage_retrofit.inp")
    storage_ids = [str(x) for x in nodes.loc[nodes["node_type"] == "storage", "node_id"].tolist()]
    by_physical_id = {
        str(row.physical_identity_sha256): row
        for row in physical.itertuples(index=False)
    }

    results: list[CaseAlignmentResult] = []
    for case in cases.itertuples(index=False):
        case_uid = str(getattr(case, "case_uid"))
        checkpoint_raw = getattr(case, "checkpoint_min", None)
        checkpoint = None if checkpoint_raw is None or pd.isna(checkpoint_raw) else float(checkpoint_raw)
        ids = _read_json_list(getattr(case, "branch_physical_ids", "[]"))
        branch_rows = [by_physical_id[x] for x in ids if x in by_physical_id]
        role_map = {str(row.branch_role): row for row in branch_rows}
        try:
            if checkpoint is None:
                raise ValueError("checkpoint is unknown")
            missing_roles = [role for role in FOUR_ROLES if role not in role_map]
            if missing_roles:
                raise ValueError(f"missing four-reference roles: {missing_roles}")
            history_times = [
                checkpoint - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN
                for i in range(N_HISTORY_FRAMES)
            ]
            future_times = [
                checkpoint + (i + 1) * HORIZON_INTERVAL_MIN
                for i in range(N_HORIZON_STEPS)
            ]
            arrays: dict[str, dict[str, np.ndarray]] = {}
            for role in FOUR_ROLES:
                detail = pd.read_csv(Path(role_map[role].detail_path))
                history = _select_times(detail, history_times)
                future = _select_times(detail, future_times)
                if "rainfall_mm_h" not in future.columns:
                    raise KeyError("detail missing rainfall_mm_h")
                rainfall = pd.to_numeric(future["rainfall_mm_h"], errors="coerce").to_numpy(float)
                if not np.isfinite(rainfall).all():
                    raise ValueError("future rainfall contains non-finite values")
                arrays[role] = {
                    "depth": _columns_by_ids(history, "h:", node_ids),
                    "storage": _columns_by_ids(history, "storage_volume:", storage_ids),
                    "flow": _columns_by_ids(history, "flow:", facility_ids),
                    "setting": _columns_by_ids(history, "setting:", facility_ids),
                    "rainfall": rainfall,
                }
            cand = arrays["candidate"]
            depth_diff = max(_max_abs(cand["depth"], arrays[role]["depth"]) for role in FOUR_ROLES[1:])
            storage_diff = max(_max_abs(cand["storage"], arrays[role]["storage"]) for role in FOUR_ROLES[1:])
            flow_diff = max(_max_abs(cand["flow"], arrays[role]["flow"]) for role in FOUR_ROLES[1:])
            setting_diff = max(_max_abs(cand["setting"], arrays[role]["setting"]) for role in FOUR_ROLES[1:])
            rain_diff = max(_max_abs(cand["rainfall"], arrays[role]["rainfall"]) for role in FOUR_ROLES[1:])
            same_state = bool(
                depth_diff <= depth_atol_m
                and storage_diff <= storage_atol_m3
                and flow_diff <= flow_atol_m3s
                and setting_diff <= setting_atol
            )
            same_forcing = bool(rain_diff <= rainfall_atol_mm_h)
            result = CaseAlignmentResult(
                case_uid=case_uid,
                branch_count=len(role_map),
                checkpoint_min=checkpoint,
                same_state_numeric_pass=same_state,
                same_forcing_pass=same_forcing,
                max_depth_prefix_diff_m=depth_diff,
                max_storage_prefix_diff_m3=storage_diff,
                max_facility_flow_prefix_diff_m3s=flow_diff,
                max_setting_prefix_diff=setting_diff,
                max_future_rainfall_diff_mm_h=rain_diff,
                error="",
            )
        except Exception as exc:
            result = CaseAlignmentResult(
                case_uid=case_uid,
                branch_count=len(role_map),
                checkpoint_min=checkpoint,
                same_state_numeric_pass=False,
                same_forcing_pass=False,
                max_depth_prefix_diff_m=None,
                max_storage_prefix_diff_m3=None,
                max_facility_flow_prefix_diff_m3s=None,
                max_setting_prefix_diff=None,
                max_future_rainfall_diff_mm_h=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)

    frame = pd.DataFrame([result.as_dict() for result in results])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        frame.to_csv(target, index=False)
    return frame
