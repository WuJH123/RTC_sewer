"""Build the formal trajectory-first V4.2 paper dataset from raw SWMM detail.

This builder is separate from the historical V4.2 KPI-head dataset.  It enforces
three paper-critical contracts:

* 13 causal state-history frames at 5 min and 12 future frames at 10 min;
* all four Candidate/NC/DI/Hold branches are present and hydraulically labelled;
* branch action inputs use **actual/readback** ``setting:<facility>`` columns,
  never requested/target ``a:<facility>`` columns.

Every required hydraulic target (including explicit outfall flow) must be
present.  Missing targets make the row inadmissible; they are not imputed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import (
    N_FACILITIES,
    N_HISTORY_FRAMES,
    N_HORIZON_STEPS,
    _load_graph_topology,
    _parse_inp_topology,
    _scan_all_run_dirs,
)


BRANCH_FILE_ROLES = {
    "candidate": "candidate",
    "no_control": "no_control",
    "dynamic_internal": "dynamic_internal_rules",
    "hold_previous": "hold_previous",
}
STATE_FEATURE_NAMES = ["depth_m", "head_m", "filling_degree", "flood_m3s"]
DT_SEC = 600.0
TIME_ATOL_MIN = 1.0e-6


@dataclass(frozen=True)
class PaperDatasetResult:
    manifest_path: Path
    audit_path: Path
    accepted_count: int
    rejected_count: int
    formal_complete: bool


def _json_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _detail_path(run_dir: Path, role: str) -> Path:
    completion_path = run_dir / "completion.json"
    if not completion_path.exists():
        raise FileNotFoundError(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    text = str(completion.get("branches", {}).get(role, {}).get("detail_path", ""))
    if not text:
        raise KeyError(f"{run_dir}: branch {role} missing detail_path")
    path = Path(text)
    if path.exists():
        return path
    local = run_dir / path.name
    if local.exists():
        return local
    raise FileNotFoundError(text)


def _select_exact_times(df: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    if "elapsed_min" not in df.columns:
        raise KeyError("detail.csv missing elapsed_min")
    elapsed = pd.to_numeric(df["elapsed_min"], errors="coerce").to_numpy(float)
    if not np.isfinite(elapsed).all():
        raise ValueError("elapsed_min contains NaN/Inf")
    indices: list[int] = []
    for target in times:
        matches = np.flatnonzero(np.abs(elapsed - float(target)) <= TIME_ATOL_MIN)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one raw detail row at {target} min, found {len(matches)}"
            )
        indices.append(int(matches[0]))
    return df.iloc[indices].reset_index(drop=True)


def _columns_by_ids(
    df: pd.DataFrame,
    prefix: str,
    ids: list[str],
) -> np.ndarray:
    lookup = {
        str(c)[len(prefix):].casefold(): str(c)
        for c in df.columns
        if str(c).startswith(prefix)
    }
    missing = [item for item in ids if item.casefold() not in lookup]
    if missing:
        raise KeyError(f"missing {prefix} columns: {missing[:10]}")
    cols = [lookup[item.casefold()] for item in ids]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"{prefix} target contains NaN/non-numeric values")
    return numeric.to_numpy(dtype=np.float32)


def _rainfall(df: pd.DataFrame) -> np.ndarray:
    if "rainfall_mm_h" not in df.columns:
        raise KeyError("detail.csv missing rainfall_mm_h")
    values = pd.to_numeric(df["rainfall_mm_h"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("rainfall_mm_h contains NaN/Inf")
    return values.astype(np.float32)


def _node_physical_metadata(project_root: Path, graph_node_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    nodes, _ = _parse_inp_topology(project_root / "data" / "wuhan_v8_storage_retrofit.inp")
    by_id = {str(row.node_id).casefold(): row for row in nodes.itertuples(index=False)}
    invert: list[float] = []
    max_depth: list[float] = []
    storage: list[str] = []
    outfalls: list[str] = []
    for nid in graph_node_ids:
        key = nid.casefold()
        if key not in by_id:
            raise KeyError(f"graph node {nid} missing from INP metadata")
        row = by_id[key]
        invert.append(float(row.invert))
        max_depth.append(float(row.max_depth))
        ntype = str(row.node_type)
        if ntype == "storage":
            storage.append(nid)
        if ntype == "outfall":
            outfalls.append(nid)
    return (
        np.asarray(invert, dtype=np.float32),
        np.asarray(max_depth, dtype=np.float32),
        storage,
        outfalls,
    )


def _history_state(
    history: pd.DataFrame,
    *,
    node_ids: list[str],
    invert_m: np.ndarray,
    max_depth_m: np.ndarray,
) -> np.ndarray:
    depth = _columns_by_ids(history, "h:", node_ids)
    # Prefer authoritative head when present; otherwise physical invert+depth.
    try:
        head = _columns_by_ids(history, "head:", node_ids)
    except KeyError:
        head = depth + invert_m[None, :]
    flood = _columns_by_ids(history, "flood:", node_ids)
    filling = np.full_like(depth, np.nan, dtype=np.float32)
    valid = max_depth_m > 1.0e-8
    filling[:, valid] = depth[:, valid] / max_depth_m[None, valid]
    if not np.isfinite(filling[:, valid]).all():
        raise ValueError("filling-degree history contains NaN/Inf")
    # Outfalls/zero-full-depth nodes have no meaningful filling degree.  Use a
    # separate neutral numeric value only for the model tensor; static is_outfall
    # remains available and no physical KPI is derived from this feature.
    filling[:, ~valid] = 0.0
    return np.stack([depth, head, filling, flood], axis=-1).astype(np.float32)


def _branch_arrays(
    detail: pd.DataFrame,
    *,
    future_times: np.ndarray,
    node_ids: list[str],
    storage_ids: list[str],
    facility_ids: list[str],
    outfall_ids: list[str],
) -> dict[str, np.ndarray]:
    future = _select_exact_times(detail, future_times)
    return {
        "depth": _columns_by_ids(future, "h:", node_ids),
        "flood": _columns_by_ids(future, "flood:", node_ids),
        "storage_volume": _columns_by_ids(future, "storage_volume:", storage_ids),
        # Formal input authority: actual/readback setting, not requested a:.
        "action_readback": _columns_by_ids(future, "setting:", facility_ids),
        "facility_flow": _columns_by_ids(future, "flow:", facility_ids),
        "outfall_flow": _columns_by_ids(future, "outfall_flow:", outfall_ids),
        "rainfall": _rainfall(future),
    }


def _stable_kpi_deltas(
    branches: dict[str, dict[str, np.ndarray]],
    *,
    priority_indices: list[int],
) -> tuple[float, float, float]:
    cand = branches["candidate"]["flood"].astype(np.float64)
    nc = branches["no_control"]["flood"].astype(np.float64)
    di = branches["dynamic_internal"]["flood"].astype(np.float64)
    pfv_delta = float((cand[:, priority_indices] - nc[:, priority_indices]).sum() * DT_SEC)
    tfv_delta = float((cand.sum(axis=1) - di.sum(axis=1)).sum() * DT_SEC)
    peak_delta = float(cand.sum(axis=1).max() - di.sum(axis=1).max())
    return pfv_delta, tfv_delta, peak_delta


def _sha_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_paper_dataset(
    *,
    project_root: str | Path,
    output_root: str | Path,
    source_manifest: str | Path,
    output_manifest: str | Path,
    audit_output: str | Path,
    require_all_rows: bool = True,
) -> PaperDatasetResult:
    """Build the formal raw-physics dataset or fail closed with row diagnostics."""
    project_root = Path(project_root)
    output_root = Path(output_root)
    source_manifest = Path(source_manifest)
    output_manifest = Path(output_manifest)
    audit_output = Path(audit_output)
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    source = (
        pd.read_parquet(source_manifest)
        if source_manifest.suffix.lower() == ".parquet"
        else pd.read_csv(source_manifest)
    )
    if source.empty:
        raise ValueError("source trajectory manifest is empty")

    graph = _load_graph_topology(project_root)
    node_ids = list(graph["node_ids"])
    facility_ids = list(graph["facility_ids"])
    if len(facility_ids) != N_FACILITIES:
        raise ValueError("formal paper dataset requires Engineering36")
    invert_m, max_depth_m, storage_ids, outfall_ids = _node_physical_metadata(
        project_root, node_ids
    )
    priority_indices = get_pfv_core_node_indices(node_ids)
    case_map = _scan_all_run_dirs(output_root)
    if not case_map:
        raise FileNotFoundError("no Train1600 run directories found")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row_idx, row in source.iterrows():
        case_id = str(row.get("case_id", ""))
        try:
            if not case_id or case_id not in case_map:
                raise KeyError(f"run directory not found for {case_id!r}")
            history_times = _json_array(row["history_elapsed_min"], name="history_elapsed_min")
            future_times = _json_array(row["future_elapsed_min"], name="future_elapsed_min")
            if history_times.shape != (N_HISTORY_FRAMES,):
                raise ValueError(f"history must contain {N_HISTORY_FRAMES} frames")
            if future_times.shape != (N_HORIZON_STEPS,):
                raise ValueError(f"future must contain {N_HORIZON_STEPS} steps")

            run_dir = case_map[case_id]
            details: dict[str, pd.DataFrame] = {}
            for public_role, file_role in BRANCH_FILE_ROLES.items():
                detail_path = _detail_path(run_dir, file_role)
                details[public_role] = pd.read_csv(detail_path)
                if details[public_role].empty:
                    raise ValueError(f"empty detail for {file_role}")

            candidate_history = _select_exact_times(
                details["candidate"], history_times
            )
            state_history = _history_state(
                candidate_history,
                node_ids=node_ids,
                invert_m=invert_m,
                max_depth_m=max_depth_m,
            )
            history_actions = _columns_by_ids(
                candidate_history, "setting:", facility_ids
            )
            if history_actions.shape != (N_HISTORY_FRAMES, N_FACILITIES):
                raise ValueError("history readback action shape mismatch")

            branches = {
                role: _branch_arrays(
                    detail,
                    future_times=future_times,
                    node_ids=node_ids,
                    storage_ids=storage_ids,
                    facility_ids=facility_ids,
                    outfall_ids=outfall_ids,
                )
                for role, detail in details.items()
            }
            # Rainfall forcing is shared across branches.  Mismatch implies a
            # broken same-event counterfactual and is not admissible.
            rainfall = branches["candidate"]["rainfall"]
            for role in ("no_control", "dynamic_internal", "hold_previous"):
                if not np.allclose(rainfall, branches[role]["rainfall"], atol=1e-7, rtol=0.0):
                    raise ValueError(f"rainfall forcing differs in {role} branch")

            pfv_delta, tfv_delta, peak_delta = _stable_kpi_deltas(
                branches, priority_indices=priority_indices
            )
            record: dict[str, Any] = {
                "event_id": str(row.get("event_id", "")),
                "checkpoint_id": str(row.get("checkpoint_id", "")),
                "state_key": str(row.get("state_key", "")),
                "case_id": case_id,
                "split": str(row.get("split", "train")),
                "checkpoint_min": float(row["checkpoint_min"]),
                "node_order_sha256": _sha_ids(node_ids),
                "facility_order_sha256": _sha_ids(facility_ids),
                "state_feature_names": json.dumps(STATE_FEATURE_NAMES),
                "history_state": json.dumps(state_history.tolist(), allow_nan=False),
                "history_actions_readback": json.dumps(history_actions.tolist(), allow_nan=False),
                "history_elapsed_min": json.dumps(history_times.tolist(), allow_nan=False),
                "future_elapsed_min": json.dumps(future_times.tolist(), allow_nan=False),
                "rainfall_forecast": json.dumps(rainfall.tolist(), allow_nan=False),
                "pfv_delta": pfv_delta,
                "tfv_delta": tfv_delta,
                "peak_delta": peak_delta,
                "action_authority": "actual_readback_setting",
                "kpi_authority": "derived_from_raw_flooding_rate_trajectory",
            }
            for role, arrays in branches.items():
                record[f"action_{role}_readback"] = json.dumps(
                    arrays["action_readback"].tolist(), allow_nan=False
                )
                for quantity in (
                    "depth",
                    "flood",
                    "storage_volume",
                    "facility_flow",
                    "outfall_flow",
                ):
                    record[f"trajectory_{quantity}_{role}"] = json.dumps(
                        arrays[quantity].tolist(), allow_nan=False
                    )
            records.append(record)
        except Exception as exc:
            failures.append(
                {
                    "row_index": int(row_idx),
                    "case_id": case_id,
                    "event_id": str(row.get("event_id", "")),
                    "checkpoint_id": str(row.get("checkpoint_id", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(records)
    if output_manifest.suffix.lower() == ".parquet":
        result_df.to_parquet(output_manifest, index=False)
    else:
        result_df.to_csv(output_manifest, index=False)
    formal_complete = len(records) == len(source) and not failures
    audit = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "dataset_role": "trajectory_first_hydraulic_surrogate",
        "source_manifest": str(source_manifest),
        "source_row_count": int(len(source)),
        "accepted_count": int(len(records)),
        "rejected_count": int(len(failures)),
        "formal_complete": bool(formal_complete),
        "required_branches": list(BRANCH_FILE_ROLES),
        "action_authority": "actual_readback_setting",
        "required_hydraulic_targets": [
            "node_depth",
            "node_flooding_rate",
            "storage_volume",
            "managed_facility_flow",
            "outfall_flow",
        ],
        "failures": failures,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    if require_all_rows and not formal_complete:
        raise RuntimeError(
            f"formal paper dataset incomplete: {len(records)}/{len(source)} rows accepted; "
            f"see {audit_output}"
        )
    return PaperDatasetResult(
        manifest_path=output_manifest,
        audit_path=audit_output,
        accepted_count=len(records),
        rejected_count=len(failures),
        formal_complete=formal_complete,
    )
