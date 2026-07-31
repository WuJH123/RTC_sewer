"""Materialize the formal V4.2 hydraulic-surrogate dataset from Phase-R0.

This is the canonical bridge from the project-wide evidence pool to Step 2.  It
intentionally does not reuse the historical Train1600-only resolver.

Critical online/offline contract:
* Step-2 history input is full-network **depth only** plus actual historical
  Engineering36 readback actions.  Flooding rate remains a prediction target;
  it is not leaked into the model input because Step 1 does not reconstruct a
  full-network flooding-rate history online.
* Candidate/NC/DI/Hold branches come from the strict R0 case manifest.
* every formal branch is finite, aligned, same-forcing and all-target complete;
* source physical IDs and raw detail paths are retained so the Independent
  Oracle can validate the exact population without rediscovering Train1600.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v42_priority_contract import get_pfv_core_node_indices
from .v42_trajectory_builder import (
    HISTORY_INTERVAL_MIN,
    HORIZON_INTERVAL_MIN,
    N_FACILITIES,
    N_HISTORY_FRAMES,
    N_HORIZON_STEPS,
    TIME_ATOL_MIN,
    _load_graph_topology,
    _parse_inp_topology,
)


FOUR_ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")
DT_SEC = float(HORIZON_INTERVAL_MIN * 60)


@dataclass(frozen=True)
class R0PaperDatasetResult:
    manifest_path: Path
    audit_path: Path
    accepted_count: int
    rejected_count: int
    lineage_sha256: str


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _write(frame: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, index=False)
    return p


def _json_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(x) for x in json.loads(str(value))]


def _exact_times(checkpoint: float) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(
        [
            checkpoint - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN
            for i in range(N_HISTORY_FRAMES)
        ],
        dtype=np.float64,
    )
    future = np.asarray(
        [checkpoint + (i + 1) * HORIZON_INTERVAL_MIN for i in range(N_HORIZON_STEPS)],
        dtype=np.float64,
    )
    return history, future


def _select(df: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    if "elapsed_min" not in df.columns:
        raise KeyError("detail missing elapsed_min")
    elapsed = pd.to_numeric(df["elapsed_min"], errors="coerce").to_numpy(float)
    if not np.isfinite(elapsed).all():
        raise ValueError("elapsed_min contains NaN/Inf")
    idx: list[int] = []
    for t in times:
        matches = np.flatnonzero(np.isclose(elapsed, t, atol=TIME_ATOL_MIN, rtol=0.0))
        if len(matches) != 1:
            raise ValueError(f"expected one row at {t} min, found {len(matches)}")
        idx.append(int(matches[0]))
    return df.iloc[idx].reset_index(drop=True)


def _cols(df: pd.DataFrame, prefix: str, ids: list[str]) -> np.ndarray:
    lookup = {
        str(c)[len(prefix):].casefold(): str(c)
        for c in df.columns
        if str(c).startswith(prefix)
    }
    missing = [x for x in ids if x.casefold() not in lookup]
    if missing:
        raise KeyError(f"missing {prefix} columns: {missing[:10]}")
    cols = [lookup[x.casefold()] for x in ids]
    values = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {prefix} values")
    return values


def _rain(df: pd.DataFrame) -> np.ndarray:
    if "rainfall_mm_h" not in df.columns:
        raise KeyError("detail missing rainfall_mm_h")
    values = pd.to_numeric(df["rainfall_mm_h"], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("rainfall contains NaN/Inf")
    return values


def _storage_outfall_ids(project_root: Path, node_ids: list[str]) -> tuple[list[str], list[str]]:
    nodes, _ = _parse_inp_topology(project_root / "data" / "wuhan_v8_storage_retrofit.inp")
    meta = {str(r.node_id).casefold(): str(r.node_type) for r in nodes.itertuples(index=False)}
    storage = [x for x in node_ids if meta.get(x.casefold()) == "storage"]
    outfall = [x for x in node_ids if meta.get(x.casefold()) == "outfall"]
    return storage, outfall


def _role_rows(case, physical_by_id: dict[str, Any]) -> dict[str, Any]:
    candidates: dict[str, list[Any]] = {r: [] for r in FOUR_ROLES}
    for pid in _json_ids(getattr(case, "branch_physical_ids", "[]")):
        row = physical_by_id.get(pid)
        if row is None:
            continue
        role = str(getattr(row, "branch_role", ""))
        if role in candidates:
            candidates[role].append(row)
    result: dict[str, Any] = {}
    for role in FOUR_ROLES:
        usable = [
            r
            for r in candidates[role]
            if bool(getattr(r, "mask_finite", False))
            and bool(getattr(r, "formal_complete_branch", False))
        ]
        if not usable:
            raise ValueError(f"no finite formal physical branch for role={role}")
        result[role] = sorted(
            usable,
            key=lambda r: (str(getattr(r, "physical_identity_sha256", "")), str(getattr(r, "detail_path", ""))),
        )[0]
    return result


def _branch_future(
    detail: pd.DataFrame,
    *,
    future_times: np.ndarray,
    node_ids: list[str],
    storage_ids: list[str],
    facility_ids: list[str],
    outfall_ids: list[str],
) -> dict[str, np.ndarray]:
    f = _select(detail, future_times)
    return {
        "depth": _cols(f, "h:", node_ids),
        "flood": _cols(f, "flood:", node_ids),
        "storage_volume": _cols(f, "storage_volume:", storage_ids),
        "facility_flow": _cols(f, "flow:", facility_ids),
        "outfall_flow": _cols(f, "outfall_flow:", outfall_ids),
        "action_readback": _cols(f, "setting:", facility_ids),
        "rainfall": _rain(f),
    }


def _kpis(branches: dict[str, dict[str, np.ndarray]], priority_idx: list[int]) -> tuple[float, float, float]:
    cand = branches["candidate"]["flood"].astype(np.float64)
    nc = branches["no_control"]["flood"].astype(np.float64)
    di = branches["dynamic_internal"]["flood"].astype(np.float64)
    pfv = float((cand[:, priority_idx] - nc[:, priority_idx]).sum() * DT_SEC)
    tfv = float((cand.sum(axis=1) - di.sum(axis=1)).sum() * DT_SEC)
    peak = float(cand.sum(axis=1).max() - di.sum(axis=1).max())
    return pfv, tfv, peak


def _lineage_sha(records: list[dict[str, Any]]) -> str:
    payload = [
        {
            "case_uid": str(r["case_uid"]),
            "candidate": str(r["source_physical_id_candidate"]),
            "no_control": str(r["source_physical_id_no_control"]),
            "dynamic_internal": str(r["source_physical_id_dynamic_internal"]),
            "hold_previous": str(r["source_physical_id_hold_previous"]),
        }
        for r in records
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_r0_paper_dataset(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    case_manifest: str | Path,
    split_manifest: str | Path,
    output_manifest: str | Path,
    audit_output: str | Path,
) -> R0PaperDatasetResult:
    project_root = Path(project_root)
    physical = _read(physical_manifest)
    cases = _read(case_manifest)
    split = _read(split_manifest)
    if physical.empty or cases.empty or split.empty:
        raise ValueError("R0 manifests cannot be empty")
    if "eligible_formal_all_target" not in cases.columns:
        raise KeyError("strict R0 case manifest missing eligible_formal_all_target")

    graph = _load_graph_topology(project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    facility_ids = [str(x) for x in graph["facility_ids"]]
    if len(facility_ids) != N_FACILITIES:
        raise ValueError("formal V4.2 requires Engineering36")
    storage_ids, outfall_ids = _storage_outfall_ids(project_root, node_ids)
    priority_idx = get_pfv_core_node_indices(node_ids)

    physical_by_id = {
        str(r.physical_identity_sha256): r for r in physical.itertuples(index=False)
    }
    split_by_id = {
        str(r.physical_identity_sha256): str(r.split_group_key)
        for r in split.itertuples(index=False)
    }

    admitted = cases[
        cases["eligible_formal_all_target"].fillna(False).astype(bool)
    ].copy()
    if "source_role" in admitted.columns:
        admitted = admitted[admitted["source_role"].astype(str) != "reserved_evaluation"]
    admitted = admitted.sort_values(["case_uid"], kind="mergesort")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    detail_cache: dict[str, pd.DataFrame] = {}
    for case in admitted.itertuples(index=False):
        case_uid = str(getattr(case, "case_uid"))
        try:
            checkpoint = float(getattr(case, "checkpoint_min"))
            history_times, future_times = _exact_times(checkpoint)
            roles = _role_rows(case, physical_by_id)
            details: dict[str, pd.DataFrame] = {}
            for role, row in roles.items():
                path = str(getattr(row, "detail_path"))
                if path not in detail_cache:
                    detail_cache[path] = pd.read_csv(path)
                details[role] = detail_cache[path]

            history = _select(details["candidate"], history_times)
            history_depth = _cols(history, "h:", node_ids)
            history_actions = _cols(history, "setting:", facility_ids)
            branches = {
                role: _branch_future(
                    detail,
                    future_times=future_times,
                    node_ids=node_ids,
                    storage_ids=storage_ids,
                    facility_ids=facility_ids,
                    outfall_ids=outfall_ids,
                )
                for role, detail in details.items()
            }
            rainfall = branches["candidate"]["rainfall"]
            for role in FOUR_ROLES[1:]:
                if not np.allclose(rainfall, branches[role]["rainfall"], atol=1e-7, rtol=0.0):
                    raise ValueError(f"future rainfall differs for {role}")

            group_keys = {
                split_by_id.get(str(getattr(row, "physical_identity_sha256")), "")
                for row in roles.values()
            }
            group_keys.discard("")
            if len(group_keys) != 1:
                raise ValueError("four branches do not resolve to one rainfall split group")
            split_group_key = next(iter(group_keys))
            pfv, tfv, peak = _kpis(branches, priority_idx)

            rec: dict[str, Any] = {
                "case_uid": case_uid,
                "case_id": str(getattr(case, "case_id", "")),
                "event_id": str(getattr(case, "event_id", "")),
                "checkpoint_min": checkpoint,
                "rainfall_sha256": str(getattr(case, "rainfall_sha256", "")),
                "split_group_key": split_group_key,
                "history_input_contract": "gat_compatible_causal_state",
                "state_feature_names": json.dumps(["depth_m"]),
                "history_depth": json.dumps(history_depth.tolist(), allow_nan=False),
                "history_actions_readback": json.dumps(history_actions.tolist(), allow_nan=False),
                "history_elapsed_min": json.dumps(history_times.tolist(), allow_nan=False),
                "future_elapsed_min": json.dumps(future_times.tolist(), allow_nan=False),
                "rainfall_forecast": json.dumps(rainfall.tolist(), allow_nan=False),
                "action_authority": "actual_readback_setting",
                "kpi_authority": "derived_from_raw_flooding_rate_trajectory",
                "pfv_delta": pfv,
                "tfv_delta": tfv,
                "peak_delta": peak,
            }
            for role, arrays in branches.items():
                rec[f"action_{role}_readback"] = json.dumps(
                    arrays["action_readback"].tolist(), allow_nan=False
                )
                for quantity in ("depth", "flood", "storage_volume", "facility_flow", "outfall_flow"):
                    rec[f"trajectory_{quantity}_{role}"] = json.dumps(
                        arrays[quantity].tolist(), allow_nan=False
                    )
                source = roles[role]
                rec[f"source_physical_id_{role}"] = str(
                    getattr(source, "physical_identity_sha256")
                )
                rec[f"source_detail_path_{role}"] = str(getattr(source, "detail_path"))
            records.append(rec)
        except Exception as exc:
            failures.append(
                {
                    "case_uid": case_uid,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result = pd.DataFrame(records)
    lineage = _lineage_sha(records) if records else ""
    if not result.empty:
        result["sample_lineage_sha256"] = lineage
    _write(result, output_manifest)
    audit = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "dataset_role": "r0_to_trajectory_first_hydraulic_surrogate",
        "source_physical_manifest": str(physical_manifest),
        "source_case_manifest": str(case_manifest),
        "source_split_manifest": str(split_manifest),
        "candidate_case_count": int(len(admitted)),
        "accepted_count": int(len(records)),
        "rejected_count": int(len(failures)),
        "formal_complete": bool(records) and not failures and len(records) == len(admitted),
        "history_input_contract": "gat_compatible_causal_state",
        "history_features": ["depth_m"],
        "flooding_history_used_as_input": False,
        "action_authority": "actual_readback_setting",
        "four_reference_required": True,
        "rainfall_group_isolation_key_preserved": True,
        "sample_lineage_sha256": lineage,
        "failures": failures[:100],
    }
    audit_path = Path(audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return R0PaperDatasetResult(
        manifest_path=Path(output_manifest),
        audit_path=audit_path,
        accepted_count=len(records),
        rejected_count=len(failures),
        lineage_sha256=lineage,
    )
