"""Materialise explicit V4.2 Step2 hydraulic targets under a named contract.

The input is an already admitted Raw/GAT Step2 manifest. This script does not
invent or impute targets. It re-reads each authoritative branch detail at the 12
exact H120 timestamps and appends the targets required by the selected contract.

CONTROL_CORE
    node depth + node flooding (already present) + storage volume + managed
    facility flow. Explicit outfall flow is optional and reported when present.

FULL_HYDRAULIC
    CONTROL_CORE plus explicit outfall flow.

No-control is explicitly defined as all Engineering36 facilities fully open/on
(setting=1) throughout H120. Four-reference equality is audited. If two branches
have identical executed action schedules under the same admitted state/forcing,
their depth/flood trajectories must also be identical within numerical tolerance;
otherwise the row is rejected as a counterfactual alignment error.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_node_safety import load_node_physical_contract
from sewerrtc.v4.v42_reference_semantics import branch_equivalence, no_control_all_open
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology

BRANCHES = ("candidate", "no_control", "dynamic_internal", "hold_previous")
HORIZON_STEPS = 12
ATOL_MIN = 1.0e-6


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = (
        pd.read_parquet(path)
        if path.suffix.lower() == ".parquet"
        else pd.read_csv(path, low_memory=False)
    )
    if frame.empty:
        raise ValueError(f"input table is empty: {path}")
    return frame


def _json_array(value: Any) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float64)


def _exact_future(detail: pd.DataFrame, checkpoint: float) -> pd.DataFrame:
    if "elapsed_min" not in detail.columns:
        raise KeyError("detail missing elapsed_min")
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(elapsed).all():
        raise ValueError("elapsed_min contains NaN/Inf")
    rows: list[int] = []
    for k in range(1, HORIZON_STEPS + 1):
        target = float(checkpoint) + 10.0 * k
        found = np.flatnonzero(np.isclose(elapsed, target, atol=ATOL_MIN, rtol=0.0))
        if len(found) != 1:
            raise ValueError(
                f"expected exactly one row at elapsed_min={target}, got {len(found)}"
            )
        rows.append(int(found[0]))
    return detail.iloc[rows].copy()


def _columns(prefix: str, ids: list[str]) -> list[str]:
    return [f"{prefix}{item}" for item in ids]


def _extract(
    frame: pd.DataFrame, columns: list[str], *, required: bool
) -> tuple[np.ndarray | None, float]:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        if required:
            raise KeyError(f"missing required target columns: {missing[:10]}")
        return None, 0.0
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    fraction = float(finite.mean()) if finite.size else 0.0
    if required and fraction < 1.0 - 1.0e-12:
        raise ValueError(
            f"required target contains NaN/Inf, finite_fraction={fraction:.6f}"
        )
    if fraction < 1.0 - 1.0e-12:
        return None, fraction
    return values.astype(np.float32), fraction


def _cached_detail(
    cache: OrderedDict[str, pd.DataFrame],
    path: Path,
    max_items: int,
    usecols: list[str],
) -> pd.DataFrame:
    key = str(path.resolve())
    if key in cache:
        value = cache.pop(key)
        cache[key] = value
        return value
    value = pd.read_csv(path, usecols=usecols, low_memory=False)
    if value.empty:
        raise ValueError(f"empty detail: {path}")
    cache[key] = value
    while len(cache) > max_items:
        cache.popitem(last=False)
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument(
        "--target-contract",
        choices=("CONTROL_CORE", "FULL_HYDRAULIC"),
        default="CONTROL_CORE",
    )
    ap.add_argument("--min-rainfall-groups", type=int, default=65)
    ap.add_argument("--detail-cache-items", type=int, default=8)
    args = ap.parse_args()

    if args.detail_cache_items < 1:
        raise ValueError("detail-cache-items must be positive")
    frame = _read(args.input_manifest)
    required_manifest = {
        "split_group_key",
        "checkpoint_min",
        *[f"source_detail_path_{b}" for b in BRANCHES],
        *[f"action_{b}_readback" for b in BRANCHES],
        *[f"trajectory_depth_{b}" for b in BRANCHES],
        *[f"trajectory_flood_{b}" for b in BRANCHES],
    }
    missing = sorted(required_manifest - set(frame.columns))
    if missing:
        raise KeyError(f"Step2 manifest missing required columns: {missing[:20]}")

    graph = _load_graph_topology(args.project_root)
    physical = load_node_physical_contract(args.project_root)
    node_ids = list(map(str, graph["node_ids"]))
    if tuple(node_ids) != physical.node_ids:
        raise RuntimeError("raw INP node metadata is not aligned with graph node order")
    facility_ids = list(map(str, graph["facility_ids"]))
    storage_ids = [node_ids[i] for i in physical.storage_indices]
    outfall_ids = [node_ids[i] for i in physical.outfall_indices]
    storage_cols = _columns("storage_volume:", storage_ids)
    facility_cols = _columns("flow:", facility_ids)
    outfall_cols = _columns("outfall_flow:", outfall_ids)
    require_outfall = args.target_contract == "FULL_HYDRAULIC"
    detail_usecols = list(dict.fromkeys(["elapsed_min", *storage_cols, *facility_cols]))
    if require_outfall:
        detail_usecols.extend(outfall_cols)

    cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    equivalence_action_counts: dict[str, int] = {}
    equivalence_hydraulic_counts: dict[str, int] = {}
    action_equal_hydraulic_mismatch_count = 0
    outfall_rows_available = 0

    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        record = row.to_dict()
        try:
            checkpoint = float(row["checkpoint_min"])
            if not np.isfinite(checkpoint):
                raise ValueError("checkpoint_min is not finite")
            actions = {
                b: _json_array(row[f"action_{b}_readback"]) for b in BRANCHES
            }
            depths = {
                b: _json_array(row[f"trajectory_depth_{b}"]) for b in BRANCHES
            }
            floods = {
                b: _json_array(row[f"trajectory_flood_{b}"]) for b in BRANCHES
            }
            if not no_control_all_open(actions["no_control"]):
                raise RuntimeError(
                    "No-control action contract violated: Engineering36 must be all-open/on (=1)"
                )

            branch_outfall_available = True
            for branch in BRANCHES:
                path = Path(str(row[f"source_detail_path_{branch}"]))
                detail = _cached_detail(
                    cache, path, args.detail_cache_items, detail_usecols
                )
                future = _exact_future(detail, checkpoint)
                storage, storage_fraction = _extract(
                    future, storage_cols, required=True
                )
                facility, facility_fraction = _extract(
                    future, facility_cols, required=True
                )
                outfall, outfall_fraction = _extract(
                    future, outfall_cols, required=require_outfall
                )
                record[f"trajectory_storage_volume_{branch}"] = json.dumps(
                    storage.tolist(), allow_nan=False
                )
                record[f"trajectory_facility_flow_{branch}"] = json.dumps(
                    facility.tolist(), allow_nan=False
                )
                record[f"trajectory_storage_volume_{branch}_available"] = True
                record[f"trajectory_facility_flow_{branch}_available"] = True
                record[f"storage_finite_fraction_{branch}"] = storage_fraction
                record[f"facility_flow_finite_fraction_{branch}"] = facility_fraction
                if outfall is not None:
                    record[f"trajectory_outfall_flow_{branch}"] = json.dumps(
                        outfall.tolist(), allow_nan=False
                    )
                    record[f"trajectory_outfall_flow_{branch}_available"] = True
                else:
                    record[f"trajectory_outfall_flow_{branch}_available"] = False
                    branch_outfall_available = False
                record[f"outfall_flow_finite_fraction_{branch}"] = outfall_fraction

            equivalence = branch_equivalence(
                actions, depths=depths, floods=floods, atol=1.0e-6
            )
            action_pairs = set(equivalence["action_equivalent_pairs"])
            hydraulic_pairs = set(equivalence["hydraulic_equivalent_pairs"])
            inconsistent = sorted(action_pairs - hydraulic_pairs)
            if inconsistent:
                action_equal_hydraulic_mismatch_count += 1
                raise RuntimeError(
                    "same executed branch action but different depth/flood trajectory under same admitted state/forcing: "
                    f"{inconsistent}"
                )
            for pair in action_pairs:
                equivalence_action_counts[pair] = (
                    equivalence_action_counts.get(pair, 0) + 1
                )
            for pair in hydraulic_pairs:
                equivalence_hydraulic_counts[pair] = (
                    equivalence_hydraulic_counts.get(pair, 0) + 1
                )
            if branch_outfall_available:
                outfall_rows_available += 1

            record.update(
                {
                    "step2_target_contract": args.target_contract,
                    "control_core_target_coverage_complete": True,
                    "full_hydraulic_target_coverage_complete": bool(
                        branch_outfall_available
                    ),
                    "storage_supervised_available": True,
                    "facility_flow_supervised_available": True,
                    "outfall_supervised_available": bool(branch_outfall_available),
                    "no_control_action_contract": "all_engineering36_settings_equal_1.0",
                    "no_control_all_open_verified": True,
                    "reference_action_equivalent_pairs": json.dumps(
                        sorted(action_pairs)
                    ),
                    "reference_hydraulic_equivalent_pairs": json.dumps(
                        sorted(hydraulic_pairs)
                    ),
                    "reference_unique_action_branch_count": int(
                        equivalence["unique_action_branch_count"]
                    ),
                    "reference_unique_hydraulic_branch_count": int(
                        equivalence["unique_hydraulic_branch_count"] or 0
                    ),
                    "action_equal_implies_hydraulic_equal_verified": True,
                }
            )
            records.append(record)
        except Exception as exc:
            failures.append(
                {
                    "row_number": row_number,
                    "case_uid": str(row.get("case_uid", "")),
                    "rainfall_group": str(row.get("split_group_key", "")),
                    "state_key": str(row.get("state_key", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if row_number % 50 == 0:
            print(
                json.dumps(
                    {
                        "stage": "step2_target_contract_materialization",
                        "processed": row_number,
                        "total": len(frame),
                        "accepted": len(records),
                        "failed": len(failures),
                        "cache_items": len(cache),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

    out = pd.DataFrame(records)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not out.empty:
        out.to_parquet(args.output_manifest, index=False)
    groups = (
        int(out["split_group_key"].astype(str).nunique()) if not out.empty else 0
    )
    status = "pass" if groups >= args.min_rainfall_groups else "fail"
    audit = {
        "stage": "step2_target_contract_materialization",
        "status": status,
        "target_contract": args.target_contract,
        "input_rows": int(len(frame)),
        "accepted_rows": int(len(out)),
        "accepted_rainfall_groups": groups,
        "minimum_rainfall_groups": int(args.min_rainfall_groups),
        "failed_rows": int(len(failures)),
        "failure_examples": failures[:100],
        "storage_node_count": int(len(storage_ids)),
        "managed_facility_count": int(len(facility_ids)),
        "outfall_node_count": int(len(outfall_ids)),
        "physical_node_metadata_authority": "raw_frozen_INP_not_standardized_GNN_features",
        "storage_supervision_required": True,
        "facility_flow_supervision_required": True,
        "outfall_supervision_required": require_outfall,
        "outfall_rows_available": int(outfall_rows_available),
        "no_control_all_open_required": True,
        "reference_action_equivalence_counts": equivalence_action_counts,
        "reference_hydraulic_equivalence_counts": equivalence_hydraulic_counts,
        "action_equal_hydraulic_mismatch_count": int(
            action_equal_hydraulic_mismatch_count
        ),
        "equivalent_branches_are_diagnostic_not_automatically_invalid": True,
        "no_missing_target_imputation": True,
    }
    audit_path = args.output_manifest.with_name(
        args.output_manifest.stem + "_TARGET_AUDIT.json"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
