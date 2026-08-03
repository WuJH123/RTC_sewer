"""Audit raw SWMM detail files for V4.2 Step2 target contracts.

Two explicit contracts are supported:

CONTROL_CORE
    node depth, node flooding rate, storage volume, managed-facility flow.

FULL_HYDRAULIC
    CONTROL_CORE plus explicit outfall flow.

Missing targets are reported as missing and are never reconstructed or zero
filled from another variable. Backward-compatible aliases ``core_trajectory``
and ``formal_complete`` are retained for existing audit readers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetCoverage:
    detail_path: str
    node_depth: bool
    node_flooding_rate: bool
    storage_volume: bool
    managed_facility_flow: bool
    outfall_flow: bool
    finite_fraction: dict[str, float]
    missing_columns: dict[str, list[str]]

    @property
    def control_core_complete(self) -> bool:
        return bool(
            self.node_depth
            and self.node_flooding_rate
            and self.storage_volume
            and self.managed_facility_flow
        )

    @property
    def full_hydraulic_complete(self) -> bool:
        return bool(self.control_core_complete and self.outfall_flow)

    @property
    def core_trajectory_complete(self) -> bool:
        return self.control_core_complete

    @property
    def formal_complete(self) -> bool:
        """Legacy alias for the old extended FULL_HYDRAULIC contract."""
        return self.full_hydraulic_complete

    def as_dict(self) -> dict:
        return {
            "detail_path": self.detail_path,
            "node_depth": self.node_depth,
            "node_flooding_rate": self.node_flooding_rate,
            "storage_volume": self.storage_volume,
            "managed_facility_flow": self.managed_facility_flow,
            "outfall_flow": self.outfall_flow,
            "control_core_complete": self.control_core_complete,
            "full_hydraulic_complete": self.full_hydraulic_complete,
            "core_trajectory_complete": self.core_trajectory_complete,
            "formal_complete": self.formal_complete,
            "finite_fraction": self.finite_fraction,
            "missing_columns": self.missing_columns,
        }


def _expected(prefix: str, ids: Iterable[str]) -> list[str]:
    return [f"{prefix}{str(item)}" for item in ids]


def _coverage(
    df: pd.DataFrame, columns: list[str]
) -> tuple[bool, float, list[str]]:
    missing = [c for c in columns if c not in df.columns]
    if missing or not columns:
        return False, 0.0, missing
    values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    fraction = float(finite.mean()) if finite.size else 0.0
    return bool(fraction >= 1.0 - 1.0e-12), fraction, []


def audit_detail_targets(
    detail_path: str | Path,
    *,
    node_ids: Iterable[str],
    storage_node_ids: Iterable[str],
    facility_ids: Iterable[str],
    outfall_node_ids: Iterable[str],
) -> TargetCoverage:
    path = Path(detail_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"detail file is empty: {path}")

    groups = {
        "node_depth": _expected("h:", node_ids),
        "node_flooding_rate": _expected("flood:", node_ids),
        "storage_volume": _expected("storage_volume:", storage_node_ids),
        "managed_facility_flow": _expected("flow:", facility_ids),
        "outfall_flow": _expected("outfall_flow:", outfall_node_ids),
    }
    result: dict[str, bool] = {}
    finite_fraction: dict[str, float] = {}
    missing_columns: dict[str, list[str]] = {}
    for name, columns in groups.items():
        ok, fraction, missing = _coverage(df, columns)
        result[name] = ok
        finite_fraction[name] = fraction
        missing_columns[name] = missing

    return TargetCoverage(
        detail_path=str(path),
        node_depth=result["node_depth"],
        node_flooding_rate=result["node_flooding_rate"],
        storage_volume=result["storage_volume"],
        managed_facility_flow=result["managed_facility_flow"],
        outfall_flow=result["outfall_flow"],
        finite_fraction=finite_fraction,
        missing_columns=missing_columns,
    )


def audit_detail_pool(
    detail_paths: Iterable[str | Path],
    *,
    node_ids: Iterable[str],
    storage_node_ids: Iterable[str],
    facility_ids: Iterable[str],
    outfall_node_ids: Iterable[str],
    sample_lineage_sha256: str | None = None,
) -> dict:
    rows: list[dict] = []
    for path in detail_paths:
        rows.append(
            audit_detail_targets(
                path,
                node_ids=node_ids,
                storage_node_ids=storage_node_ids,
                facility_ids=facility_ids,
                outfall_node_ids=outfall_node_ids,
            ).as_dict()
        )
    control_core_count = sum(bool(row["control_core_complete"]) for row in rows)
    full_count = sum(bool(row["full_hydraulic_complete"]) for row in rows)
    control_core_targets = [
        "node_depth",
        "node_flooding_rate",
        "storage_volume",
        "managed_facility_flow",
    ]
    return {
        "contract": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "detail_count": len(rows),
        "default_target_contract": "CONTROL_CORE",
        "control_core_complete_count": int(control_core_count),
        "full_hydraulic_complete_count": int(full_count),
        "control_core_complete": bool(rows) and control_core_count == len(rows),
        "full_hydraulic_complete": bool(rows) and full_count == len(rows),
        "core_trajectory_complete_count": int(control_core_count),
        "formal_complete_count": int(full_count),
        "core_trajectory_complete": bool(rows) and control_core_count == len(rows),
        "formal_complete": bool(rows) and full_count == len(rows),
        "control_core_required_targets": control_core_targets,
        "full_hydraulic_additional_targets": ["outfall_flow"],
        "core_target_groups": control_core_targets,
        "extended_target_groups": ["outfall_flow"],
        "required_target_groups": control_core_targets,
        "sample_lineage_sha256": str(sample_lineage_sha256 or ""),
        "population_lineage_required_for_formal_admission": True,
        "rows": rows,
    }


def write_audit(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
