"""Authoritative hydraulic target recorder for the V4.2 paper surrogate.

The legacy detail writer already records depth, head, storage volume, flooding
and managed-facility flow, but it does not expose an explicit outfall-flow
column.  Formal trajectory-first training needs every target in physical units,
so this module defines one fail-closed recorder contract for new SWMM runs.

For an SWMM outfall, ``Node.total_inflow`` is the flow entering the outfall from
the conveyance system and is recorded as ``outfall_flow:<node_id>``.  Missing
objects/properties are NaN, never zero-filled.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def storage_volume_from_depth_v42(
    depth: Any,
    *,
    shape: str,
    functional_params: Sequence[float] | None = None,
    curve_depth: Sequence[float] | None = None,
    curve_area: Sequence[float] | None = None,
) -> np.ndarray:
    """Recover SWMM storage volume from depth for a frozen geometry.

    ``FUNCTIONAL`` uses SWMM's ``Area = A0 + A1 * depth ** A2`` form; the
    input-file parameter order is ``A1 A2 A0``.  ``TABULAR`` curves are
    depth-area points and are integrated piecewise linearly.  This helper is
    only a geometry transform; callers must validate it against recorded
    authoritative volume before using it for target recovery.
    """
    d = np.asarray(depth, dtype=np.float64)
    out = np.full(d.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(d) & (d >= 0.0)
    kind = str(shape).strip().upper()
    if kind == "FUNCTIONAL":
        if functional_params is None or len(functional_params) < 3:
            raise ValueError("FUNCTIONAL storage requires A1, A2, A0")
        a1, a2, a0 = (float(functional_params[i]) for i in range(3))
        if a2 <= -1.0:
            raise ValueError("FUNCTIONAL A2 must be greater than -1")
        out[finite] = a0 * d[finite] + a1 * np.power(d[finite], a2 + 1.0) / (a2 + 1.0)
        return out
    if kind != "TABULAR" or curve_depth is None or curve_area is None:
        raise ValueError(f"unsupported storage geometry: {shape!r}")
    x = np.asarray(curve_depth, dtype=np.float64)
    y = np.asarray(curve_area, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("TABULAR storage requires at least two depth-area points")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0.0):
        raise ValueError("TABULAR storage curve must be finite and strictly increasing")
    if x[0] > 0.0:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([y[0]], y))
    if np.any(d[finite] > x[-1] + 1.0e-8):
        raise ValueError("depth exceeds TABULAR storage curve")
    cumulative = np.zeros_like(x)
    cumulative[1:] = np.cumsum(np.diff(x) * (y[:-1] + y[1:]) * 0.5)
    out[finite] = np.interp(d[finite], x, cumulative)
    return out


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_attr(obj: Any, attr: str) -> float:
    try:
        return _finite_or_nan(getattr(obj, attr))
    except Exception:
        return float("nan")


def record_v42_hydraulic_targets(
    *,
    row: dict[str, Any],
    node_objects: Mapping[str, Any],
    facility_link_objects: Mapping[str, Any],
    graph_node_ids: Sequence[str],
    storage_node_ids: Sequence[str],
    facility_ids: Sequence[str],
    outfall_node_ids: Sequence[str],
) -> dict[str, Any]:
    """Append all formal Step-2 hydraulic targets to one SWMM detail row.

    Required physical columns
    -------------------------
    ``h:<node>``
        Node depth [m].
    ``head:<node>``
        Hydraulic head [m].
    ``flood:<node>``
        Node flooding rate [m3/s].
    ``storage_volume:<storage>``
        Current SWMM storage volume [m3].
    ``flow:<facility>``
        Managed facility/link flow [m3/s].
    ``outfall_flow:<outfall>``
        Total inflow entering the outfall [m3/s].

    The function returns *row* for convenient chaining and deliberately leaves
    unavailable targets as NaN so the downstream target audit can fail closed.
    """
    node_set = set(str(v) for v in graph_node_ids)
    storage_set = set(str(v) for v in storage_node_ids)
    outfall_set = set(str(v) for v in outfall_node_ids)
    if not storage_set.issubset(node_set):
        raise ValueError("storage_node_ids contains nodes outside graph_node_ids")
    if not outfall_set.issubset(node_set):
        raise ValueError("outfall_node_ids contains nodes outside graph_node_ids")

    for node_id in graph_node_ids:
        node_id = str(node_id)
        obj = node_objects.get(node_id)
        if obj is None:
            row[f"h:{node_id}"] = float("nan")
            row[f"head:{node_id}"] = float("nan")
            row[f"flood:{node_id}"] = float("nan")
            if node_id in storage_set:
                row[f"storage_volume:{node_id}"] = float("nan")
            if node_id in outfall_set:
                row[f"outfall_flow:{node_id}"] = float("nan")
            continue
        row[f"h:{node_id}"] = _safe_attr(obj, "depth")
        row[f"head:{node_id}"] = _safe_attr(obj, "head")
        row[f"flood:{node_id}"] = _safe_attr(obj, "flooding")
        if node_id in storage_set:
            row[f"storage_volume:{node_id}"] = _safe_attr(obj, "volume")
        if node_id in outfall_set:
            # PySWMM Node.total_inflow is authoritative node result output.  Do
            # not infer outfall discharge from node flooding or neighbouring
            # link flow because those are not generally equivalent.
            row[f"outfall_flow:{node_id}"] = _safe_attr(obj, "total_inflow")

    for facility_id in facility_ids:
        facility_id = str(facility_id)
        obj = facility_link_objects.get(facility_id)
        row[f"flow:{facility_id}"] = (
            _safe_attr(obj, "flow") if obj is not None else float("nan")
        )
        row[f"setting:{facility_id}"] = (
            _safe_attr(obj, "current_setting")
            if obj is not None
            else float("nan")
        )
    return row


def formal_target_columns(
    *,
    graph_node_ids: Sequence[str],
    storage_node_ids: Sequence[str],
    facility_ids: Sequence[str],
    outfall_node_ids: Sequence[str],
) -> dict[str, list[str]]:
    """Return the exact raw column contract consumed by target admission."""
    return {
        "node_depth": [f"h:{v}" for v in graph_node_ids],
        "node_flooding_rate": [f"flood:{v}" for v in graph_node_ids],
        "storage_volume": [f"storage_volume:{v}" for v in storage_node_ids],
        "managed_facility_flow": [f"flow:{v}" for v in facility_ids],
        "outfall_flow": [f"outfall_flow:{v}" for v in outfall_node_ids],
    }
