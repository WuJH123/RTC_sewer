"""Action-authority and hydraulic-response classification for Gate 5R."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class ActionAuthorityReport:
    authority_class: str
    command_realized: bool
    locally_responsive: bool
    kpi_responsive: bool
    requested_action_distance: float
    actual_action_distance: float
    local_hydraulic_distance: float
    kpi_distance: float
    response_lag_min: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _max_distance(
    reference: pd.DataFrame, candidate: pd.DataFrame, columns: list[str]
) -> float:
    shared = [
        column
        for column in columns
        if column in reference.columns and column in candidate.columns
    ]
    if not shared:
        return 0.0
    rows = min(len(reference), len(candidate))
    if rows == 0:
        return 0.0
    ref = reference.iloc[:rows][shared].apply(pd.to_numeric, errors="coerce").to_numpy()
    cand = candidate.iloc[:rows][shared].apply(pd.to_numeric, errors="coerce").to_numpy()
    distance = np.abs(cand - ref)
    return float(np.nanmax(distance)) if np.isfinite(distance).any() else 0.0


def classify_action_authority(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    facility_ids: list[str],
    tolerance: float = 1e-9,
) -> ActionAuthorityReport:
    """Classify a candidate into the mutually exclusive Gate 5R A-E classes."""
    requested_columns = [f"requested_setting:{fid}" for fid in facility_ids]
    actual_columns = [f"a:{fid}" for fid in facility_ids]
    flow_columns = [f"flow:{fid}" for fid in facility_ids]
    local_columns = [
        column
        for column in set(reference.columns).union(candidate.columns)
        if column.startswith(("flow:", "head:", "h:", "storage_volume:"))
    ]
    flood_columns = [
        column
        for column in set(reference.columns).union(candidate.columns)
        if column.startswith("flood:")
    ]

    requested_distance = _max_distance(
        reference, candidate, requested_columns
    )
    actual_distance = _max_distance(reference, candidate, actual_columns)
    local_distance = _max_distance(reference, candidate, local_columns)
    kpi_distance = _max_distance(reference, candidate, flood_columns)
    command_realized = actual_distance > tolerance
    locally_responsive = local_distance > tolerance
    kpi_responsive = kpi_distance > tolerance

    if requested_distance > tolerance and not command_realized:
        authority_class = "A_requested_diff_actual_equal"
    elif command_realized:
        flow_magnitude = 0.0
        for frame in (reference, candidate):
            present = [column for column in flow_columns if column in frame.columns]
            if present and len(frame):
                values = frame[present].apply(pd.to_numeric, errors="coerce").to_numpy()
                if np.isfinite(values).any():
                    flow_magnitude = max(flow_magnitude, float(np.nanmax(np.abs(values))))
        if flow_magnitude <= tolerance:
            authority_class = "B_realized_no_hydraulic_opportunity"
            locally_responsive = False
        elif locally_responsive and not kpi_responsive:
            authority_class = "C_local_response_kpi_flat"
        elif not locally_responsive:
            authority_class = "D_realized_hydraulically_flat"
        else:
            authority_class = "E_kpi_responsive"
    else:
        authority_class = "D_realized_hydraulically_flat"

    response_lag: float | None = None
    if locally_responsive and "elapsed_min" in reference.columns and "elapsed_min" in candidate.columns:
        rows = min(len(reference), len(candidate))
        shared_local = [
            column
            for column in local_columns
            if column in reference.columns and column in candidate.columns
        ]
        if rows and shared_local:
            ref = reference.iloc[:rows][shared_local].apply(pd.to_numeric, errors="coerce").to_numpy()
            cand = candidate.iloc[:rows][shared_local].apply(pd.to_numeric, errors="coerce").to_numpy()
            changed_rows = np.flatnonzero(np.any(np.abs(cand - ref) > tolerance, axis=1))
            if len(changed_rows):
                elapsed = pd.to_numeric(
                    candidate.iloc[:rows]["elapsed_min"], errors="coerce"
                ).to_numpy()
                response_lag = float(elapsed[changed_rows[0]] - elapsed[0])

    return ActionAuthorityReport(
        authority_class=authority_class,
        command_realized=bool(command_realized),
        locally_responsive=bool(locally_responsive),
        kpi_responsive=bool(kpi_responsive),
        requested_action_distance=requested_distance,
        actual_action_distance=actual_distance,
        local_hydraulic_distance=local_distance,
        kpi_distance=kpi_distance,
        response_lag_min=response_lag,
    )
