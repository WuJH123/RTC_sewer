"""Build the Step-1 temporal-window manifest from strict Phase R0.

Step 1 is state reconstruction, so hydraulically valid source-domain runs may be
used as *auxiliary pretraining* evidence.  They must never be mixed silently
with the formal Wuhan target-domain validation/calibration population.

The manifest therefore keeps both populations and labels every window as either
``target_formal`` or ``auxiliary_pretrain``.  Downstream formal training must
validate/calibrate only on ``target_formal`` rainfall groups.

The manifest references authoritative raw detail files and exact 13x5-minute
history anchors.  Historical actions are read only from
``setting:<Engineering36>`` readback columns.  Requested ``a:<facility>``
columns are never an acceptable fallback.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .v42_trajectory_builder import (
    HISTORY_INTERVAL_MIN,
    N_HISTORY_FRAMES,
    _load_graph_topology,
)


@dataclass(frozen=True)
class Step1WindowResult:
    manifest_path: Path
    audit_path: Path
    window_count: int
    rainfall_group_count: int


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Parse persisted booleans fail-closed; string ``False`` is not truthy."""
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.fillna(0).astype(float).ne(0.0)
    text = series.fillna("").astype(str).str.strip().str.casefold()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "none", "nan"}
    unknown = sorted(set(text.unique()) - true_values - false_values)
    if unknown:
        raise ValueError(
            f"boolean column {column!r} has unsupported values: {unknown[:10]}"
        )
    return text.isin(true_values)


def _sha_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _anchors(path: Path) -> list[float]:
    header = pd.read_csv(path, nrows=0)
    if "elapsed_min" not in header.columns:
        return []
    elapsed = pd.to_numeric(
        pd.read_csv(path, usecols=["elapsed_min"])["elapsed_min"], errors="coerce"
    ).to_numpy(float)
    if len(elapsed) == 0 or not np.isfinite(elapsed).all():
        return []
    values = np.unique(elapsed)
    value_set = {round(float(x), 6) for x in values}
    out: list[float] = []
    for center in values:
        expected = [
            float(center) - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN
            for i in range(N_HISTORY_FRAMES)
        ]
        if all(round(x, 6) in value_set for x in expected):
            out.append(float(center))
    return out


def build_step1_window_manifest(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    split_manifest: str | Path,
    output_manifest: str | Path,
    audit_output: str | Path,
) -> Step1WindowResult:
    project_root = Path(project_root)
    physical = _read(physical_manifest)
    split = _read(split_manifest)
    if physical.empty or split.empty:
        raise ValueError("strict R0 physical/split manifests cannot be empty")
    required = {
        "physical_identity_sha256",
        "detail_path",
        "branch_role",
        "domain_id",
        "source_role",
        "eligible_dynamics_pretrain",
        "mask_readback",
        "mask_finite",
    }
    missing = required - set(physical.columns)
    if missing:
        raise KeyError(f"physical manifest missing Step1 fields: {sorted(missing)}")

    split_by_id = {
        str(r.physical_identity_sha256): str(r.split_group_key)
        for r in split.itertuples(index=False)
    }
    eligible = physical[
        _bool_series(physical, "eligible_dynamics_pretrain")
        & _bool_series(physical, "mask_readback")
        & _bool_series(physical, "mask_finite")
    ].copy()
    eligible = eligible[
        eligible["source_role"].astype(str) != "reserved_evaluation"
    ].copy()

    domain = eligible["domain_id"].fillna("").astype(str)
    eligible["formal_target_domain"] = domain.str.startswith("target_no_dwf")
    eligible["step1_domain_role"] = np.where(
        eligible["formal_target_domain"],
        "target_formal",
        "auxiliary_pretrain",
    )

    graph = _load_graph_topology(project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    facility_ids = [str(x) for x in graph["facility_ids"]]
    node_sha = _sha_ids(node_ids)
    facility_sha = _sha_ids(facility_ids)

    rows: list[dict] = []
    blocked: list[dict] = []
    for item in eligible.sort_values("physical_identity_sha256").itertuples(index=False):
        pid = str(item.physical_identity_sha256)
        path = Path(str(item.detail_path))
        group = split_by_id.get(pid, "")
        if not group:
            blocked.append(
                {"physical_identity_sha256": pid, "reason": "missing_split_group"}
            )
            continue
        try:
            anchors = _anchors(path)
        except Exception as exc:
            blocked.append(
                {
                    "physical_identity_sha256": pid,
                    "reason": f"anchor_scan_error:{type(exc).__name__}:{exc}",
                }
            )
            continue
        for center in anchors:
            rows.append(
                {
                    "physical_identity_sha256": pid,
                    "detail_path": str(path),
                    "event_id": str(getattr(item, "event_id", "")),
                    "rainfall_sha256": str(getattr(item, "rainfall_sha256", "")),
                    "split_group_key": group,
                    "anchor_min": center,
                    "history_start_min": center - 60.0,
                    "history_end_min": center,
                    "frame_count": N_HISTORY_FRAMES,
                    "frame_interval_min": HISTORY_INTERVAL_MIN,
                    "node_order_sha256": node_sha,
                    "facility_order_sha256": facility_sha,
                    "action_authority": "actual_readback_setting",
                    "requested_action_fallback_allowed": False,
                    "target_authority": "full_network_SWMM_depth_truth",
                    "future_hydraulic_truth_in_input": False,
                    "source_role": str(getattr(item, "source_role", "")),
                    "domain_id": str(getattr(item, "domain_id", "")),
                    "formal_target_domain": bool(
                        getattr(item, "formal_target_domain", False)
                    ),
                    "step1_domain_role": str(
                        getattr(item, "step1_domain_role", "auxiliary_pretrain")
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    out = Path(output_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)

    groups = int(frame["split_group_key"].nunique()) if not frame.empty else 0
    if frame.empty:
        target_windows = target_groups = aux_windows = aux_groups = 0
    else:
        target = frame[frame["step1_domain_role"] == "target_formal"]
        aux = frame[frame["step1_domain_role"] == "auxiliary_pretrain"]
        target_windows = int(len(target))
        target_groups = int(target["split_group_key"].nunique())
        aux_windows = int(len(aux))
        aux_groups = int(aux["split_group_key"].nunique())

    audit = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "stage": "step1_window_materialization",
        "eligible_physical_runs": int(len(eligible)),
        "window_count": int(len(frame)),
        "rainfall_group_count": groups,
        "target_formal_window_count": target_windows,
        "target_formal_rainfall_group_count": target_groups,
        "auxiliary_pretrain_window_count": aux_windows,
        "auxiliary_pretrain_rainfall_group_count": aux_groups,
        "blocked_physical_runs": int(len(blocked)),
        "action_authority": "actual_readback_setting",
        "requested_action_fallback_allowed": False,
        "history_contract": "13x5min_causal",
        "reserved_evaluation_excluded": True,
        "domain_policy": (
            "target_no_dwf authorizes formal validation/calibration; other "
            "hydraulically valid domains are auxiliary pretraining only"
        ),
        "blocked_examples": blocked[:100],
    }
    audit_path = Path(audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    return Step1WindowResult(out, audit_path, int(len(frame)), groups)
