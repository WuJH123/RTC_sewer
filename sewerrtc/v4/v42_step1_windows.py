"""Build the formal Step-1 temporal-window manifest from strict Phase R0.

The manifest references authoritative raw detail files and exact 13x5-minute
history anchors.  A downstream trainer must read ``h:<node>`` truth to create
sparse sensor inputs/targets and must read **only** ``setting:<Engineering36>``
for historical actions.  Requested ``a:<facility>`` columns are never an
acceptable fallback in this formal path.
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
    TIME_ATOL_MIN,
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
        physical["eligible_dynamics_pretrain"].fillna(False).astype(bool)
        & physical["mask_readback"].fillna(False).astype(bool)
        & physical["mask_finite"].fillna(False).astype(bool)
    ].copy()
    eligible = eligible[eligible["source_role"].astype(str) != "reserved_evaluation"]
    # NOTE: domain_id filter deliberately removed for Step1.
    # Step1 is temporal sparse-sensor state reconstruction (h truth → full
    # network depth).  It does NOT perform counterfactual comparison and
    # therefore does not require target_no_dwf domain membership.
    # The domain_id gate belongs to Step2 counterfactual admission only.
    # domain_id is preserved in the manifest for provenance tracking.
    domain_filter_applied = False

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
            blocked.append({"physical_identity_sha256": pid, "reason": "missing_split_group"})
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
    audit = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "stage": "step1_window_materialization",
        "eligible_physical_runs": int(len(eligible)),
        "window_count": int(len(frame)),
        "rainfall_group_count": groups,
        "blocked_physical_runs": int(len(blocked)),
        "action_authority": "actual_readback_setting",
        "requested_action_fallback_allowed": False,
        "history_contract": "13x5min_causal",
        "reserved_evaluation_excluded": True,
        "domain_filter_applied": domain_filter_applied,
        "domain_filter_rationale": "step1_state_reconstruction_does_not_require_target_domain",
        "blocked_examples": blocked[:100],
    }
    audit_path = Path(audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return Step1WindowResult(out, audit_path, int(len(frame)), groups)
