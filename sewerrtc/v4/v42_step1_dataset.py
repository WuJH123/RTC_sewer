"""Step-1 temporal sparse GAT dataset for formal V4.2 training.

Reads the Step-1 window manifest and detail CSVs to produce training
samples for :class:`TemporalSparseGATReconstructorV42`.

Each sample provides:
- 13×5-min sparse depth history (sensor-masked)
- sensor mask history
- rainfall history
- historical actual actions (setting readback)
- full-network depth target at anchor time
- graph topology and static attributes

Sensor masks are generated deterministically per window (seeded by a hash
of the window identity) so that train/val masks are reproducible and
no future hydraulic truth leaks into the input.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sewerrtc.v4.v42_trajectory_builder import (
    HISTORY_INTERVAL_MIN,
    N_FACILITIES,
    N_HISTORY_FRAMES,
    TIME_ATOL_MIN,
    _load_graph_topology,
)

logger = logging.getLogger(__name__)


@dataclass
class Step1GraphAssets:
    """Precomputed graph topology and static features."""

    n_nodes: int
    n_edges: int
    n_facilities: int
    node_ids: list[str]
    facility_ids: list[str]
    edge_index: np.ndarray  # [2, E] int64
    node_static: np.ndarray  # [N, F_node] float32 (normalised)
    link_static: np.ndarray  # [E, F_edge] float32
    action_node_map: np.ndarray  # [F, N] float32
    node_static_raw: np.ndarray  # before normalisation
    node_static_mean: np.ndarray
    node_static_std: np.ndarray


@dataclass
class Step1Sample:
    """A single training/evaluation sample."""

    sparse_depth_history: np.ndarray  # [13, N]
    sensor_mask_history: np.ndarray  # [13, N]
    rainfall_history: np.ndarray  # [13]
    historical_actions: np.ndarray  # [13, F]
    target_depth: np.ndarray  # [N]
    split_group: str
    window_key: str


@dataclass
class Step1DatasetBuild:
    samples: list[Step1Sample]
    graph: Step1GraphAssets
    warnings: list[str] = field(default_factory=list)
    skipped: int = 0


def load_graph_assets(project_root: Path) -> Step1GraphAssets:
    """Load topology and build normalised static features."""
    from sewerrtc.v4.v42_trajectory_builder import _parse_inp_topology
    from sewerrtc.graph.graph_builder import build_node_link_graph

    project_root = Path(project_root)
    graph = _load_graph_topology(project_root)

    # Parse link types for edge features
    inp_path = project_root / "data" / "wuhan_v8_storage_retrofit.inp"
    nodes_df, links_df = _parse_inp_topology(inp_path)
    nodes_enriched, links_enriched, _ = build_node_link_graph(nodes_df, links_df)

    # Link static: one-hot [is_conduit, is_pump, is_orifice, is_weir]
    link_types = links_enriched["link_type"].astype(str).str.lower().values
    link_static = np.stack([
        (link_types == "conduit"),
        (link_types == "pump"),
        (link_types == "orifice"),
        (link_types == "weir"),
    ], axis=1).astype(np.float32)

    # Node static normalisation
    node_static_raw = graph["node_static"].copy()
    mean = node_static_raw.mean(axis=0, keepdims=True)
    std = node_static_raw.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    node_static_norm = (node_static_raw - mean) / std

    return Step1GraphAssets(
        n_nodes=graph["n_nodes"],
        n_edges=graph["n_edges"],
        n_facilities=graph["n_facilities"],
        node_ids=graph["node_ids"],
        facility_ids=graph["facility_ids"],
        edge_index=graph["edge_index"],
        node_static=node_static_norm,
        link_static=link_static,
        action_node_map=graph["action_node_map"],
        node_static_raw=node_static_raw,
        node_static_mean=mean,
        node_static_std=std,
    )


def _build_usecols(node_ids: list[str], facility_ids: list[str]) -> list[str]:
    """Pre-compute the minimal set of columns to read from detail CSV."""
    cols = ["elapsed_min", "rainfall_mm_h"]
    for nid in node_ids:
        cols.append(f"h:{nid}")
    for fid in facility_ids:
        cols.append(f"setting:{fid}")
    return cols


def _detail_extract_window(
    detail: pd.DataFrame,
    anchor_min: float,
    node_ids: list[str],
    facility_ids: list[str],
    *,
    h_col_index: dict[str, int] | None = None,
    s_col_index: dict[str, int] | None = None,
    rain_col_idx: int | None = None,
) -> dict[str, np.ndarray] | None:
    """Extract 13 history frames + target from a detail CSV at exact times."""
    elapsed = detail.iloc[:, 0].to_numpy(np.float64)  # elapsed_min is always first
    history_times = [
        anchor_min - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN
        for i in range(N_HISTORY_FRAMES)
    ]

    indices: list[int] = []
    for t in history_times:
        matches = np.flatnonzero(np.isclose(elapsed, t, atol=TIME_ATOL_MIN, rtol=0.0))
        if len(matches) != 1:
            return None
        indices.append(int(matches[0]))

    # Vectorised extraction using pre-computed column indices
    idx_arr = np.array(indices, dtype=np.int64)

    # Depth: [13, N]
    n_nodes = len(node_ids)
    depth_history = np.zeros((N_HISTORY_FRAMES, n_nodes), dtype=np.float32)
    if h_col_index is not None:
        for ni, nid in enumerate(node_ids):
            ci = h_col_index.get(nid.casefold())
            if ci is not None:
                depth_history[:, ni] = detail.iloc[idx_arr, ci].to_numpy(np.float32)

    # Rainfall: [13]
    if rain_col_idx is not None:
        rainfall = detail.iloc[idx_arr, rain_col_idx].to_numpy(np.float32)
    else:
        rainfall = np.zeros(N_HISTORY_FRAMES, dtype=np.float32)

    # Actions (setting readback): [13, F]
    n_fac = len(facility_ids)
    actions = np.zeros((N_HISTORY_FRAMES, n_fac), dtype=np.float32)
    if s_col_index is not None:
        for fi, fid in enumerate(facility_ids):
            ci = s_col_index.get(fid.casefold())
            if ci is not None:
                actions[:, fi] = detail.iloc[idx_arr, ci].to_numpy(np.float32)

    if not np.isfinite(depth_history).all():
        return None
    if not np.isfinite(rainfall).all():
        return None
    if not np.isfinite(actions).all():
        return None

    return {
        "depth_history": depth_history,
        "rainfall": rainfall,
        "actions": actions,
    }


def _sensor_mask_for_window(
    window_key: str,
    n_nodes: int,
    sensor_ratio: float,
    rng_seed: int,
) -> np.ndarray:
    """Generate a deterministic sensor mask for a window.

    The mask is constant across all 13 frames (sensor placement is fixed
    for a given monitoring configuration).  Different windows get different
    random subsets, seeded deterministically.
    """
    seed_bytes = hashlib.sha256(
        f"{window_key}:{rng_seed}".encode()
    ).digest()
    seed_int = int.from_bytes(seed_bytes[:4], "big") % (2**31)
    rng = np.random.RandomState(seed_int)
    n_sensors = max(1, int(round(n_nodes * sensor_ratio)))
    sensor_indices = rng.choice(n_nodes, size=n_sensors, replace=False)
    mask = np.zeros(n_nodes, dtype=np.float32)
    mask[sensor_indices] = 1.0
    return mask


def build_step1_dataset(
    *,
    project_root: Path,
    manifest_path: Path,
    sensor_ratio: float = 0.10,
    rng_seed: int = 42,
    max_samples: int | None = None,
) -> Step1DatasetBuild:
    """Build the Step-1 training dataset from the window manifest."""
    project_root = Path(project_root)
    manifest = pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet" else pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError("Step1 window manifest is empty")

    graph = load_graph_assets(project_root)
    samples: list[Step1Sample] = []
    warnings: list[str] = []
    skipped = 0

    # Pre-compute column names for selective CSV reading
    usecols = _build_usecols(graph.node_ids, graph.facility_ids)
    usecols_set = frozenset(usecols)

    # Pre-compute column index lookups (built once per detail file)
    _col_lookup_cache: dict[str, tuple[dict[str, int], dict[str, int], int | None]] = {}

    # Cache detail CSVs to avoid re-reading
    detail_cache: dict[str, pd.DataFrame] = {}

    for wi, row in enumerate(manifest.itertuples(index=False)):
        if max_samples is not None and len(samples) >= max_samples:
            break
        detail_path = str(row.detail_path)
        anchor_min = float(row.anchor_min)
        split_group = str(row.split_group_key)
        window_key = f"{row.physical_identity_sha256}:{anchor_min}"

        if detail_path not in detail_cache:
            try:
                detail_cache[detail_path] = pd.read_csv(
                    detail_path, usecols=lambda c, _s=usecols_set: c in _s,
                    low_memory=False,
                )
                # Build column index lookup
                d = detail_cache[detail_path]
                h_idx: dict[str, int] = {}
                s_idx: dict[str, int] = {}
                rain_ci: int | None = None
                for ci, c in enumerate(d.columns):
                    if c.startswith("h:"):
                        h_idx[c[2:].casefold()] = ci
                    elif c.startswith("setting:"):
                        s_idx[c[8:].casefold()] = ci
                    elif c == "rainfall_mm_h":
                        rain_ci = ci
                _col_lookup_cache[detail_path] = (h_idx, s_idx, rain_ci)
            except Exception as exc:
                warnings.append(f"detail_read_error:{detail_path}:{exc}")
                skipped += 1
                continue

        detail = detail_cache[detail_path]
        lookup = _col_lookup_cache.get(detail_path)
        h_idx, s_idx, rain_ci = lookup if lookup else ({}, {}, None)

        extracted = _detail_extract_window(
            detail, anchor_min, graph.node_ids, graph.facility_ids,
            h_col_index=h_idx, s_col_index=s_idx, rain_col_idx=rain_ci,
        )
        if extracted is None:
            skipped += 1
            continue

        # Sensor mask: constant across 13 frames
        mask_1d = _sensor_mask_for_window(window_key, graph.n_nodes, sensor_ratio, rng_seed)
        mask_history = np.broadcast_to(
            mask_1d[None, :], (N_HISTORY_FRAMES, graph.n_nodes)
        ).copy()
        sparse_depth = extracted["depth_history"] * mask_history

        samples.append(Step1Sample(
            sparse_depth_history=sparse_depth,
            sensor_mask_history=mask_history,
            rainfall_history=extracted["rainfall"],
            historical_actions=extracted["actions"],
            target_depth=extracted["depth_history"][-1],  # anchor-time full depth
            split_group=split_group,
            window_key=window_key,
        ))

    return Step1DatasetBuild(
        samples=samples,
        graph=graph,
        warnings=warnings,
        skipped=skipped,
    )


class Step1TorchDataset(Dataset):
    """PyTorch Dataset wrapper for Step-1 samples."""

    def __init__(self, samples: Sequence[Step1Sample], graph: Step1GraphAssets) -> None:
        self.samples = list(samples)
        self.graph = graph

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "sparse_depth_history": torch.from_numpy(s.sparse_depth_history),
            "sensor_mask_history": torch.from_numpy(s.sensor_mask_history),
            "rainfall_history": torch.from_numpy(s.rainfall_history),
            "historical_actions": torch.from_numpy(s.historical_actions),
            "target_depth": torch.from_numpy(s.target_depth),
        }
