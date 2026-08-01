"""Formal Step-1 dataset for the V4.2 temporal sparse GAT.

Scientific rules enforced here:

* 13 causal 5-min history frames ending at the reconstruction time;
* a *fixed* sensor deployment for an experiment (not a different sensor set for
  every window);
* actual Engineering36 ``setting:`` readback only;
* no missing hydraulic/action/rainfall column is silently replaced with zero;
* source-domain windows are retained with an explicit auxiliary-pretraining
  role, while target-domain windows remain identifiable for formal validation;
* graph edge features are aligned to edge order and include basic hydraulic
  physical attributes in addition to link type;
* graph messages are bidirectional because current sewer states can be coupled
  upstream and downstream by backwater/control effects.  Edge direction is an
  explicit feature rather than being inferred from row order.
"""
from __future__ import annotations

import hashlib
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
    NODE_STATIC_COLS,
    N_HISTORY_FRAMES,
    TIME_ATOL_MIN,
    _load_graph_topology,
    _parse_inp_topology,
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
    edge_index: np.ndarray  # [2,E_bidir] int64
    node_static: np.ndarray  # [N,F_node] float32, normalised once
    link_static: np.ndarray  # [E_bidir,F_edge] float32
    action_node_map: np.ndarray  # [F,N] float32
    node_static_raw: np.ndarray
    node_static_mean: np.ndarray
    node_static_std: np.ndarray


@dataclass
class Step1Sample:
    """A single training/evaluation sample."""

    sparse_depth_history: np.ndarray
    sensor_mask_history: np.ndarray
    rainfall_history: np.ndarray
    historical_actions: np.ndarray
    target_depth: np.ndarray
    split_group: str
    window_key: str
    event_id: str = ""
    physical_identity_sha256: str = ""
    domain_id: str = ""
    step1_domain_role: str = "auxiliary_pretrain"


@dataclass
class Step1DatasetBuild:
    samples: list[Step1Sample]
    graph: Step1GraphAssets
    warnings: list[str] = field(default_factory=list)
    skipped: int = 0
    sensor_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    sensor_layout_sha256: str = ""


def _safe_float(parts: list[str], index: int, default: float = 0.0) -> float:
    try:
        value = float(parts[index])
        return value if np.isfinite(value) else default
    except (IndexError, TypeError, ValueError):
        return default


def _parse_edge_physics(inp_path: Path) -> dict[str, dict[str, float]]:
    """Read stable numeric edge attributes from the frozen SWMM INP."""
    attrs: dict[str, dict[str, float]] = {}
    xsections: dict[str, tuple[float, float]] = {}
    section = ""
    with inp_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            text = raw.strip()
            if not text or text.startswith(";"):
                continue
            if text.startswith("["):
                section = text.split("]", 1)[0].lstrip("[").strip().upper()
                continue
            parts = text.split()
            if not parts:
                continue
            lid = parts[0].casefold()
            if section == "CONDUITS":
                attrs.setdefault(lid, {}).update(
                    length=_safe_float(parts, 3),
                    roughness=_safe_float(parts, 4),
                    offset=_safe_float(parts, 5),
                    qcoeff=0.0,
                )
            elif section == "ORIFICES":
                attrs.setdefault(lid, {}).update(
                    length=0.0,
                    roughness=0.0,
                    offset=_safe_float(parts, 4),
                    qcoeff=_safe_float(parts, 5),
                )
            elif section == "WEIRS":
                attrs.setdefault(lid, {}).update(
                    length=0.0,
                    roughness=0.0,
                    offset=_safe_float(parts, 4),
                    qcoeff=_safe_float(parts, 5),
                )
            elif section in {"PUMPS", "OUTLETS"}:
                attrs.setdefault(lid, {}).update(
                    length=0.0, roughness=0.0, offset=0.0, qcoeff=0.0
                )
            elif section == "XSECTIONS":
                xsections[lid] = (
                    _safe_float(parts, 2),
                    _safe_float(parts, 3),
                )
    for lid, (geom1, geom2) in xsections.items():
        attrs.setdefault(lid, {}).update(geom1=geom1, geom2=geom2)
    return attrs


def _normalise_numeric(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return ((values - mean) / std).astype(np.float32)


def load_graph_assets(project_root: Path) -> Step1GraphAssets:
    """Load a bidirectional, physically attributed graph in canonical order."""
    from sewerrtc.graph.graph_builder import build_node_link_graph

    project_root = Path(project_root)
    inp_path = project_root / "data" / "wuhan_v8_storage_retrofit.inp"
    graph = _load_graph_topology(project_root)
    nodes_df, links_df = _parse_inp_topology(inp_path)
    nodes_enriched, links_enriched, forward_edge_index = build_node_link_graph(
        nodes_df, links_df
    )

    node_ids = [str(x) for x in nodes_enriched["node_id"].tolist()]
    if node_ids != [str(x) for x in graph["node_ids"]]:
        raise RuntimeError("Step1 graph node order differs from canonical V4.2 graph")

    node_static_raw = (
        nodes_enriched[NODE_STATIC_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    )
    node_mean = node_static_raw.mean(axis=0, keepdims=True)
    node_std = node_static_raw.std(axis=0, keepdims=True)
    node_std[node_std < 1.0e-6] = 1.0
    node_static = ((node_static_raw - node_mean) / node_std).astype(np.float32)

    valid_links = links_enriched[
        links_enriched["from_index"].notna() & links_enriched["to_index"].notna()
    ].copy()
    if len(valid_links) != int(forward_edge_index.shape[1]):
        raise RuntimeError("link feature rows do not align with graph edge_index")

    link_types = valid_links["link_type"].astype(str).str.lower().to_numpy()
    type_order = ("conduit", "pump", "orifice", "weir", "outlet")
    type_onehot = np.stack(
        [(link_types == kind) for kind in type_order], axis=1
    ).astype(np.float32)

    physics = _parse_edge_physics(inp_path)
    numeric = np.zeros((len(valid_links), 6), dtype=np.float32)
    for i, lid in enumerate(valid_links["link_id"].astype(str)):
        meta = physics.get(lid.casefold(), {})
        numeric[i] = np.asarray(
            [
                meta.get("length", 0.0),
                meta.get("roughness", 0.0),
                meta.get("geom1", 0.0),
                meta.get("geom2", 0.0),
                meta.get("offset", 0.0),
                meta.get("qcoeff", 0.0),
            ],
            dtype=np.float32,
        )
    numeric = _normalise_numeric(numeric)

    # Explicit direction features: [is_forward, is_reverse].
    forward_attr = np.concatenate(
        [type_onehot, numeric, np.tile([[1.0, 0.0]], (len(valid_links), 1))],
        axis=1,
    ).astype(np.float32)
    reverse_attr = np.concatenate(
        [type_onehot, numeric, np.tile([[0.0, 1.0]], (len(valid_links), 1))],
        axis=1,
    ).astype(np.float32)
    edge_index = np.concatenate(
        [forward_edge_index, forward_edge_index[[1, 0], :]], axis=1
    ).astype(np.int64)
    link_static = np.concatenate([forward_attr, reverse_attr], axis=0)

    return Step1GraphAssets(
        n_nodes=int(graph["n_nodes"]),
        n_edges=int(edge_index.shape[1]),
        n_facilities=int(graph["n_facilities"]),
        node_ids=node_ids,
        facility_ids=[str(x) for x in graph["facility_ids"]],
        edge_index=edge_index,
        node_static=node_static,
        link_static=link_static,
        action_node_map=np.asarray(graph["action_node_map"], dtype=np.float32),
        node_static_raw=node_static_raw,
        node_static_mean=node_mean,
        node_static_std=node_std,
    )


def _build_usecols(node_ids: list[str], facility_ids: list[str]) -> list[str]:
    cols = ["elapsed_min", "rainfall_mm_h"]
    cols.extend(f"h:{nid}" for nid in node_ids)
    cols.extend(f"setting:{fid}" for fid in facility_ids)
    return cols


def _detail_extract_window(
    detail: pd.DataFrame,
    anchor_min: float,
    node_ids: list[str],
    facility_ids: list[str],
) -> dict[str, np.ndarray] | None:
    """Extract exact 13-frame causal input and anchor-time depth target."""
    required = ["elapsed_min", "rainfall_mm_h"]
    required.extend(f"h:{nid}" for nid in node_ids)
    required.extend(f"setting:{fid}" for fid in facility_ids)
    missing = [col for col in required if col not in detail.columns]
    if missing:
        raise KeyError(f"formal Step1 detail missing required columns: {missing[:10]}")

    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(
        np.float64
    )
    if not np.isfinite(elapsed).all():
        return None
    history_times = [
        anchor_min - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN
        for i in range(N_HISTORY_FRAMES)
    ]
    indices: list[int] = []
    for t in history_times:
        matches = np.flatnonzero(
            np.isclose(elapsed, t, atol=TIME_ATOL_MIN, rtol=0.0)
        )
        if len(matches) != 1:
            return None
        indices.append(int(matches[0]))
    idx = np.asarray(indices, dtype=np.int64)

    depth_cols = [f"h:{nid}" for nid in node_ids]
    setting_cols = [f"setting:{fid}" for fid in facility_ids]
    depth = detail.iloc[idx][depth_cols].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    rain = pd.to_numeric(
        detail.iloc[idx]["rainfall_mm_h"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    actions = detail.iloc[idx][setting_cols].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    if not (
        np.isfinite(depth).all()
        and np.isfinite(rain).all()
        and np.isfinite(actions).all()
    ):
        return None
    return {"depth_history": depth, "rainfall": rain, "actions": actions}


def _sensor_layout(
    n_nodes: int,
    sensor_ratio: float,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Create one frozen deployment for the complete experiment."""
    if not (0.0 < float(sensor_ratio) <= 1.0):
        raise ValueError("sensor_ratio must lie in (0,1]")
    rng = np.random.RandomState(int(rng_seed))
    n_sensors = max(1, int(round(n_nodes * float(sensor_ratio))))
    indices = np.sort(rng.choice(n_nodes, size=n_sensors, replace=False)).astype(
        np.int64
    )
    mask = np.zeros(n_nodes, dtype=np.float32)
    mask[indices] = 1.0
    sha = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
    return mask, indices, sha


def _sensor_mask_for_window(
    window_key: str,
    n_nodes: int,
    sensor_ratio: float,
    rng_seed: int,
) -> np.ndarray:
    """Backward-compatible helper; formal semantics use a fixed layout.

    ``window_key`` is intentionally ignored.  Keeping this name avoids breaking
    diagnostics that imported the old helper while removing the scientifically
    invalid per-window sensor relocation.
    """
    del window_key
    return _sensor_layout(n_nodes, sensor_ratio, rng_seed)[0]


def build_step1_dataset(
    *,
    project_root: Path,
    manifest_path: Path,
    sensor_ratio: float = 0.10,
    rng_seed: int = 42,
    max_samples: int | None = None,
) -> Step1DatasetBuild:
    """Build Step1 samples with one frozen sensor deployment."""
    project_root = Path(project_root)
    manifest = (
        pd.read_parquet(manifest_path)
        if manifest_path.suffix.lower() == ".parquet"
        else pd.read_csv(manifest_path)
    )
    if manifest.empty:
        raise ValueError("Step1 window manifest is empty")

    graph = load_graph_assets(project_root)
    mask_1d, sensor_indices, layout_sha = _sensor_layout(
        graph.n_nodes, sensor_ratio, rng_seed
    )
    mask_history = np.broadcast_to(
        mask_1d[None, :], (N_HISTORY_FRAMES, graph.n_nodes)
    ).copy()

    samples: list[Step1Sample] = []
    warnings: list[str] = []
    skipped = 0
    usecols = _build_usecols(graph.node_ids, graph.facility_ids)
    usecols_set = frozenset(usecols)
    detail_cache: dict[str, pd.DataFrame] = {}

    for row in manifest.itertuples(index=False):
        if max_samples is not None and len(samples) >= max_samples:
            break
        detail_path = str(row.detail_path)
        anchor_min = float(row.anchor_min)
        split_group = str(row.split_group_key)
        pid = str(row.physical_identity_sha256)
        window_key = f"{pid}:{anchor_min}"

        if detail_path not in detail_cache:
            try:
                detail = pd.read_csv(
                    detail_path,
                    usecols=lambda c, _s=usecols_set: c in _s,
                    low_memory=False,
                )
                missing = [col for col in usecols if col not in detail.columns]
                if missing:
                    raise KeyError(
                        f"missing formal Step1 columns: {missing[:10]}"
                    )
                detail_cache[detail_path] = detail
            except Exception as exc:
                warnings.append(
                    f"detail_read_error:{detail_path}:{type(exc).__name__}:{exc}"
                )
                skipped += 1
                continue

        try:
            extracted = _detail_extract_window(
                detail_cache[detail_path],
                anchor_min,
                graph.node_ids,
                graph.facility_ids,
            )
        except Exception as exc:
            warnings.append(
                f"window_extract_error:{detail_path}:{anchor_min}:"
                f"{type(exc).__name__}:{exc}"
            )
            skipped += 1
            continue
        if extracted is None:
            skipped += 1
            continue

        sparse_depth = extracted["depth_history"] * mask_history
        domain_id = str(getattr(row, "domain_id", ""))
        role = str(
            getattr(
                row,
                "step1_domain_role",
                "target_formal"
                if domain_id.startswith("target_no_dwf")
                else "auxiliary_pretrain",
            )
        )
        samples.append(
            Step1Sample(
                sparse_depth_history=sparse_depth,
                sensor_mask_history=mask_history.copy(),
                rainfall_history=extracted["rainfall"],
                historical_actions=extracted["actions"],
                target_depth=extracted["depth_history"][-1],
                split_group=split_group,
                window_key=window_key,
                event_id=str(getattr(row, "event_id", "")),
                physical_identity_sha256=pid,
                domain_id=domain_id,
                step1_domain_role=role,
            )
        )

    return Step1DatasetBuild(
        samples=samples,
        graph=graph,
        warnings=warnings,
        skipped=skipped,
        sensor_indices=sensor_indices,
        sensor_layout_sha256=layout_sha,
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
