"""Bounded-memory streaming dataset for the formal V4.2 Step-1 GAT.

Why this exists
---------------
The Step-1 window manifest can contain hundreds of thousands of highly
overlapping temporal windows.  Materialising every ``13 x N`` sample and
keeping every source ``DataFrame`` alive is neither necessary nor safe.  This
module keeps only manifest metadata in memory, reads one physical detail file at
a time, projects columns by *name* in canonical order, yields tensors lazily,
and then releases the file.

Scientific contracts enforced here
-----------------------------------
* exact 13 x 5-minute causal history;
* one frozen sensor deployment per experiment;
* actual ``setting:<Engineering36>`` readback only;
* missing depth/rainfall/action columns fail closed;
* source/unknown domains remain explicitly auxiliary;
* target-domain rainfall groups can be selected independently for train/val;
* no future hydraulic truth enters the model input.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from sewerrtc.v4.v42_step1_dataset import (
    Step1GraphAssets,
    _build_usecols,
    _detail_extract_window,
    _sensor_layout,
    load_graph_assets,
)
from sewerrtc.v4.v42_trajectory_builder import N_HISTORY_FRAMES


@dataclass(frozen=True)
class Step1StreamingSummary:
    rows: int
    rainfall_groups: int
    physical_runs: int
    detail_files: int
    domain_roles: tuple[str, ...]
    selection_sha256: str


def _read_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    frame = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    if frame.empty:
        raise ValueError("Step1 window manifest is empty")
    required = {
        "detail_path",
        "anchor_min",
        "split_group_key",
        "physical_identity_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Step1 window manifest missing required columns: {missing}")
    if "step1_domain_role" not in frame.columns:
        domain = frame.get("domain_id", pd.Series("", index=frame.index)).fillna("").astype(str)
        frame["step1_domain_role"] = np.where(
            domain.str.startswith("target_no_dwf"),
            "target_formal",
            "auxiliary_pretrain",
        )
    return frame


def _selection_sha(frame: pd.DataFrame) -> str:
    cols = [
        c
        for c in (
            "physical_identity_sha256",
            "detail_path",
            "anchor_min",
            "split_group_key",
            "step1_domain_role",
        )
        if c in frame.columns
    ]
    canon = frame.loc[:, cols].copy()
    canon["anchor_min"] = pd.to_numeric(canon["anchor_min"], errors="raise").map(
        lambda x: f"{float(x):.6f}"
    )
    canon = canon.astype(str).sort_values(cols, kind="mergesort")
    payload = "\n".join("|".join(row) for row in canon.to_numpy(str))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_manifest_rows(
    manifest: pd.DataFrame,
    *,
    domain_roles: Sequence[str] | None = None,
    allowed_groups: Sequence[str] | None = None,
    max_windows_per_group: int | None = None,
    max_windows_per_physical_run: int | None = None,
    sampling_seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic metadata-only selection.

    ``max_windows_per_group`` and ``max_windows_per_physical_run`` are intended
    for auxiliary pretraining where adjacent 5-minute windows are extremely
    redundant.  The selection is deterministic and approximately spans the
    complete event timeline by choosing evenly spaced anchors rather than the
    first N rows.
    """
    frame = manifest.copy()
    if domain_roles is not None:
        roles = {str(x) for x in domain_roles}
        frame = frame[frame["step1_domain_role"].astype(str).isin(roles)].copy()
    if allowed_groups is not None:
        groups = {str(x) for x in allowed_groups}
        frame = frame[frame["split_group_key"].astype(str).isin(groups)].copy()
    if frame.empty:
        return frame

    def _spread(group: pd.DataFrame, limit: int, salt: str) -> pd.DataFrame:
        if len(group) <= limit:
            return group
        ordered = group.sort_values(["anchor_min", "detail_path"], kind="mergesort")
        # Evenly cover the timeline, then rotate the equally spaced choices by a
        # deterministic salt so repeated experiments can use a frozen selection.
        base = np.linspace(0, len(ordered) - 1, num=limit, dtype=int)
        if limit > 1:
            digest = hashlib.sha256(f"{sampling_seed}:{salt}".encode("utf-8")).digest()
            shift = int.from_bytes(digest[:4], "little") % limit
            base = np.roll(base, shift)
        return ordered.iloc[np.sort(base)].copy()

    if max_windows_per_physical_run is not None:
        if max_windows_per_physical_run <= 0:
            raise ValueError("max_windows_per_physical_run must be positive")
        pieces = []
        for pid, grp in frame.groupby("physical_identity_sha256", sort=True):
            pieces.append(_spread(grp, int(max_windows_per_physical_run), f"pid:{pid}"))
        frame = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()

    if max_windows_per_group is not None:
        if max_windows_per_group <= 0:
            raise ValueError("max_windows_per_group must be positive")
        pieces = []
        for group_id, grp in frame.groupby("split_group_key", sort=True):
            pieces.append(_spread(grp, int(max_windows_per_group), f"group:{group_id}"))
        frame = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()

    return frame.sort_values(
        ["detail_path", "anchor_min", "physical_identity_sha256"], kind="mergesort"
    ).reset_index(drop=True)


def summarise_selection(frame: pd.DataFrame) -> Step1StreamingSummary:
    if frame.empty:
        return Step1StreamingSummary(0, 0, 0, 0, tuple(), _selection_sha(frame))
    return Step1StreamingSummary(
        rows=int(len(frame)),
        rainfall_groups=int(frame["split_group_key"].astype(str).nunique()),
        physical_runs=int(frame["physical_identity_sha256"].astype(str).nunique()),
        detail_files=int(frame["detail_path"].astype(str).nunique()),
        domain_roles=tuple(sorted(frame["step1_domain_role"].astype(str).unique())),
        selection_sha256=_selection_sha(frame),
    )


_PROJECTED_CACHE_STATS = {"hits": 0, "misses": 0, "writes": 0, "invalid": 0}


def _projected_cache_location(
    path: str | Path,
    required_columns: Sequence[str],
    *,
    cache_dir: str | Path,
    source_identity: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Return the content-addressed projected-detail cache path and authority."""
    p = Path(path)
    stat = p.stat()
    authority = {
        "source_identity": str(source_identity or p.resolve()),
        "source_path": str(p.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "required_columns": [str(c) for c in required_columns],
    }
    key = hashlib.sha256(
        json.dumps(authority, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Path(cache_dir) / f"{key}.npz", authority


def projected_cache_stats(*, reset: bool = False) -> dict[str, int]:
    stats = {key: int(value) for key, value in _PROJECTED_CACHE_STATS.items()}
    if reset:
        for key in _PROJECTED_CACHE_STATS:
            _PROJECTED_CACHE_STATS[key] = 0
    return stats


def _write_projected_cache(path: Path, frame: pd.DataFrame, authority: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    arrays = {f"column_{index}": frame[column].to_numpy(copy=True) for index, column in enumerate(frame.columns)}
    arrays["__meta__"] = np.asarray(
        json.dumps({"authority": authority, "columns": frame.columns.tolist()}, sort_keys=True)
    )
    try:
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _PROJECTED_CACHE_STATS["writes"] += 1
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_projected_cache(path: Path, authority: dict[str, object]) -> pd.DataFrame | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["__meta__"].item()))
            if meta.get("authority") != authority:
                _PROJECTED_CACHE_STATS["invalid"] += 1
                return None
            columns = [str(column) for column in meta["columns"]]
            frame = pd.DataFrame(
                {column: archive[f"column_{index}"] for index, column in enumerate(columns)}
            )
            return frame.loc[:, columns]
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        _PROJECTED_CACHE_STATS["invalid"] += 1
        return None


def _read_projected_detail(
    path: str | Path,
    required_columns: Sequence[str],
    *,
    cache_dir: str | Path | None = None,
    source_identity: str | None = None,
) -> pd.DataFrame:
    """Read required columns by *name* and return canonical column order.

    Pandas intentionally ignores the element order of ``usecols``.  Therefore
    the returned frame is explicitly reordered after reading.  This avoids the
    silent semantic corruption that can occur when integer projections from one
    historical CSV layout are manually renamed as though another layout had
    been returned.
    """
    p = Path(path)
    if cache_dir is not None:
        cache_path, authority = _projected_cache_location(
            p,
            required_columns,
            cache_dir=cache_dir,
            source_identity=source_identity,
        )
        if cache_path.exists():
            cached = _load_projected_cache(cache_path, authority)
            if cached is not None:
                _PROJECTED_CACHE_STATS["hits"] += 1
                return cached
        _PROJECTED_CACHE_STATS["misses"] += 1
    header = pd.read_csv(p, nrows=0)
    available = set(map(str, header.columns))
    missing = [str(c) for c in required_columns if str(c) not in available]
    if missing:
        raise KeyError(f"formal Step1 detail missing required columns: {missing[:10]}")
    required = [str(c) for c in required_columns]
    frame = pd.read_csv(p, usecols=required, low_memory=False)
    frame = frame.loc[:, required]
    if frame.columns.tolist() != required:
        raise RuntimeError("formal Step1 projected CSV columns are not in canonical order")
    if cache_dir is not None:
        _write_projected_cache(cache_path, frame, authority)
    return frame


class Step1StreamingDataset(IterableDataset):
    """Metadata-backed, bounded-memory iterable dataset.

    The dataset retains only a compact manifest selection.  Each worker receives
    a disjoint set of physical detail files, reads one file at a time, yields all
    requested windows from that file, and then drops the ``DataFrame`` before
    advancing.  Formal runs should begin with ``num_workers=0`` on Windows; the
    worker partitioning is nevertheless deterministic for later optimisation.
    """

    def __init__(
        self,
        *,
        project_root: str | Path,
        manifest_path: str | Path | None = None,
        manifest_frame: pd.DataFrame | None = None,
        sensor_ratio: float = 0.10,
        sensor_layout_seed: int = 42,
        domain_roles: Sequence[str] | None = None,
        allowed_groups: Sequence[str] | None = None,
        max_windows_per_group: int | None = None,
        max_windows_per_physical_run: int | None = None,
        sampling_seed: int = 42,
        shuffle_files: bool = False,
        iteration_seed: int = 42,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        if manifest_frame is None:
            if manifest_path is None:
                raise ValueError("manifest_path or manifest_frame is required")
            manifest = _read_manifest(manifest_path)
        else:
            manifest = manifest_frame.copy()
            if "step1_domain_role" not in manifest.columns:
                domain = manifest.get("domain_id", pd.Series("", index=manifest.index)).fillna("").astype(str)
                manifest["step1_domain_role"] = np.where(
                    domain.str.startswith("target_no_dwf"),
                    "target_formal",
                    "auxiliary_pretrain",
                )
        self.rows = select_manifest_rows(
            manifest,
            domain_roles=domain_roles,
            allowed_groups=allowed_groups,
            max_windows_per_group=max_windows_per_group,
            max_windows_per_physical_run=max_windows_per_physical_run,
            sampling_seed=sampling_seed,
        )
        if self.rows.empty:
            raise ValueError("Step1StreamingDataset selection is empty")
        self.graph: Step1GraphAssets = load_graph_assets(self.project_root)
        self.required_columns = _build_usecols(self.graph.node_ids, self.graph.facility_ids)
        mask, indices, sha = _sensor_layout(
            self.graph.n_nodes, float(sensor_ratio), int(sensor_layout_seed)
        )
        self.sensor_mask = mask.astype(np.float32, copy=False)
        self.sensor_indices = indices.astype(np.int64, copy=False)
        self.sensor_layout_sha256 = sha
        self.mask_history = np.broadcast_to(
            self.sensor_mask[None, :], (N_HISTORY_FRAMES, self.graph.n_nodes)
        ).copy()
        self.summary = summarise_selection(self.rows)
        self.shuffle_files = bool(shuffle_files)
        self.iteration_seed = int(iteration_seed)
        self.epoch = 0
        self.cache_dir = None if cache_dir is None else Path(cache_dir)

    def __len__(self) -> int:
        return int(len(self.rows))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _file_groups(self) -> list[tuple[str, pd.DataFrame]]:
        groups = [
            (str(path), grp.copy())
            for path, grp in self.rows.groupby("detail_path", sort=True)
        ]
        if self.shuffle_files:
            rng = np.random.RandomState(self.iteration_seed + self.epoch)
            rng.shuffle(groups)
            for i, (path, grp) in enumerate(groups):
                order = np.arange(len(grp))
                rng.shuffle(order)
                groups[i] = (path, grp.iloc[order].reset_index(drop=True))
        return groups

    def __iter__(self) -> Iterator[dict[str, object]]:
        file_groups = self._file_groups()
        worker = get_worker_info()
        if worker is not None:
            file_groups = file_groups[worker.id :: worker.num_workers]

        for detail_path, rows in file_groups:
            source_identity = None
            if "physical_identity_sha256" in rows.columns:
                source_identity = "|".join(
                    sorted(rows["physical_identity_sha256"].astype(str).unique())
                )
            detail = _read_projected_detail(
                detail_path,
                self.required_columns,
                cache_dir=self.cache_dir,
                source_identity=source_identity,
            )
            try:
                # Prepare file-level arrays once.  Without this, every window
                # rescans elapsed_min and repeats pandas conversions over the
                # same physical trajectory.
                elapsed_values = pd.to_numeric(
                    detail["elapsed_min"], errors="coerce"
                ).to_numpy(np.float64)
                elapsed_index: dict[float, int] = {}
                duplicate_times: set[float] = set()
                for index, value in enumerate(elapsed_values):
                    key = round(float(value), 6)
                    if key in elapsed_index:
                        duplicate_times.add(key)
                    else:
                        elapsed_index[key] = index
                for key in duplicate_times:
                    elapsed_index.pop(key, None)
                depth_cols = [f"h:{node_id}" for node_id in self.graph.node_ids]
                setting_cols = [f"setting:{facility_id}" for facility_id in self.graph.facility_ids]
                depth_values = detail[depth_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=np.float32)
                rain_values = pd.to_numeric(
                    detail["rainfall_mm_h"], errors="coerce"
                ).to_numpy(dtype=np.float32)
                action_values = detail[setting_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=np.float32)
                prepared_detail = None
                for row in rows.itertuples(index=False):
                    extracted = _detail_extract_window(
                        prepared_detail,
                        float(row.anchor_min),
                        self.graph.node_ids,
                        self.graph.facility_ids,
                        elapsed_values=elapsed_values,
                        elapsed_index=elapsed_index,
                        depth_values=depth_values,
                        rain_values=rain_values,
                        action_values=action_values,
                    )
                    if extracted is None:
                        raise RuntimeError(
                            "formal Step1 manifest window cannot be reconstructed exactly: "
                            f"{detail_path}@{float(row.anchor_min):.6f}"
                        )
                    sparse_depth = extracted["depth_history"] * self.mask_history
                    yield {
                        "sparse_depth_history": torch.from_numpy(sparse_depth.astype(np.float32, copy=False)),
                        "sensor_mask_history": torch.from_numpy(self.mask_history.copy()),
                        "rainfall_history": torch.from_numpy(extracted["rainfall"].astype(np.float32, copy=False)),
                        "historical_actions": torch.from_numpy(extracted["actions"].astype(np.float32, copy=False)),
                        "target_depth": torch.from_numpy(extracted["depth_history"][-1].astype(np.float32, copy=False)),
                        "split_group_key": str(row.split_group_key),
                        "physical_identity_sha256": str(row.physical_identity_sha256),
                        "detail_path": detail_path,
                        "anchor_min": float(row.anchor_min),
                        "step1_domain_role": str(getattr(row, "step1_domain_role", "auxiliary_pretrain")),
                    }
            finally:
                # Make the bounded-memory contract explicit; do not retain the
                # historical DataFrame after all requested windows are yielded.
                del detail


def target_rainfall_groups(manifest_path: str | Path) -> tuple[str, ...]:
    frame = _read_manifest(manifest_path)
    target = frame[frame["step1_domain_role"].astype(str) == "target_formal"]
    return tuple(sorted(target["split_group_key"].astype(str).unique()))


def split_target_groups(
    groups: Sequence[str],
    *,
    split_seed: int = 42,
    validation_group: str | None = None,
    calibration_group: str | None = None,
    reserve_calibration: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Split target rainfall groups without coupling to model/sensor seeds."""
    unique = sorted({str(g) for g in groups})
    if len(unique) < 2:
        raise ValueError("Step1 development split requires at least two target rainfall groups")
    ranked = sorted(
        unique,
        key=lambda g: hashlib.sha256(f"{int(split_seed)}:{g}".encode("utf-8")).hexdigest(),
    )
    if validation_group is not None:
        val = str(validation_group)
        if val not in unique:
            raise ValueError(f"validation group {val!r} is not a target rainfall group")
    else:
        val = ranked[0]

    remaining = [g for g in ranked if g != val]
    cal: str | None = None
    if reserve_calibration:
        if calibration_group is not None:
            cal = str(calibration_group)
            if cal == val or cal not in remaining:
                raise ValueError("calibration group must be a distinct target rainfall group")
        elif len(remaining) >= 2:
            cal = remaining[0]
        else:
            raise ValueError("formal Step1 train/validation/calibration requires >=3 target groups")
    train = tuple(g for g in remaining if g != cal)
    if not train:
        raise ValueError("Step1 target split has no training rainfall group")
    return {
        "train": tuple(sorted(train)),
        "validation": (val,),
        "calibration": tuple() if cal is None else (cal,),
    }


__all__ = [
    "Step1StreamingDataset",
    "Step1StreamingSummary",
    "select_manifest_rows",
    "split_target_groups",
    "summarise_selection",
    "target_rainfall_groups",
    "_read_projected_detail",
    "_projected_cache_location",
    "projected_cache_stats",
]
