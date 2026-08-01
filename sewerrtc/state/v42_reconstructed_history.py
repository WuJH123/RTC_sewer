"""Causal reconstructed-state history for the formal V4.2 online chain.

Step 2 is trained on thirteen 5-minute full-network depth frames.  Online those
frames must come from Step-1 reconstructions that were produced causally at the
corresponding past timestamps; they must never be copied from the current frame
or replaced with authoritative SWMM truth.  This buffer is the formal bridge.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


HISTORY_FRAMES = 13
HISTORY_INTERVAL_MIN = 5.0
TIME_ATOL_MIN = 1.0e-6
CONTRACT_ID = "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1"


@dataclass(frozen=True)
class ReconstructedStateFrame:
    elapsed_min: float
    node_depth_m: np.ndarray
    node_uncertainty_m: np.ndarray
    gat_model_sha256: str


@dataclass(frozen=True)
class ReconstructedHistorySnapshot:
    elapsed_min: np.ndarray
    node_depth_m: np.ndarray
    node_uncertainty_m: np.ndarray
    gat_model_sha256: str
    contract_id: str = CONTRACT_ID

    @property
    def ready(self) -> bool:
        return bool(
            self.node_depth_m.ndim == 2
            and self.node_depth_m.shape[0] == HISTORY_FRAMES
        )


class CausalReconstructedStateBufferV42:
    """Keep exactly the latest causal 13x5-min reconstructed network states."""

    def __init__(
        self,
        *,
        n_nodes: int,
        gat_model_sha256: str,
        frames: int = HISTORY_FRAMES,
        interval_min: float = HISTORY_INTERVAL_MIN,
        atol_min: float = TIME_ATOL_MIN,
    ) -> None:
        if int(n_nodes) <= 0:
            raise ValueError("n_nodes must be positive")
        if int(frames) != HISTORY_FRAMES:
            raise ValueError("formal V4.2 reconstructed history requires 13 frames")
        if not np.isclose(float(interval_min), HISTORY_INTERVAL_MIN, atol=atol_min, rtol=0.0):
            raise ValueError("formal V4.2 reconstructed history requires 5-min cadence")
        if not str(gat_model_sha256).strip():
            raise ValueError("gat_model_sha256 is required")
        self.n_nodes = int(n_nodes)
        self.gat_model_sha256 = str(gat_model_sha256)
        self.frames = int(frames)
        self.interval_min = float(interval_min)
        self.atol_min = float(atol_min)
        self._frames: deque[ReconstructedStateFrame] = deque(maxlen=self.frames)

    @staticmethod
    def _vector(name: str, value: Iterable[float] | np.ndarray, n_nodes: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1 or arr.size != int(n_nodes):
            raise ValueError(f"{name} must be [N] or [1,N] with N={n_nodes}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")
        return arr.copy()

    def clear(self) -> None:
        self._frames.clear()

    def append(
        self,
        *,
        elapsed_min: float,
        node_depth_m: Iterable[float] | np.ndarray,
        node_uncertainty_m: Iterable[float] | np.ndarray,
        gat_model_sha256: str | None = None,
    ) -> None:
        t = float(elapsed_min)
        if not np.isfinite(t):
            raise ValueError("elapsed_min must be finite")
        model_sha = self.gat_model_sha256 if gat_model_sha256 is None else str(gat_model_sha256)
        if model_sha != self.gat_model_sha256:
            raise RuntimeError("reconstructed-history model hash changed after buffer creation")
        depth = self._vector("node_depth_m", node_depth_m, self.n_nodes)
        uncertainty = self._vector("node_uncertainty_m", node_uncertainty_m, self.n_nodes)
        if (uncertainty < 0).any():
            raise ValueError("node_uncertainty_m cannot be negative")

        if self._frames:
            previous = float(self._frames[-1].elapsed_min)
            expected = previous + self.interval_min
            if not np.isclose(t, expected, atol=self.atol_min, rtol=0.0):
                raise RuntimeError(
                    f"causal reconstructed history requires contiguous 5-min frames: "
                    f"expected {expected}, got {t}"
                )
        self._frames.append(
            ReconstructedStateFrame(
                elapsed_min=t,
                node_depth_m=depth,
                node_uncertainty_m=uncertainty,
                gat_model_sha256=model_sha,
            )
        )

    @property
    def ready(self) -> bool:
        if len(self._frames) != self.frames:
            return False
        times = np.asarray([f.elapsed_min for f in self._frames], dtype=np.float64)
        return bool(
            np.allclose(
                np.diff(times),
                self.interval_min,
                atol=self.atol_min,
                rtol=0.0,
            )
        )

    def snapshot(self) -> ReconstructedHistorySnapshot:
        if not self.ready:
            raise RuntimeError(
                "formal Step-2 online state history is not ready; execute the frozen "
                "fallback until 13 causal reconstructed frames are available"
            )
        frames = list(self._frames)
        return ReconstructedHistorySnapshot(
            elapsed_min=np.asarray([f.elapsed_min for f in frames], dtype=np.float64),
            node_depth_m=np.stack([f.node_depth_m for f in frames], axis=0),
            node_uncertainty_m=np.stack(
                [f.node_uncertainty_m for f in frames], axis=0
            ),
            gat_model_sha256=self.gat_model_sha256,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "contract_id": CONTRACT_ID,
            "history_frame_count": self.frames,
            "history_interval_min": self.interval_min,
            "gat_model_sha256": self.gat_model_sha256,
            "ready": self.ready,
            "frame_count_available": len(self._frames),
            "future_hydraulic_truth_used": False,
            "authoritative_swmm_history_used_as_online_input": False,
            "current_frame_repetition_used": False,
        }
