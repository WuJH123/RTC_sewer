"""Stable mappings between global 109-action and frozen 36-action tensors."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np


def _clean(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value).removeprefix("a:") for value in values)


@dataclass(frozen=True)
class CanonicalActionOrder:
    """Canonical 36-action order derived exclusively from the 109 registry."""

    global_ids: tuple[str, ...]
    canonical_ids: tuple[str, ...]
    old_mask_ids: tuple[str, ...]
    canonical_global_indices: tuple[int, ...]
    canonical_to_old_indices: tuple[int, ...]
    old_to_canonical_indices: tuple[int, ...]

    @classmethod
    def from_global_registry(cls, global_ids: Sequence[str], old_mask_ids: Sequence[str]) -> "CanonicalActionOrder":
        global_clean = _clean(global_ids)
        old_clean = _clean(old_mask_ids)
        if len(global_clean) != 109 or len(set(global_clean)) != 109:
            raise ValueError("global actuator registry must contain 109 unique IDs")
        if len(old_clean) != 36 or len(set(old_clean)) != 36:
            raise ValueError("frozen 36-asset mask must contain 36 unique IDs")
        global_index = {aid: i for i, aid in enumerate(global_clean)}
        missing = sorted(set(old_clean).difference(global_index))
        if missing:
            raise ValueError(f"mask IDs absent from global registry: {missing}")
        canonical = tuple(aid for aid in global_clean if aid in set(old_clean))
        if len(canonical) != 36:
            raise ValueError("canonical projection did not retain exactly 36 IDs")
        old_index = {aid: i for i, aid in enumerate(old_clean)}
        canonical_to_old = tuple(old_index[aid] for aid in canonical)
        canonical_index = {aid: i for i, aid in enumerate(canonical)}
        old_to_canonical = tuple(canonical_index[aid] for aid in old_clean)
        return cls(
            global_ids=global_clean,
            canonical_ids=canonical,
            old_mask_ids=old_clean,
            canonical_global_indices=tuple(global_index[aid] for aid in canonical),
            canonical_to_old_indices=canonical_to_old,
            old_to_canonical_indices=old_to_canonical,
        )

    @property
    def manifest_hash(self) -> str:
        payload = {
            "global_ids": self.global_ids,
            "canonical_ids": self.canonical_ids,
            "old_mask_ids": self.old_mask_ids,
            "canonical_global_indices": self.canonical_global_indices,
        }
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()

    def project_global109(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[-1] != 109:
            raise ValueError(f"expected global action width 109, got {values.shape[-1]}")
        return values[..., list(self.canonical_global_indices)]

    def expand_to_global109(self, values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[-1] != 36:
            raise ValueError(f"expected canonical action width 36, got {values.shape[-1]}")
        out = np.full((*values.shape[:-1], 109), fill_value, dtype=values.dtype)
        out[..., list(self.canonical_global_indices)] = values
        return out

    def old_mask_to_canonical(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[-1] != 36:
            raise ValueError(f"expected old-mask action width 36, got {values.shape[-1]}")
        return values[..., list(self.canonical_to_old_indices)]

    def canonical_to_old_mask(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[-1] != 36:
            raise ValueError(f"expected canonical action width 36, got {values.shape[-1]}")
        return values[..., list(self.old_to_canonical_indices)]

    def align_action_dict(self, values: dict[str, float], default: float = 0.0) -> np.ndarray:
        """Online controller helper: materialize values in canonical order."""
        return np.asarray([float(values.get(aid, default)) for aid in self.canonical_ids], dtype=np.float32)
