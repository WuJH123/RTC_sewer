from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


METADATA_KEYS = (
    "split",
    "candidate_kind",
    "candidate_family",
    "phase",
    "checkpoint_id",
    "source_dataset",
)


def _contains(container: object, key: str) -> bool:
    files = getattr(container, "files", None)
    return key in files if files is not None else key in container


def resolve_old_metadata(old: Mapping[str, np.ndarray] | object, fallback_manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Resolve row metadata without requiring every old pair in a new manifest."""
    event_ids = np.asarray(old["event_ids"]).astype(str)
    pair_ids = np.asarray(old["pair_ids"]).astype(str)
    row_count = len(event_ids)
    if len(pair_ids) != row_count:
        raise ValueError("old event_ids and pair_ids must have equal length")
    if all(_contains(old, key) for key in METADATA_KEYS):
        embedded = {key: np.asarray(old[key]).astype(str) for key in METADATA_KEYS}
        if any(len(values) != row_count for values in embedded.values()):
            raise ValueError("embedded old metadata length differs from old dataset rows")
        return embedded

    candidates = fallback_manifest.copy()
    if "branch" in candidates:
        candidates = candidates[candidates["branch"].astype(str).eq("B")]
    if "pair_id" in candidates:
        candidates = candidates.drop_duplicates("pair_id").set_index(candidates["pair_id"].astype(str))
    else:
        candidates = pd.DataFrame(index=pd.Index([], dtype=str))

    validation_events = set(sorted(set(event_ids))[::4])
    split, kind, family, phase, checkpoint = [], [], [], [], []
    for event_id, pair_id in zip(event_ids, pair_ids):
        row = candidates.loc[pair_id] if pair_id in candidates.index else pd.Series(dtype=object)
        row_phase = str(row.get("phase", "unknown"))
        fraction = float(row.get("split_timestamp_fraction", 0.0) or 0.0)
        split.append("validation" if event_id in validation_events else "train")
        kind.append(str(row.get("candidate_kind", "legacy_targeted")))
        family.append(str(row.get("candidate_family", "legacy_targeted")))
        phase.append(row_phase)
        checkpoint.append(f"{event_id}|{row_phase}|{fraction:.3f}")
    return {
        "split": np.asarray(split),
        "candidate_kind": np.asarray(kind),
        "candidate_family": np.asarray(family),
        "phase": np.asarray(phase),
        "checkpoint_id": np.asarray(checkpoint),
        "source_dataset": np.repeat("legacy_336_filtered", row_count),
    }
