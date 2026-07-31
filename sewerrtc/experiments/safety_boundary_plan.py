from __future__ import annotations

import numpy as np


def select_events_by_reference_load(
    *,
    event_ids: np.ndarray,
    splits: np.ndarray,
    reference_risk_rate_seq: np.ndarray,
    train_count: int,
    validation_count: int,
    dt_sec: int,
) -> dict[str, list[str]]:
    """Select fixed event groups using No-control load only, never candidate labels."""
    events = np.asarray(event_ids).astype(str)
    split_values = np.asarray(splits).astype(str)
    reference = np.asarray(reference_risk_rate_seq, dtype=np.float64)
    if reference.ndim != 3 or reference.shape[0] != len(events):
        raise ValueError("reference risk must align with event rows and have [N,H,C]")
    if reference.shape[2] < 1:
        raise ValueError("reference risk must contain a PFV-rate channel")
    if len(events) != len(split_values):
        raise ValueError("event_ids and splits do not align")
    load_by_event: dict[str, float] = {}
    split_by_event: dict[str, str] = {}
    horizon_pfv = reference[:, :, 0].sum(axis=1) * float(dt_sec)
    for event_id, split, load in zip(events, split_values, horizon_pfv):
        prior = split_by_event.setdefault(event_id, split)
        if prior != split:
            raise ValueError(f"event-group split leakage: {event_id}")
        load_by_event[event_id] = max(load_by_event.get(event_id, float("-inf")), float(load))
    selected: dict[str, list[str]] = {}
    for split, count in (("train", int(train_count)), ("validation", int(validation_count))):
        available = [event_id for event_id, event_split in split_by_event.items() if event_split == split]
        if len(available) < count:
            raise ValueError(f"not enough {split} events: requested {count}, available {len(available)}")
        selected[split] = sorted(available, key=lambda event_id: (-load_by_event[event_id], event_id))[:count]
    return selected


def build_boundary_case_slots(selected_events: dict[str, list[str]]) -> list[dict[str, object]]:
    """Allocate 72 stress cases while retaining the frozen event split."""
    slots: list[dict[str, object]] = []
    layouts = {
        "validation": (("peak", (0, 1, 2)), ("recession", (3, 4))),
        "train": (("peak", (2, 3)), ("recession", (3, 4))),
    }
    for split in ("train", "validation"):
        for event_id in selected_events.get(split, []):
            for phase, specification_indices in layouts[split]:
                for specification_index in specification_indices:
                    slots.append({
                        "split": split,
                        "event_id": str(event_id),
                        "phase": phase,
                        "specification_index": int(specification_index),
                    })
    return slots


def build_boundary_round2_slots(
    selected_events: dict[str, list[str]],
    *,
    recession_offsets_min: tuple[float, ...] = (15.0, 60.0),
) -> list[dict[str, object]]:
    """Add cross-checkpoint contrasts for the empirically informative outlet group."""
    offsets = tuple(float(value) for value in recession_offsets_min)
    if not offsets or any(value <= 0.0 for value in offsets):
        raise ValueError("recession offsets must be positive")
    slots: list[dict[str, object]] = []
    for split in ("train", "validation"):
        for event_id in selected_events.get(split, []):
            for offset in offsets:
                slots.append({
                    "split": split,
                    "event_id": str(event_id),
                    "phase": "recession",
                    "recession_offset_min": offset,
                })
    return slots
