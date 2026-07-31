from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ROW_KEYS = (
    "event_ids", "pair_ids", "state", "candidate_action_seq", "reference_action_seq", "rain_seq",
    "reference_risk_rate_seq", "delta_risk_rate_seq", "priority_depth_seq", "storage_level_seq",
    "target_state_seq", "split", "candidate_kind", "candidate_family", "phase", "checkpoint_id",
    "source_dataset",
)


def merge_effect_payloads(
    base: Mapping[str, np.ndarray],
    supplement: Mapping[str, np.ndarray],
    *,
    base_split_policy: str = "preserve",
    locked_validation_events: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if base_split_policy not in {"preserve", "all_train"}:
        raise ValueError("base_split_policy must be preserve or all_train")
    for key in ("node_ids", "action_ids"):
        if not np.array_equal(np.asarray(base[key]).astype(str), np.asarray(supplement[key]).astype(str)):
            raise ValueError(f"{key} differs between base and supplement")
    if tuple(np.asarray(base["candidate_action_seq"]).shape[1:]) != (6, 36):
        raise ValueError("base dataset is not strict [N,6,36]")
    if tuple(np.asarray(supplement["candidate_action_seq"]).shape[1:]) != (6, 36):
        raise ValueError("supplement dataset is not strict [N,6,36]")

    locked = {str(value) for value in (locked_validation_events or set())}
    base_events = np.asarray(base["event_ids"]).astype(str)
    supplement_events = np.asarray(supplement["event_ids"]).astype(str)
    if locked & set(base_events):
        raise ValueError("locked validation events overlap the frozen base dataset")
    if locked and not locked.issubset(set(supplement_events)):
        missing = sorted(locked - set(supplement_events))
        raise ValueError(f"locked validation events missing from supplement: {missing}")

    existing_pairs = set(np.asarray(base["pair_ids"]).astype(str))
    keep = np.asarray([pair_id not in existing_pairs for pair_id in np.asarray(supplement["pair_ids"]).astype(str)])
    payload: dict[str, np.ndarray] = {}
    for key in ROW_KEYS:
        if key not in base or key not in supplement:
            raise KeyError(f"missing row-level dataset key: {key}")
        base_values = np.asarray(base[key]).copy()
        supplement_values = np.asarray(supplement[key])[keep].copy()
        if key == "split":
            if base_split_policy == "all_train":
                base_values = np.full(base_values.shape, "train", dtype="<U10")
            if locked:
                kept_events = supplement_events[keep]
                supplement_values = np.asarray(
                    ["validation" if event_id in locked else "train" for event_id in kept_events]
                )
        payload[key] = np.concatenate([base_values, supplement_values], axis=0)
    for key in base:
        if key not in payload:
            payload[key] = np.asarray(base[key])

    event_split: dict[str, str] = {}
    for event_id, split in zip(payload["event_ids"].astype(str), payload["split"].astype(str)):
        prior = event_split.setdefault(event_id, split)
        if prior != split:
            raise ValueError(f"event-group leakage after merge: {event_id}")
    actual_validation = {event for event, split in event_split.items() if split == "validation"}
    if locked and actual_validation != locked:
        raise ValueError(
            f"final validation events differ from lock: actual={sorted(actual_validation)}, locked={sorted(locked)}"
        )
    report = {
        "base_rows": int(len(base_events)),
        "supplement_rows": int(len(supplement_events)),
        "supplement_rows_added": int(keep.sum()),
        "duplicates_skipped": int((~keep).sum()),
        "combined_rows": int(len(payload["event_ids"])),
        "combined_events": int(len(event_split)),
        "action_shape": list(payload["candidate_action_seq"].shape),
        "base_split_policy": base_split_policy,
        "locked_validation_events": sorted(locked),
        "event_group_leakage": False,
    }
    return payload, report
