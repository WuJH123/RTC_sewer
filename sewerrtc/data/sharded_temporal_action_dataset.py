from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Iterator

import numpy as np


class _PrefetchFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


def prefetch_batches(
    source: Iterator[dict[str, np.ndarray]],
    *,
    prefetch_depth: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Overlap NPZ decompression/indexing with GPU work without reordering batches."""
    depth = max(0, int(prefetch_depth))
    if depth == 0:
        yield from source
        return
    queue: Queue[object] = Queue(maxsize=depth)
    stop = Event()
    sentinel = object()

    def put(value: object) -> bool:
        while not stop.is_set():
            try:
                queue.put(value, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce() -> None:
        try:
            for batch in source:
                if not put(batch):
                    return
        except BaseException as exc:  # propagate producer failures in consumer thread
            put(_PrefetchFailure(exc))
        finally:
            put(sentinel)

    worker = Thread(target=produce, name="temporal-action-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = queue.get(timeout=0.2)
            except Empty:
                if not worker.is_alive():
                    break
                continue
            if item is sentinel:
                break
            if isinstance(item, _PrefetchFailure):
                raise item.error
            yield item  # type: ignore[misc]
    finally:
        stop.set()
        worker.join(timeout=1.0)


TENSOR_KEYS = (
    "state",
    "candidate_action_seq",
    "rain_seq",
    "risk_rate_seq",
    "local_state_seq",
)


@dataclass(frozen=True)
class ShardedTemporalActionIndex:
    path: Path
    shard_files: tuple[Path, ...]
    sample_count: int
    node_cols: tuple[str, ...]
    local_node_cols: tuple[str, ...]
    action_ids: tuple[str, ...]


def load_sharded_index(path: str | Path) -> ShardedTemporalActionIndex:
    index_path = Path(path).resolve()
    with np.load(index_path, allow_pickle=True) as data:
        required = {"shard_files", "sample_count", "node_cols", "local_node_cols", "action_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Not a sharded temporal-action index; missing {sorted(missing)}")
        shards = tuple(Path(str(value)).resolve() for value in data["shard_files"].tolist())
        sample_count = int(data["sample_count"][0])
        node_cols = tuple(map(str, data["node_cols"].tolist()))
        local_node_cols = tuple(map(str, data["local_node_cols"].tolist()))
        action_ids = tuple(map(str, data["action_ids"].tolist()))
    for shard in shards:
        if not shard.exists():
            raise FileNotFoundError(f"Missing temporal-action shard: {shard}")
    if not shards or sample_count <= 0:
        raise ValueError("Sharded temporal-action index is empty")
    return ShardedTemporalActionIndex(
        path=index_path,
        shard_files=shards,
        sample_count=sample_count,
        node_cols=node_cols,
        local_node_cols=local_node_cols,
        action_ids=action_ids,
    )


def event_group_split(
    shard_files: tuple[Path, ...] | list[Path],
    *,
    validation_fraction: float = 0.2,
    seed: int = 20260714,
) -> tuple[set[str], set[str]]:
    events: set[str] = set()
    for shard in shard_files:
        with np.load(shard, allow_pickle=True) as data:
            events.update(map(str, data["event_ids"].tolist()))
    ordered = np.asarray(sorted(events), dtype=object)
    if len(ordered) < 2:
        raise ValueError("At least two independent events are required for event-group validation")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ordered)
    validation_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * float(validation_fraction)))))
    validation = set(map(str, ordered[:validation_count]))
    train = set(map(str, ordered[validation_count:]))
    return train, validation


def iter_sharded_batches(
    shard_files: tuple[Path, ...] | list[Path],
    *,
    allowed_events: set[str],
    batch_size: int,
    max_samples: int = 0,
    seed: int = 20260714,
) -> Iterator[dict[str, np.ndarray]]:
    if not allowed_events:
        raise ValueError("allowed_events cannot be empty")
    batch_size = max(1, int(batch_size))
    limit = max(0, int(max_samples or 0))
    yielded = 0
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(shard_files))
    rng.shuffle(order)
    for shard_index in order:
        with np.load(shard_files[int(shard_index)], allow_pickle=True) as data:
            missing = set(TENSOR_KEYS + ("event_ids",)).difference(data.files)
            if missing:
                raise ValueError(f"Shard {shard_files[int(shard_index)]} missing {sorted(missing)}")
            event_ids = data["event_ids"].astype(str)
            rows = np.flatnonzero(np.isin(event_ids, sorted(allowed_events)))
            rng.shuffle(rows)
            if limit:
                rows = rows[: max(0, limit - yielded)]
            if not len(rows):
                continue
            arrays = {key: data[key] for key in TENSOR_KEYS}
            for start in range(0, len(rows), batch_size):
                selected = rows[start : start + batch_size]
                batch = {key: arrays[key][selected] for key in TENSOR_KEYS}
                batch["event_ids"] = event_ids[selected]
                yield batch
                yielded += len(selected)
                if limit and yielded >= limit:
                    return
