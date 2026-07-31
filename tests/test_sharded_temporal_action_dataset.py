from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_shard(path: Path, event_ids: list[str], offset: float) -> None:
    n = len(event_ids)
    np.savez_compressed(
        path,
        state=np.full((n, 3), offset, dtype=np.float32),
        candidate_action_seq=np.full((n, 2, 4), offset, dtype=np.float32),
        rain_seq=np.zeros((n, 2, 1), dtype=np.float32),
        risk_rate_seq=np.full((n, 2, 3), offset, dtype=np.float32),
        local_state_seq=np.full((n, 2, 2), offset, dtype=np.float32),
        event_ids=np.asarray(event_ids, dtype=object),
        policy_ids=np.asarray(["p"] * n, dtype=object),
        source_files=np.asarray(["f.csv"] * n, dtype=object),
        row_indices=np.arange(n, dtype=np.int64),
        label_roles=np.asarray(["observational_dynamics_pretraining"] * n, dtype=object),
    )


def test_sharded_dataset_event_split_and_batches_are_grouped(tmp_path):
    from sewerrtc.data.sharded_temporal_action_dataset import (
        event_group_split,
        iter_sharded_batches,
        load_sharded_index,
    )

    shard0 = tmp_path / "chunk_00000.npz"
    shard1 = tmp_path / "chunk_00001.npz"
    _write_shard(shard0, ["E1", "E1", "E2"], 1.0)
    _write_shard(shard1, ["E2", "E3", "E4"], 2.0)
    index_path = tmp_path / "index.npz"
    np.savez_compressed(
        index_path,
        shard_files=np.asarray([str(shard0), str(shard1)], dtype=object),
        sample_count=np.asarray([6], dtype=np.int64),
        node_cols=np.asarray(["h:N1", "h:N2", "h:N3"], dtype=object),
        local_node_cols=np.asarray(["h:N1", "h:N2"], dtype=object),
        action_ids=np.asarray(["A1", "A2", "A3", "A4"], dtype=object),
    )

    index = load_sharded_index(index_path)
    train_events, validation_events = event_group_split(index.shard_files, validation_fraction=0.25, seed=7)
    assert train_events.isdisjoint(validation_events)
    assert train_events | validation_events == {"E1", "E2", "E3", "E4"}

    batches = list(
        iter_sharded_batches(
            index.shard_files,
            allowed_events=train_events,
            batch_size=2,
            max_samples=3,
            seed=9,
        )
    )
    assert sum(len(batch["event_ids"]) for batch in batches) == 3
    assert all(set(batch["event_ids"].astype(str)).issubset(train_events) for batch in batches)
    assert all(batch["candidate_action_seq"].shape[1:] == (2, 4) for batch in batches)


def test_sharded_dataset_rejects_missing_shards(tmp_path):
    from sewerrtc.data.sharded_temporal_action_dataset import load_sharded_index

    index_path = tmp_path / "index.npz"
    np.savez_compressed(
        index_path,
        shard_files=np.asarray([str(tmp_path / "missing.npz")], dtype=object),
        sample_count=np.asarray([1], dtype=np.int64),
        node_cols=np.asarray(["h:N1"], dtype=object),
        local_node_cols=np.asarray(["h:N1"], dtype=object),
        action_ids=np.asarray(["A1"], dtype=object),
    )

    try:
        load_sharded_index(index_path)
    except FileNotFoundError as exc:
        assert "missing.npz" in str(exc)
    else:
        raise AssertionError("missing shard was not rejected")


def test_prefetch_batches_preserves_batch_order_and_values():
    from sewerrtc.data.sharded_temporal_action_dataset import prefetch_batches

    source = (
        {"event_ids": np.asarray([f"E{i}"]), "state": np.asarray([[float(i)]])}
        for i in range(5)
    )

    batches = list(prefetch_batches(source, prefetch_depth=3))

    assert [str(batch["event_ids"][0]) for batch in batches] == [f"E{i}" for i in range(5)]
    assert [float(batch["state"][0, 0]) for batch in batches] == list(map(float, range(5)))
