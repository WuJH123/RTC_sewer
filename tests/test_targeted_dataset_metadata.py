from __future__ import annotations

import numpy as np
import pandas as pd

from sewerrtc.data.targeted_dataset_metadata import resolve_old_metadata


def test_resolve_old_metadata_prefers_metadata_embedded_in_npz() -> None:
    old = {
        "event_ids": np.asarray(["event_a", "event_b"]),
        "pair_ids": np.asarray(["pair_not_in_manifest", "pair_b"]),
        "split": np.asarray(["train", "validation"]),
        "candidate_kind": np.asarray(["joint", "pump"]),
        "candidate_family": np.asarray(["legacy_plus_residual", "binary_pump"]),
        "phase": np.asarray(["rising", "peak"]),
        "checkpoint_id": np.asarray(["event_a|rising|1", "event_b|peak|2"]),
        "source_dataset": np.asarray(["v3", "v3"]),
    }
    fallback = pd.DataFrame([{"pair_id": "pair_b", "candidate_kind": "wrong"}])

    metadata = resolve_old_metadata(old, fallback)

    assert metadata["candidate_kind"].tolist() == ["joint", "pump"]
    assert metadata["checkpoint_id"].tolist() == ["event_a|rising|1", "event_b|peak|2"]
    assert metadata["source_dataset"].tolist() == ["v3", "v3"]


def test_resolve_old_metadata_uses_defaults_for_pair_missing_from_legacy_manifest() -> None:
    old = {
        "event_ids": np.asarray(["event_a"]),
        "pair_ids": np.asarray(["missing_pair"]),
    }

    metadata = resolve_old_metadata(old, pd.DataFrame(columns=["pair_id"]))

    assert metadata["candidate_kind"].tolist() == ["legacy_targeted"]
    assert metadata["phase"].tolist() == ["unknown"]
    assert metadata["checkpoint_id"].tolist() == ["event_a|unknown|0.000"]
