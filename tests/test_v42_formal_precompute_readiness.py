import json

import numpy as np
import pandas as pd

from scripts.audit_v42_formal_precompute_readiness import (
    _hash_json_array,
    _stats,
    _target_columns,
    _weighted_group_stats,
)
from scripts.materialize_v42_formal_gat_history_f2 import _signature


def test_h3_action_hash_is_format_stable():
    a = [[0.0] * 36, [1.0] * 36, [0.5] * 36, [0.2] * 36]
    assert _hash_json_array(json.dumps(a)) == _hash_json_array(json.dumps(np.asarray(a).tolist()))
    assert _hash_json_array(json.dumps(a)) != _hash_json_array(json.dumps([[0.0] * 36, [1.0] * 36, [0.7] * 36]))


def test_group_weight_effective_sample_size_and_stats():
    frame = pd.DataFrame({"rainfall": ["a", "a", "a", "b", "c", "c"]})
    result = _weighted_group_stats(frame, "rainfall", "windows")
    assert result["groups"] == 3
    assert result["effective_group_count"] == 36 / 14
    assert _stats([1, 2, 3])["p50"] == 2.0


def test_target_columns_keep_outfall_as_explicit_required_family():
    result = _target_columns(["N1"], ["S1"], ["ADD301.2"], ["O1"])
    assert result["storage_volume"] == ["storage_volume:S1"]
    assert result["managed_facility_flow"] == ["flow:ADD301.2"]
    assert result["outfall_flow"] == ["outfall_flow:O1"]


def test_formal_gat_history_signature_excludes_checkpoint_transition(monkeypatch):
    class Graph:
        node_ids = ["N1"]
        facility_ids = ["A1"]

    extracted = {
        "depth_history": np.asarray([[0.1], [0.2]], dtype=float),
        "actions": np.asarray([[0.0], [1.0]], dtype=float),
        "rainfall": np.asarray([0.5], dtype=float),
    }
    monkeypatch.setattr(
        "scripts.materialize_v42_formal_gat_history_f2._detail_extract_window",
        lambda *_args: extracted,
    )
    depth, pre_action, rainfall = _signature(pd.DataFrame(), 120.0, Graph())
    assert depth.tolist() == [0.2]
    assert pre_action.shape == (1, 1)
    assert pre_action.tolist() == [[0.0]]
    assert rainfall.tolist() == [0.5]
