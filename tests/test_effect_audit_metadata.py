from __future__ import annotations

import json

import numpy as np
import pandas as pd


def test_audit_metadata_falls_back_to_embedded_dataset_provenance():
    from sewerrtc.experiments.effect_audit import resolve_audit_metadata

    data = {
        "phase": np.asarray(["peak"]),
        "checkpoint_id": np.asarray(["E1|peak|0.500"]),
        "candidate_kind": np.asarray(["legacy_group"]),
        "candidate_family": np.asarray(["legacy_v8"]),
        "source_dataset": np.asarray(["base_effect_dataset.npz"]),
    }
    resolved = resolve_audit_metadata(
        pair_id="old-pair",
        event_id="E1",
        row_index=0,
        candidates=pd.DataFrame().set_index(pd.Index([], name="pair_id")),
        data=data,
    )

    assert resolved["manifest_match"] is False
    assert resolved["phase"] == "peak"
    assert resolved["checkpoint_id"] == "E1|peak|0.500"
    assert resolved["candidate_kind"] == "legacy_group"
    assert resolved["candidate_mode"] == "legacy_v8"
    assert resolved["source_dataset"] == "base_effect_dataset.npz"


def test_audit_metadata_uses_manifest_details_when_pair_is_available():
    from sewerrtc.experiments.effect_audit import resolve_audit_metadata

    specification = {
        "kind": "joint_continuous",
        "mode": "decrease_peak_0.10",
        "actuators": ["A", "B"],
        "signed_profiles": {"A": [-0.1] * 6, "B": [-0.1] * 6},
    }
    candidates = pd.DataFrame([
        {
            "pair_id": "new-pair",
            "phase": "peak",
            "split_timestamp_fraction": 0.5,
            "executed_action_sequence": json.dumps(specification),
        }
    ]).set_index("pair_id")
    data = {
        "phase": np.asarray(["peak"]),
        "checkpoint_id": np.asarray(["E2|peak|0.500"]),
        "candidate_kind": np.asarray(["joint_continuous"]),
        "candidate_family": np.asarray(["causal_coverage_joint_group"]),
        "source_dataset": np.asarray(["supplement.npz"]),
    }
    resolved = resolve_audit_metadata(
        pair_id="new-pair",
        event_id="E2",
        row_index=0,
        candidates=candidates,
        data=data,
    )

    assert resolved["manifest_match"] is True
    assert resolved["specification"] == specification
    assert resolved["candidate_mode"] == "decrease_peak_0.10"
