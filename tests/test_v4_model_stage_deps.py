"""New V4 model stage dependency wiring tests (spec section 4/9)."""
from __future__ import annotations

from sewerrtc.v4.pipeline import ALL_STAGES, PREREQUISITES, STAGE_ARTIFACTS

CHAIN = [
    "TrainV4Baselines",
    "EvaluateV4Baselines",
    "TrainV4TrueState",
    "CalibrateV4TrueState",
    "EvaluateV4TrueStateLocked",
    "AuditV4OfflineSafetyGate",
]


def test_chain_roots_at_authorization_v4():
    assert PREREQUISITES["TrainV4Baselines"] == (
        "AuditModelTrainingAuthorizationV4",
    )


def test_chain_is_linear():
    expected_prev = "TrainV4Baselines"
    for stage in CHAIN[1:]:
        assert PREREQUISITES[stage] == (expected_prev,)
        expected_prev = stage


def test_chain_does_not_use_legacy_dataset_path():
    # None of the new stages may depend on the legacy AuditTrain1600Dataset.
    for stage in CHAIN:
        assert "AuditTrain1600Dataset" not in PREREQUISITES.get(stage, ())
    # The legacy generic TrainV4 keeps its own (old) chain, untouched.
    assert PREREQUISITES["TrainV4"] == ("AuditTrain1600Dataset",)


def test_all_stages_ordering():
    idx = {s: i for i, s in enumerate(ALL_STAGES)}
    assert idx["AuditModelTrainingAuthorizationV4"] < idx["TrainV4Baselines"]
    for a, b in zip(CHAIN, CHAIN[1:]):
        assert idx[a] < idx[b]


def test_each_stage_has_artifact():
    for stage in CHAIN:
        assert stage in STAGE_ARTIFACTS
        assert STAGE_ARTIFACTS[stage].startswith("models/v4_true_state/")
