import importlib.util
from pathlib import Path

import pandas as pd


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "93_train_raw_joint_action_surrogate_v3.py"
    spec = importlib.util.spec_from_file_location("train_raw_joint_action_surrogate_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_v3 = _load_train_module()


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_accepts_residual10_mixed_reference_semantics():
    assert train_v3.label_semantics_supported("mixed_reference_residual10_core_conditioned")


def test_training_keeps_legacy_same_state_semantics():
    assert train_v3.label_semantics_supported("same_state_candidate_minus_no_control")


def test_training_rejects_unknown_label_semantics():
    assert not train_v3.label_semantics_supported("trajectory_policy_correlation")


def test_training_auto_architecture_uses_warm_checkpoint_version():
    assert train_v3.resolve_architecture_version(
        "auto",
        {"architecture_version": "causal_phase_direction_v6"},
    ) == "causal_phase_direction_v6"


def test_training_auto_architecture_falls_back_for_legacy_checkpoint():
    assert train_v3.resolve_architecture_version("auto", {}) == "priority_aware_safety_v3"


def test_training_explicit_architecture_overrides_checkpoint_version():
    assert train_v3.resolve_architecture_version(
        "priority_aware_safety_v4",
        {"architecture_version": "causal_phase_direction_v6"},
    ) == "priority_aware_safety_v4"


def test_residual10_dataset_uses_two_percent_pfv_noninferiority_margin():
    builder = _load_script("121_build_residual10_core_effect_dataset.py")

    margin = builder._noninferiority_margin(
        pd.Series([1000.0, 10000.0]),
        absolute_margin=100.0,
        relative_margin=0.02,
    )

    assert margin.tolist() == [100.0, 200.0]


def test_residual10_guard_uses_reference_relative_pfv_margin():
    guard = _load_script("122_build_fit_only_residual10_empirical_guard.py")

    noninferior = guard._pfv_noninferior(
        pd.Series([150.0, 250.0]),
        pd.Series([10000.0, 10000.0]),
        absolute_margin=100.0,
        relative_margin=0.02,
    )

    assert noninferior.tolist() == [True, False]
