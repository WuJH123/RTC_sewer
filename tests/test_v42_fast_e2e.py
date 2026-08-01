from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_fast_e2e import (
    FAST_E2E_CONTRACT_ID,
    build_development_baseline_actions,
    build_fast_step1_aux_allowlist_64plus,
    control_horizon_sequence,
    make_causal_rainfall_forecast,
    nearest_recorded_action_proxy,
)
from sewerrtc.v4.v42_fast_e2e_warm import build_warm_fast_step2_dataset_64plus


def test_step1_fast_e2e_requires_at_least_64_rainfall_groups(tmp_path: Path) -> None:
    rows = [
        {
            "detail_path": f"d{i}.csv",
            "anchor_min": 120.0,
            "split_group_key": f"rain_{i:03d}",
            "physical_identity_sha256": f"p{i:03d}",
            "step1_domain_role": "auxiliary_pretrain",
        }
        for i in range(63)
    ]
    manifest = tmp_path / "step1.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    with pytest.raises(RuntimeError, match="at least 64"):
        build_fast_step1_aux_allowlist_64plus(
            manifest_path=manifest,
            output_path=tmp_path / "allow.json",
            target_groups=96,
            min_groups=64,
            seed=42,
        )


def test_step1_fast_e2e_accepts_more_than_64_groups(tmp_path: Path) -> None:
    rows = [
        {
            "detail_path": f"d{i}.csv",
            "anchor_min": 120.0,
            "split_group_key": f"rain_{i:03d}",
            "physical_identity_sha256": f"p{i:03d}",
            "step1_domain_role": "auxiliary_pretrain",
        }
        for i in range(80)
    ]
    manifest = tmp_path / "step1.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    payload = build_fast_step1_aux_allowlist_64plus(
        manifest_path=manifest,
        output_path=tmp_path / "allow.json",
        target_groups=72,
        min_groups=64,
        seed=42,
    )
    assert payload["selected_aux_groups"] == 72
    assert payload["diversity_gate_pass"] is True
    assert payload["fast_e2e_contract_id"] == FAST_E2E_CONTRACT_ID


def test_causal_rainfall_forecast_uses_only_last_observed_value() -> None:
    a = make_causal_rainfall_forecast([100.0, 20.0, 7.0])
    b = make_causal_rainfall_forecast([0.0, 999.0, 7.0])
    assert a.shape == (12,)
    assert np.allclose(a, b)
    assert np.all(a >= 0.0)
    assert np.allclose(a[:3], 7.0)
    assert np.allclose(a[6:], 0.0)


def test_control_horizon_is_h12_h3_with_frozen_tail() -> None:
    anchor = np.linspace(0.0, 1.0, 36, dtype=np.float32)
    desired = 1.0 - anchor
    seq = control_horizon_sequence(desired, anchor)
    assert seq.shape == (12, 36)
    assert np.allclose(seq[:3], desired[None, :])
    assert np.allclose(seq[3:], anchor[None, :])


def test_development_efd_rbc_share_k8_and_binary_contract() -> None:
    n_nodes = 40
    n_facilities = 36
    current = np.linspace(0.0, 2.0, n_nodes, dtype=np.float32)
    max_depth = np.full(n_nodes, 2.0, dtype=np.float32)
    action_map = np.zeros((n_facilities, n_nodes), dtype=np.float32)
    for i in range(n_facilities):
        action_map[i, i % n_nodes] = 1.0
    anchor = np.full(n_facilities, 0.5, dtype=np.float32)
    # ADD301.2 / ADD301.3 analogues must start from legal readback states.
    anchor[0] = 0.0
    anchor[1] = 1.0
    binary = (0, 1)
    schedules = build_development_baseline_actions(
        current_depth=current,
        max_depth=max_depth,
        action_node_map=action_map,
        anchor_action=anchor,
        binary_indices=binary,
        max_changed_facilities=8,
    )
    assert set(schedules) == {"efd", "auto_rbc", "all_close"}
    for name in ("efd", "auto_rbc"):
        seq = schedules[name]
        assert seq.shape == (12, 36)
        changed = int(np.sum(np.abs(seq[0] - anchor) > 1e-9))
        assert changed <= 8
        assert seq[0, 0] in (0.0, 1.0)
        assert seq[0, 1] in (0.0, 1.0)
        assert np.allclose(seq[3:], anchor[None, :])
    assert np.allclose(schedules["all_close"], 0.0)


def test_nearest_recorded_action_proxy_scores_only_h3() -> None:
    target = np.zeros((12, 36), dtype=np.float32)
    a = np.ones((12, 36), dtype=np.float32)
    b = np.zeros((12, 36), dtype=np.float32)
    b[3:] = 1.0  # different tail must not matter to the H3 proxy metric
    idx, distance = nearest_recorded_action_proxy(target, [a, b])
    assert idx == 1
    assert distance == pytest.approx(0.0)


def test_warm_wrapper_has_explicit_120min_contract() -> None:
    import inspect

    signature = inspect.signature(build_warm_fast_step2_dataset_64plus)
    assert signature.parameters["min_checkpoint_min"].default == 120.0
