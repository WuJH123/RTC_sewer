import numpy as np
import pandas as pd
import pytest

from scripts.run_v42_targeted_candidate_expansion import _horizon_arrays


def _detail(*, with_targets: bool) -> pd.DataFrame:
    rows = {"elapsed_min": np.arange(10.0, 130.0, 10.0)}
    for node_id in ("J1", "S1"):
        rows[f"h:{node_id}"] = np.ones(12)
        rows[f"flood:{node_id}"] = np.zeros(12)
    rows["setting:P1"] = np.full(12, 0.5)
    rows["flow:P1"] = np.full(12, 1.0)
    if with_targets:
        rows["storage_volume:S1"] = np.full(12, 100.0)
    return pd.DataFrame(rows)


def test_targeted_horizon_exports_control_core_targets():
    arrays = _horizon_arrays(
        _detail(with_targets=True),
        checkpoint=0.0,
        node_ids=["J1", "S1"],
        storage_node_ids=["S1"],
        actuator_ids=["P1"],
    )
    assert arrays["storage"].shape == (12, 1)
    assert arrays["facility_flow"].shape == (12, 1)
    assert np.isfinite(arrays["storage"]).all()
    assert np.isfinite(arrays["facility_flow"]).all()


def test_targeted_horizon_fails_closed_without_storage_targets():
    with pytest.raises(RuntimeError, match="storage_volume"):
        _horizon_arrays(
            _detail(with_targets=False),
            checkpoint=0.0,
            node_ids=["J1", "S1"],
            storage_node_ids=["S1"],
            actuator_ids=["P1"],
        )


def test_targeted_horizon_starts_after_checkpoint_row():
    detail = _detail(with_targets=True)
    detail["storage_volume:S1"] = np.arange(12.0)
    detail = pd.concat([detail, detail.iloc[[-1]].assign(elapsed_min=130.0, **{"storage_volume:S1": 12.0})], ignore_index=True)
    arrays = _horizon_arrays(
        detail,
        checkpoint=10.0,
        node_ids=["J1", "S1"],
        storage_node_ids=["S1"],
        actuator_ids=["P1"],
    )
    assert arrays["storage"][0, 0] == 1.0
