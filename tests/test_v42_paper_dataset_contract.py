from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_paper_dataset import (
    _branch_arrays,
    _stable_kpi_deltas,
)


def _detail():
    times = [10.0, 20.0]
    return pd.DataFrame(
        {
            "elapsed_min": times,
            "rainfall_mm_h": [5.0, 4.0],
            "h:N1": [1.0, 1.1],
            "h:N2": [2.0, 2.1],
            "flood:N1": [1.0, 2.0],
            "flood:N2": [0.0, 1.0],
            "storage_volume:N2": [10.0, 11.0],
            # requested/target differs intentionally from readback
            "a:A1": [1.0, 1.0],
            "setting:A1": [0.2, 0.3],
            "flow:A1": [3.0, 3.1],
            "outfall_flow:N1": [4.0, 4.1],
        }
    )


def test_branch_arrays_use_actual_readback_not_requested_action():
    arrays = _branch_arrays(
        _detail(),
        future_times=np.asarray([10.0, 20.0]),
        node_ids=["N1", "N2"],
        storage_ids=["N2"],
        facility_ids=["A1"],
        outfall_ids=["N1"],
    )
    assert np.allclose(arrays["action_readback"].ravel(), [0.2, 0.3])
    assert not np.allclose(arrays["action_readback"].ravel(), [1.0, 1.0])


def test_branch_arrays_fail_closed_without_explicit_outfall_flow():
    df = _detail().drop(columns=["outfall_flow:N1"])
    with pytest.raises(KeyError, match="outfall_flow"):
        _branch_arrays(
            df,
            future_times=np.asarray([10.0, 20.0]),
            node_ids=["N1", "N2"],
            storage_ids=["N2"],
            facility_ids=["A1"],
            outfall_ids=["N1"],
        )


def test_volume_deltas_integrate_rate_difference_before_subtraction():
    cand = np.asarray([[1000.001, 1000.0], [1000.002, 1000.0]], dtype=np.float32)
    nc = np.asarray([[1000.0, 1000.0], [1000.0, 1000.0]], dtype=np.float32)
    di = np.asarray([[999.999, 1000.0], [999.999, 1000.0]], dtype=np.float32)
    branches = {
        "candidate": {"flood": cand},
        "no_control": {"flood": nc},
        "dynamic_internal": {"flood": di},
    }
    pfv, tfv, peak = _stable_kpi_deltas(branches, priority_indices=[0])
    expected_pfv = float((cand[:, 0].astype(np.float64) - nc[:, 0].astype(np.float64)).sum() * 600.0)
    expected_tfv = float((cand.astype(np.float64).sum(axis=1) - di.astype(np.float64).sum(axis=1)).sum() * 600.0)
    expected_peak = float(cand.astype(np.float64).sum(axis=1).max() - di.astype(np.float64).sum(axis=1).max())
    assert pfv == pytest.approx(expected_pfv)
    assert tfv == pytest.approx(expected_tfv)
    assert peak == pytest.approx(expected_peak)
