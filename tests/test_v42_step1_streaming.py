from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_step1_streaming import (
    _read_projected_detail,
    select_manifest_rows,
    split_target_groups,
    summarise_selection,
)


def test_step1_reordered_csv_columns_preserve_semantics(tmp_path):
    required = [
        "elapsed_min",
        "rainfall_mm_h",
        "h:N1",
        "h:N2",
        "setting:F1",
    ]
    canonical = pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0],
            "rainfall_mm_h": [11.0, 12.0],
            "h:N1": [101.0, 102.0],
            "h:N2": [201.0, 202.0],
            "setting:F1": [0.25, 0.50],
        }
    )
    path = tmp_path / "reordered.csv"
    canonical[["h:N2", "setting:F1", "rainfall_mm_h", "elapsed_min", "h:N1"]].to_csv(
        path, index=False
    )

    got = _read_projected_detail(path, required)
    assert got.columns.tolist() == required
    pd.testing.assert_frame_equal(got.reset_index(drop=True), canonical[required])


def test_step1_streaming_projection_missing_column_fails_closed(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame(
        {
            "elapsed_min": [0.0],
            "rainfall_mm_h": [1.0],
            "h:N1": [0.2],
        }
    ).to_csv(path, index=False)
    with pytest.raises(KeyError, match="missing required columns"):
        _read_projected_detail(
            path,
            ["elapsed_min", "rainfall_mm_h", "h:N1", "setting:F1"],
        )


def _manifest() -> pd.DataFrame:
    rows = []
    for group in ("g1", "g2"):
        for i in range(10):
            rows.append(
                {
                    "physical_identity_sha256": f"{group}-pid-{i // 5}",
                    "detail_path": f"/{group}/detail-{i // 5}.csv",
                    "anchor_min": float(i * 5),
                    "split_group_key": group,
                    "step1_domain_role": "auxiliary_pretrain",
                }
            )
    return pd.DataFrame(rows)


def test_aux_sampling_is_group_balanced_and_deterministic():
    frame = _manifest()
    a = select_manifest_rows(
        frame,
        domain_roles=("auxiliary_pretrain",),
        max_windows_per_group=4,
        sampling_seed=17,
    )
    b = select_manifest_rows(
        frame,
        domain_roles=("auxiliary_pretrain",),
        max_windows_per_group=4,
        sampling_seed=17,
    )
    assert len(a) == 8
    assert a.groupby("split_group_key").size().to_dict() == {"g1": 4, "g2": 4}
    pd.testing.assert_frame_equal(a, b)
    assert summarise_selection(a).selection_sha256 == summarise_selection(b).selection_sha256


def test_target_split_supports_leave_one_event_out_without_calibration():
    groups = ("g1", "g2", "g3", "g4")
    split = split_target_groups(
        groups,
        split_seed=42,
        validation_group="g3",
        reserve_calibration=False,
    )
    assert split["validation"] == ("g3",)
    assert split["calibration"] == tuple()
    assert set(split["train"]) == {"g1", "g2", "g4"}
    assert not set(split["train"]).intersection(split["validation"])


def test_target_split_reserves_distinct_calibration_group():
    groups = ("g1", "g2", "g3", "g4")
    split = split_target_groups(groups, split_seed=9, reserve_calibration=True)
    train = set(split["train"])
    val = set(split["validation"])
    cal = set(split["calibration"])
    assert train
    assert len(val) == 1
    assert len(cal) == 1
    assert not train & val
    assert not train & cal
    assert not val & cal
    assert train | val | cal == set(groups)
