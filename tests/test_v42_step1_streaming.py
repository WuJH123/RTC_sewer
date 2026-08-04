from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_step1_streaming import (
    _projected_cache_location,
    _read_projected_detail,
    projected_cache_stats,
    select_manifest_rows,
    split_target_groups,
    summarise_selection,
)
from sewerrtc.v4.v42_step1_dataset import _detail_extract_window
from scripts.train_v42_step1_streaming import _save_full_checkpoint


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


def test_projected_cache_roundtrip_preserves_canonical_values(tmp_path):
    required = ["elapsed_min", "rainfall_mm_h", "h:N1", "setting:F1"]
    expected = pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0],
            "rainfall_mm_h": [1.25, 2.5],
            "h:N1": [0.125, 0.25],
            "setting:F1": [0.0, 1.0],
        }
    )
    source = tmp_path / "detail.csv"
    expected.to_csv(source, index=False)
    cache = tmp_path / "cache"
    projected_cache_stats(reset=True)
    first = _read_projected_detail(
        source, required, cache_dir=cache, source_identity="physical-sha"
    )
    second = _read_projected_detail(
        source, required, cache_dir=cache, source_identity="physical-sha"
    )
    pd.testing.assert_frame_equal(first, expected)
    pd.testing.assert_frame_equal(second, expected)
    stats = projected_cache_stats()
    assert stats["misses"] == 1
    assert stats["writes"] == 1
    assert stats["hits"] == 1
    cache_path, _ = _projected_cache_location(
        source, required, cache_dir=cache, source_identity="physical-sha"
    )
    assert cache_path.exists()


def test_prepared_window_extraction_matches_reference_path():
    times = np.arange(0.0, 65.0, 5.0)
    detail = pd.DataFrame(
        {
            "elapsed_min": times,
            "rainfall_mm_h": times + 1.0,
            "h:N1": times + 10.0,
            "setting:F1": times / 10.0,
        }
    )
    reference = _detail_extract_window(detail, 60.0, ["N1"], ["F1"])
    elapsed = detail["elapsed_min"].to_numpy(np.float64)
    elapsed_index = {round(float(value), 6): index for index, value in enumerate(elapsed)}
    prepared = _detail_extract_window(
        None,
        60.0,
        ["N1"],
        ["F1"],
        elapsed_values=elapsed,
        elapsed_index=elapsed_index,
        depth_values=detail[["h:N1"]].to_numpy(np.float32),
        rain_values=detail["rainfall_mm_h"].to_numpy(np.float32),
        action_values=detail[["setting:F1"]].to_numpy(np.float32),
    )
    assert reference is not None and prepared is not None
    for key in ("depth_history", "rainfall", "actions"):
        np.testing.assert_array_equal(reference[key], prepared[key])


def test_full_resume_checkpoint_contains_model_optimizer_rng_and_identity(tmp_path):
    import torch

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last_resume.pt"
    _save_full_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        meta={"epoch": 1, "next_batch_index": 7, "manifest_sha256": "m"},
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert set(("model_state_dict", "optimizer_state_dict", "rng_state", "meta")) <= set(checkpoint)
    assert checkpoint["meta"]["next_batch_index"] == 7


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
