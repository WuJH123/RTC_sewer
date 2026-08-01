from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalGATOutput
from sewerrtc.v4.v42_step1_dataset import (
    Step1Sample,
    _detail_extract_window,
    _sensor_mask_for_window,
)
from sewerrtc.v4.v42_step1_training import (
    build_formal_step1_split,
    step1_reconstruction_loss,
)


def _sample(group: str, role: str) -> Step1Sample:
    return Step1Sample(
        sparse_depth_history=np.zeros((13, 2), dtype=np.float32),
        sensor_mask_history=np.zeros((13, 2), dtype=np.float32),
        rainfall_history=np.zeros(13, dtype=np.float32),
        historical_actions=np.zeros((13, 1), dtype=np.float32),
        target_depth=np.zeros(2, dtype=np.float32),
        split_group=group,
        window_key=f"{role}:{group}",
        step1_domain_role=role,
    )


def test_formal_sensor_layout_is_fixed_across_windows():
    a = _sensor_mask_for_window("window-a", 100, 0.10, 42)
    b = _sensor_mask_for_window("window-b", 100, 0.10, 42)
    assert np.array_equal(a, b)
    assert int(a.sum()) == 10


def test_detail_extract_rejects_missing_readback_instead_of_zero_fill():
    elapsed = np.arange(0.0, 65.0, 5.0)
    frame = pd.DataFrame(
        {
            "elapsed_min": elapsed,
            "rainfall_mm_h": np.ones(len(elapsed)),
            "h:N1": np.linspace(0.0, 1.0, len(elapsed)),
        }
    )
    with pytest.raises(KeyError, match="missing required columns"):
        _detail_extract_window(frame, 60.0, ["N1"], ["F1"])


def test_formal_split_keeps_auxiliary_out_of_validation_and_calibration():
    samples = [
        _sample("g1", "target_formal"),
        _sample("g2", "target_formal"),
        _sample("g3", "target_formal"),
        _sample("g4", "target_formal"),
        _sample("aux1", "auxiliary_pretrain"),
        _sample("aux2", "auxiliary_pretrain"),
    ]
    split = build_formal_step1_split(samples, seed=7)
    aux = set(split.auxiliary_pretrain_indices)
    assert aux == {4, 5}
    assert not aux.intersection(split.target_validation_indices)
    assert not aux.intersection(split.target_calibration_indices)
    target_groups = (
        set(split.target_train_groups)
        | set(split.target_validation_groups)
        | set(split.target_calibration_groups)
    )
    assert target_groups == {"g1", "g2", "g3", "g4"}


def test_uncertainty_head_receives_gradient_from_formal_loss():
    mean = torch.tensor([[0.2, 0.4]], requires_grad=True)
    raw_std = torch.tensor([[0.3, 0.5]], requires_grad=True)
    output = TemporalGATOutput(
        depth_mean=mean,
        depth_std=raw_std,
        latent_state=torch.zeros((1, 2, 1)),
    )
    target = torch.tensor([[0.4, 0.1]])
    sensor_mask = torch.zeros_like(target)
    priority = torch.tensor([True, False])
    losses = step1_reconstruction_loss(
        output,
        target,
        sensor_mask,
        priority,
    )
    losses["total"].backward()
    assert raw_std.grad is not None
    assert torch.isfinite(raw_std.grad).all()
    assert not torch.allclose(raw_std.grad, torch.zeros_like(raw_std.grad))
