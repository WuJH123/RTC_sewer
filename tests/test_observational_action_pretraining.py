from __future__ import annotations

import torch

from sewerrtc.models.observational_action_pretraining import (
    action_excitation,
    action_rich_sample_weights,
    actuator_neighbour_state_loss,
)


def test_action_excitation_retains_actuator_identity() -> None:
    action = torch.zeros((2, 6, 3))
    action[0, 2:, 1] = 0.4
    action[1, 1:4, 2] = 1.0

    excitation = action_excitation(action)

    assert excitation.shape == (2, 3)
    assert excitation[0, 1] > 0
    assert excitation[0, 0] == 0
    assert excitation[1, 2] > excitation[1, 1]


def test_action_rich_weights_upweight_dynamic_sequences() -> None:
    excitation = torch.tensor([[0.0, 0.0], [0.0, 0.5]])

    weights = action_rich_sample_weights(excitation, gain=3.0, minimum_excitation=0.05)

    torch.testing.assert_close(weights, torch.tensor([1.0, 4.0]))


def test_actuator_neighbour_loss_focuses_changed_actuator_endpoints() -> None:
    predicted = torch.zeros((1, 2, 3))
    target = torch.zeros_like(predicted)
    target[:, :, 1] = 2.0
    excitation = torch.tensor([[0.5, 0.0]])
    action_local_map = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    focused = actuator_neighbour_state_loss(
        predicted,
        target,
        excitation=excitation,
        action_local_map=action_local_map,
        scale=1.0,
    )
    unrelated = actuator_neighbour_state_loss(
        predicted,
        target,
        excitation=torch.tensor([[0.0, 0.5]]),
        action_local_map=action_local_map,
        scale=1.0,
    )

    assert focused > 0
    assert unrelated == 0
