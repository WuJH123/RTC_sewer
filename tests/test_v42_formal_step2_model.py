import torch

from sewerrtc.v4.models_v42.hydraulic_multi_reference import diffuse_action_embeddings


def test_action_context_reaches_downstream_priority_node():
    # Facility action is attached at node 0; node 2 is only reachable through
    # the physical graph.  Formal PFV nodes need this causal action path.
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    action = torch.zeros(1, 1, 3, 1)
    action[:, :, 0, :] = 1.0
    propagated = diffuse_action_embeddings(action, edge_index, steps=2)
    assert propagated[0, 0, 2, 0] > 0.0


def test_zero_action_diffusion_preserves_legacy_embedding():
    action = torch.randn(1, 2, 3, 4)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    torch.testing.assert_close(
        diffuse_action_embeddings(action, edge_index, steps=0), action
    )
