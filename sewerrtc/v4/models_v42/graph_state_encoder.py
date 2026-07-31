"""Graph state encoder using multi-layer GATConv with skip connections."""
from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None


def _require_pyg() -> None:
    if GATConv is None:
        raise RuntimeError(
            "torch_geometric is required. Install via `pip install torch-geometric`."
        )


def batch_edge_index(edge_index: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
    """Replicate edge_index for a batched graph (same pattern as gat_reconstructor)."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    offsets = torch.arange(batch_size, device=edge_index.device).repeat_interleave(edge_index.shape[1]) * n_nodes
    return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)


class GraphStateEncoder(nn.Module):
    """Encode graph state into node embeddings via GATConv layers.

    Input
    -----
    state_history : [B, T_hist, N]
    edge_index    : [2, E]
    node_static   : [N, F_static]

    Output
    ------
    node_embed : [B*N, hidden_dim]
    """

    def __init__(
        self,
        n_nodes: int,
        n_static_features: int,
        hidden_dim: int = 32,
        gat_heads: int = 4,
        n_gat_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        _require_pyg()
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.n_gat_layers = int(n_gat_layers)
        per_head = max(4, hidden_dim // gat_heads)
        gat_out = per_head * gat_heads

        # Input: last-frame depth (1) + static features
        in_dim = 1 + n_static_features
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GATConv layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_gat_layers):
            layer_in = hidden_dim
            self.gat_layers.append(
                GATConv(layer_in, per_head, heads=gat_heads, dropout=dropout, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(gat_out))
            # Skip connection projection if dimensions differ
            if layer_in != gat_out:
                self.skips.append(nn.Linear(layer_in, gat_out, bias=False))
            else:
                self.skips.append(nn.Identity())

        self.output_dim = gat_out

    def forward(
        self,
        state_history: torch.Tensor,
        edge_index: torch.Tensor,
        node_static: torch.Tensor,
    ) -> torch.Tensor:
        B, T_hist, N = state_history.shape
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {N}")

        # Take last frame
        last_frame = state_history[:, -1, :]  # [B, N]
        static_expanded = node_static[None, :, :].expand(B, N, -1)  # [B, N, F]

        # Concatenate and project
        x = torch.cat([last_frame[:, :, None], static_expanded], dim=-1)  # [B, N, 1+F]
        x = self.input_proj(x.reshape(B * N, -1)).relu()  # [B*N, hidden_dim]

        # Batched edge index for GAT
        be = batch_edge_index(edge_index, B, N)

        # GAT layers with skip connections + LayerNorm
        for gat, norm, skip in zip(self.gat_layers, self.norms, self.skips):
            h = gat(x, be)
            h = norm(h + skip(x))
            x = h.relu()

        return x  # [B*N, output_dim]
