"""Graph state encoder with temporal history encoding and graph attention.

The V4.2 contract uses a 60-minute history (13 frames at 5-minute spacing).
Historically this module discarded the first 12 frames and only encoded the
checkpoint frame.  That made the model effectively state-only and removed the
hydraulic trend information that the trajectory contract was designed to
provide.  The encoder now applies a shared per-node GRU over *all* supplied
history frames before graph message passing.
"""
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
    """Replicate ``edge_index`` for a batch of identical graph topologies."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    offsets = (
        torch.arange(batch_size, device=edge_index.device)
        .repeat_interleave(edge_index.shape[1])
        * n_nodes
    )
    return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)


class GraphStateEncoder(nn.Module):
    """Encode multi-frame node state histories into graph node embeddings.

    Parameters
    ----------
    state_history:
        ``[B, T_hist, N]`` depth/state history.  Every history frame contributes
        to the temporal embedding; no frame is silently discarded.
    edge_index:
        ``[2, E]`` graph connectivity.
    node_static:
        ``[N, F_static]`` frozen node attributes.

    Returns
    -------
    torch.Tensor
        ``[B*N, output_dim]`` node embeddings.
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

        # Shared temporal encoder: one GRU is applied independently to every
        # node.  Reshaping B*N as the batch dimension preserves node identity
        # while learning depth trends over the complete history window.
        self.temporal_gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Combine the learned history representation with static attributes.
        self.input_proj = nn.Linear(hidden_dim + n_static_features, hidden_dim)

        per_head = max(4, hidden_dim // gat_heads)
        gat_out = per_head * gat_heads
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        for _ in range(n_gat_layers):
            layer_in = hidden_dim if not self.gat_layers else gat_out
            self.gat_layers.append(
                GATConv(
                    layer_in,
                    per_head,
                    heads=gat_heads,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(gat_out))
            self.skips.append(
                nn.Identity()
                if layer_in == gat_out
                else nn.Linear(layer_in, gat_out, bias=False)
            )

        self.output_dim = gat_out

    def encode_history(self, state_history: torch.Tensor) -> torch.Tensor:
        """Return the per-node temporal embedding ``[B, N, hidden_dim]``."""
        if state_history.ndim != 3:
            raise ValueError("state_history must have shape [B, T_hist, N]")
        B, T_hist, N = state_history.shape
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {N}")
        if T_hist < 2:
            raise ValueError("state_history must contain at least two frames")

        # [B, T, N] -> [B, N, T, 1] -> [B*N, T, 1]
        seq = state_history.permute(0, 2, 1).unsqueeze(-1)
        seq = seq.reshape(B * N, T_hist, 1)
        _, h_last = self.temporal_gru(seq)
        return h_last[-1].reshape(B, N, self.hidden_dim)

    def forward(
        self,
        state_history: torch.Tensor,
        edge_index: torch.Tensor,
        node_static: torch.Tensor,
    ) -> torch.Tensor:
        B, _, N = state_history.shape
        history_embed = self.encode_history(state_history)

        if node_static.ndim != 2 or node_static.shape[0] != N:
            raise ValueError(
                f"node_static must have shape [N, F]; got {tuple(node_static.shape)}"
            )
        static_expanded = node_static[None, :, :].expand(B, -1, -1)
        x = torch.cat([history_embed, static_expanded], dim=-1)
        x = self.input_proj(x.reshape(B * N, -1)).relu()

        be = batch_edge_index(edge_index, B, N)
        for gat, norm, skip in zip(self.gat_layers, self.norms, self.skips):
            h = gat(x, be)
            x = norm(h + skip(x)).relu()

        return x
