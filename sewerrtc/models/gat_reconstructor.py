from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.nn import GATConv
except Exception as exc:  # pragma: no cover
    GATConv = None
    _PYG_IMPORT_ERROR = exc


def require_pyg() -> None:
    if GATConv is None:
        raise RuntimeError(
            "torch_geometric is required for the Water Research version of Project4. "
            "Install a PyTorch-compatible PyG build in your PyCharm environment, e.g. "
            "`pip install torch-geometric`, then rerun."
        ) from _PYG_IMPORT_ERROR


def batch_edge_index(edge_index: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    offsets = torch.arange(batch_size, device=edge_index.device).repeat_interleave(edge_index.shape[1]) * n_nodes
    return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)


class SparseGATReconstructor(nn.Module):
    """PyTorch Geometric GATConv sparse-sensor full-state reconstructor.

    Inputs
    ------
    sparse_depth: [B, N] observed depths, zeros elsewhere
    sensor_mask: [B, N] or [N] binary mask
    rain: [B, 1] current rainfall intensity
    node_static: [N, F] normalized node attributes
    edge_index: [2, E] directed SWMM graph edges
    """

    def __init__(self, n_nodes: int, static_dim: int, hidden_dim: int = 256, heads: int = 4, dropout: float = 0.05):
        super().__init__()
        require_pyg()
        self.n_nodes = int(n_nodes)
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        in_dim = 3 + static_dim  # sparse depth, mask, rain, static attrs
        per_head = max(8, hidden_dim // heads)
        self.input = nn.Linear(in_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, per_head, heads=heads, dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(per_head * heads, per_head, heads=heads, dropout=dropout, add_self_loops=True)
        self.norm1 = nn.LayerNorm(per_head * heads)
        self.norm2 = nn.LayerNorm(per_head * heads)
        self.out = nn.Sequential(nn.Linear(per_head * heads, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        sparse_depth: torch.Tensor,
        sensor_mask: torch.Tensor,
        rain: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if sparse_depth.ndim != 2:
            raise ValueError("sparse_depth must be [B, N]")
        B, N = sparse_depth.shape
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {N}")
        if sensor_mask.ndim == 1:
            sensor_mask = sensor_mask[None, :].expand(B, -1)
        if rain.ndim == 1:
            rain = rain[:, None]
        rain_node = rain[:, None, :].expand(B, N, 1)
        static = node_static[None, :, :].expand(B, N, -1)
        x = torch.cat([sparse_depth[:, :, None] * sensor_mask[:, :, None], sensor_mask[:, :, None], rain_node, static], dim=-1)
        x = self.input(x.reshape(B * N, -1)).relu()
        be = batch_edge_index(edge_index, B, N)
        h1 = self.gat1(x, be).relu()
        h1 = self.norm1(h1)
        h2 = self.gat2(h1, be).relu()
        h2 = self.norm2(h2 + h1)
        out = torch.relu(self.out(h2)).reshape(B, N)
        return out


def make_sensor_mask(n_nodes: int, sensor_indices: list[int], device=None) -> torch.Tensor:
    mask = torch.zeros(n_nodes, dtype=torch.float32, device=device)
    mask[sensor_indices] = 1.0
    return mask
