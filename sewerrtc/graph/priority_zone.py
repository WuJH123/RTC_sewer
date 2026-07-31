from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def select_priority_nodes(nodes: pd.DataFrame, links: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    df = nodes.copy()
    max_depth = df["max_depth"].fillna(0).to_numpy(float)
    in_deg = df.get("degree_in", 0).to_numpy(float)
    out_deg = df.get("degree_out", 0).to_numpy(float)
    storage_penalty = (df["node_type"] == "storage").astype(float).to_numpy()
    outfall_penalty = (df["node_type"] == "outfall").astype(float).to_numpy()
    score = (
        0.45 * _rank(max_depth)
        + 0.25 * _rank(in_deg)
        + 0.20 * _rank(in_deg - out_deg)
        - 0.20 * storage_penalty
        - 0.50 * outfall_penalty
    )
    df["priority_score"] = score
    keep = df[df["node_type"].isin(["junction", "storage"])].sort_values("priority_score", ascending=False).head(top_k)
    return keep[["node_id", "node_type", "invert", "max_depth", "priority_score"]].reset_index(drop=True)


def _rank(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    order = np.argsort(np.argsort(x))
    return order / max(1, len(x) - 1)

