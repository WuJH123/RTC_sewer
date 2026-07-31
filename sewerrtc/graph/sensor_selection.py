from __future__ import annotations

import numpy as np
import pandas as pd


def select_sensors(
    nodes: pd.DataFrame,
    priority_nodes: list[str],
    sensor_ratio: float,
    seed: int = 2026,
    include_priority_nodes: bool = True,
    priority_sensor_fraction: float = 0.25,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eligible = nodes[nodes["node_type"].isin(["junction", "storage"])].copy()
    n = max(1, int(round(len(eligible) * sensor_ratio)))
    priority_fraction = max(0.0, min(float(priority_sensor_fraction), 1.0))
    priority_budget = max(0, int(round(n * priority_fraction)))
    priority = eligible.iloc[0:0].copy()
    if include_priority_nodes and priority_nodes and priority_fraction > 0.0:
        priority_budget = max(1, min(len(priority_nodes), priority_budget, n))
        priority = eligible[eligible["node_id"].isin(priority_nodes)].head(priority_budget)
    rest = eligible[~eligible["node_id"].isin(priority["node_id"])]
    score = rest.get("degree_in", 0).fillna(0) + rest.get("degree_out", 0).fillna(0) + rng.random(len(rest)) * 0.01
    rest = rest.assign(sensor_score=score).sort_values("sensor_score", ascending=False)
    sensors = pd.concat([priority, rest.head(max(0, n - len(priority)))], ignore_index=True)
    sensors["sensor_index"] = range(len(sensors))
    return sensors[["sensor_index", "node_id", "node_type", "max_depth"]]
