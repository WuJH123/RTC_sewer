"""Validation helpers for isolated Project6 v8 storage-retrofit manifests."""

from __future__ import annotations

import pandas as pd


_BLOCKED_ACTUATORS = {"MSL010.7", "MSL010.8", "MSL010.9"}
_STORAGE_CLASSES = {"storage_inlet", "storage_outlet"}


def validate_retrofit_asset_mix(assets: pd.DataFrame, *, action_dim: int) -> dict[str, int]:
    """Validate the deliberately small, hydraulically interpretable retrofit set.

    The manifest is a scenario declaration, not a reliability ranking.  It
    keeps the 109-dimensional historical action ordering intact through the
    ``action_index`` column while constraining the added facilities to a
    balanced storage/regulator/pump mix.
    """
    required = {"actuator_id", "asset_class", "action_index", "link_type"}
    missing = sorted(required.difference(assets.columns))
    if missing:
        raise ValueError(f"retrofit asset manifest missing columns: {missing}")
    if assets["actuator_id"].astype(str).duplicated().any():
        raise ValueError("retrofit asset manifest contains duplicate actuator_id values")
    blocked = sorted(set(assets["actuator_id"].astype(str)).intersection(_BLOCKED_ACTUATORS))
    if blocked:
        raise ValueError(f"retrofit asset manifest contains blocked actuator(s): {blocked}")

    indices = pd.to_numeric(assets["action_index"], errors="raise").astype(int)
    if indices.duplicated().any() or (indices < 0).any() or (indices >= int(action_dim)).any():
        raise ValueError("retrofit action_index values must be unique and within the historical action dimension")

    classes = assets["asset_class"].astype(str)
    storage_count = int(classes.isin(_STORAGE_CLASSES).sum())
    regulator_count = int(classes.eq("downstream_regulator").sum())
    pump_count = int(classes.eq("pump").sum())
    if not 4 <= storage_count <= 6:
        raise ValueError("retrofit manifest requires 4-6 storage inlet/outlet assets")
    if not 2 <= regulator_count <= 3:
        raise ValueError("retrofit manifest requires 2-3 downstream regulator/weir assets")
    if pump_count != 2:
        raise ValueError("retrofit manifest requires exactly two pumps")
    if len(assets) != storage_count + regulator_count + pump_count:
        raise ValueError("retrofit manifest contains an unsupported asset_class")
    if not assets.loc[classes.isin(_STORAGE_CLASSES), "link_type"].astype(str).eq("orifice").all():
        raise ValueError("storage-linked retrofit assets must be orifices")
    if not assets.loc[classes.eq("pump"), "link_type"].astype(str).eq("pump").all():
        raise ValueError("pump retrofit assets must have link_type=pump")
    return {
        "selected_assets": int(len(assets)),
        "storage_linked_assets": storage_count,
        "downstream_regulators": regulator_count,
        "pumps": pump_count,
    }
