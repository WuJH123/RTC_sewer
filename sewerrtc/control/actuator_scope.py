from __future__ import annotations

import pandas as pd


VALID_ACTUATOR_SCOPES = {"existing_rtc", "existing_plus_retrofit", "retrofit_only", "control_enabled"}


def enrich_temporal_joint_actuator_semantics(
    actuators: pd.DataFrame,
    retrofit_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach explicit legacy/storage semantics used by the 36-action search."""
    work = actuators.copy()
    required = {"actuator_id", "asset_class", "storage_node", "inlet_or_outlet"}
    missing = required - set(retrofit_manifest.columns)
    if missing:
        raise ValueError(f"retrofit asset manifest is missing columns: {sorted(missing)}")
    manifest = retrofit_manifest.copy()
    manifest["actuator_id"] = manifest["actuator_id"].astype(str)
    if manifest["actuator_id"].duplicated().any():
        raise ValueError("retrofit asset manifest contains duplicate actuator_id values")
    manifest = manifest.set_index("actuator_id")
    ids = work["actuator_id"].astype(str)
    retrofit_ids = set(manifest.index)
    work["is_legacy_v8"] = ~ids.isin(retrofit_ids)
    work["retrofit_asset_class"] = ids.map(manifest["asset_class"]).fillna("")
    work["retrofit_storage_group"] = ids.map(manifest["storage_node"]).fillna("")
    explicit_role = ids.map(manifest["inlet_or_outlet"]).fillna("").astype(str).str.lower()
    role_map = {"inlet": "storage_inlet", "outlet": "storage_outlet"}
    explicit_storage_role = explicit_role.map(role_map).fillna("")
    current_role = work.get("storage_control_type", pd.Series("", index=work.index)).fillna("").astype(str)
    work["storage_control_type"] = explicit_storage_role.where(explicit_storage_role.ne(""), current_role)
    return work


def select_actuators_for_scope(actuators: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Select a declared engineering control scope without changing row order."""
    scope = str(scope or "existing_rtc").strip().lower()
    if scope not in VALID_ACTUATOR_SCOPES:
        raise ValueError(f"Unknown actuator scope: {scope}")
    work = actuators.copy()
    existing = work.get("is_existing_rtc", work.get("has_internal_rule", False))
    existing = pd.Series(existing, index=work.index).fillna(False).astype(bool)
    physically_controllable = work.get("is_physically_controllable", True)
    physically_controllable = pd.Series(physically_controllable, index=work.index).fillna(False).astype(bool)
    control_enabled = work.get("control_enabled", False)
    control_enabled = pd.Series(control_enabled, index=work.index).fillna(False).astype(bool)
    if scope == "existing_rtc":
        keep = existing & physically_controllable
    elif scope == "retrofit_only":
        keep = ~existing & physically_controllable
    elif scope == "control_enabled":
        # A deployment scenario must not silently expand to every link that
        # happens to be technically controllable in the INP.  This preserves
        # audit-table order while restricting the action tensor to facilities
        # explicitly enabled for the current experiment.
        keep = control_enabled & physically_controllable
    else:
        keep = physically_controllable
    return work.loc[keep].reset_index(drop=True)
