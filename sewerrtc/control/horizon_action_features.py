from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd


IDENTITY_HASH_DIM = 16
IDENTITY_HASH_PHASES = ("all", "early", "mid", "late")

ACTION_FEATURE_COLUMNS = [
    "action_mean",
    "action_min",
    "action_max",
    "action_std",
    "action_delta_mean",
    "action_delta_min",
    "action_delta_max",
    "action_delta_std",
    "action_delta_abs_mean",
    "action_delta_abs_max",
    "sequence_action_mean",
    "sequence_action_min",
    "sequence_action_max",
    "sequence_action_std",
    "sequence_delta_mean",
    "sequence_delta_min",
    "sequence_delta_max",
    "sequence_delta_std",
    "sequence_delta_abs_mean",
    "sequence_delta_abs_max",
    "sequence_early_action_mean",
    "sequence_mid_action_mean",
    "sequence_late_action_mean",
    "sequence_early_delta_mean",
    "sequence_mid_delta_mean",
    "sequence_late_delta_mean",
    "sequence_early_delta_abs_mean",
    "sequence_mid_delta_abs_mean",
    "sequence_late_delta_abs_mean",
    "pump_action_mean",
    "pump_sequence_action_mean",
    "pump_delta_mean",
    "pump_sequence_early_action_mean",
    "pump_sequence_mid_action_mean",
    "pump_sequence_late_action_mean",
    "pump_sequence_early_delta_mean",
    "pump_sequence_mid_delta_mean",
    "pump_sequence_late_delta_mean",
    "regulator_action_mean",
    "regulator_sequence_action_mean",
    "regulator_delta_mean",
    "regulator_sequence_early_action_mean",
    "regulator_sequence_mid_action_mean",
    "regulator_sequence_late_action_mean",
    "regulator_sequence_early_delta_mean",
    "regulator_sequence_mid_delta_mean",
    "regulator_sequence_late_delta_mean",
    "storage_action_mean",
    "storage_sequence_action_mean",
    "storage_delta_mean",
    "storage_sequence_early_action_mean",
    "storage_sequence_mid_action_mean",
    "storage_sequence_late_action_mean",
    "storage_sequence_early_delta_mean",
    "storage_sequence_mid_delta_mean",
    "storage_sequence_late_delta_mean",
    "storage_inlet_action_mean",
    "storage_inlet_sequence_action_mean",
    "storage_inlet_delta_mean",
    "storage_inlet_sequence_early_action_mean",
    "storage_inlet_sequence_mid_action_mean",
    "storage_inlet_sequence_late_action_mean",
    "storage_inlet_sequence_early_delta_mean",
    "storage_inlet_sequence_mid_delta_mean",
    "storage_inlet_sequence_late_delta_mean",
    "storage_outlet_action_mean",
    "storage_outlet_sequence_action_mean",
    "storage_outlet_delta_mean",
    "storage_outlet_sequence_early_action_mean",
    "storage_outlet_sequence_mid_action_mean",
    "storage_outlet_sequence_late_action_mean",
    "storage_outlet_sequence_early_delta_mean",
    "storage_outlet_sequence_mid_delta_mean",
    "storage_outlet_sequence_late_delta_mean",
    "priority_action_mean",
    "priority_sequence_action_mean",
    "priority_delta_mean",
    "priority_sequence_early_action_mean",
    "priority_sequence_mid_action_mean",
    "priority_sequence_late_action_mean",
    "priority_sequence_early_delta_mean",
    "priority_sequence_mid_delta_mean",
    "priority_sequence_late_delta_mean",
    "priority_delta_abs_mean",
    "retain_fraction",
    "release_fraction",
    "priority_retain_fraction",
    "priority_release_fraction",
] + [
    f"{kind}_hash_{phase}_{i:02d}"
    for kind in ("actuator", "path")
    for phase in IDENTITY_HASH_PHASES
    for i in range(IDENTITY_HASH_DIM)
]

_ACTION_CONTEXT_CACHE: dict[tuple, dict[str, object]] = {}


def _clean_id(value: object) -> str:
    text = str(value or "").strip()
    return text.split(":", 1)[1] if text.startswith("a:") else text


def _feature_safe_id(value: object) -> str:
    """Return a stable, model-safe token while preserving asset identity."""
    token = re.sub(r"[^0-9A-Za-z]+", "_", _clean_id(value)).strip("_")
    return token or "unnamed"


def _aligned_actuator_frame(action_ids: list[str], actuators: pd.DataFrame | None) -> pd.DataFrame:
    action_ids = [_clean_id(x) for x in action_ids]
    if actuators is None or actuators.empty:
        return pd.DataFrame({"actuator_id": action_ids})
    frame = actuators.copy()
    if "actuator_id" not in frame:
        frame["actuator_id"] = ""
    frame["actuator_id"] = frame["actuator_id"].astype(str)
    frame = frame.drop_duplicates("actuator_id").set_index("actuator_id", drop=False)
    aligned = []
    for aid in action_ids:
        if aid in frame.index:
            aligned.append(frame.loc[aid].to_dict())
        else:
            aligned.append({"actuator_id": aid})
    return pd.DataFrame(aligned)


def _priority_actuator_ids(priority_to_actuators: pd.DataFrame | None) -> set[str]:
    if priority_to_actuators is None or priority_to_actuators.empty or "actuator_id" not in priority_to_actuators:
        return set()
    return {_clean_id(x) for x in priority_to_actuators["actuator_id"].dropna().astype(str)}


def _mean(values: np.ndarray, mask: np.ndarray | None = None, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        arr = arr[mask] if arr.ndim == 1 else arr[:, mask]
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float(default)


def _fraction(values: np.ndarray, mask: np.ndarray | None = None, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=bool)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        arr = arr[mask]
    return float(np.mean(arr)) if arr.size else float(default)


def _sequence_phase_slices(n_steps: int) -> dict[str, slice]:
    n = max(1, int(n_steps))
    if n == 1:
        return {"early": slice(0, 1), "mid": slice(0, 1), "late": slice(0, 1)}
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, (2 * n) // 3)
    cut2 = min(cut2, n)
    return {
        "early": slice(0, cut1),
        "mid": slice(cut1, cut2),
        "late": slice(cut2, n),
    }


def _hash_vector(token: str) -> np.ndarray:
    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:IDENTITY_HASH_DIM]
    return np.where(bits > 0, 1.0, -1.0).astype(float) / np.sqrt(float(IDENTITY_HASH_DIM))


def _path_hash_by_actuator(
    action_ids: list[str], priority_to_actuators: pd.DataFrame | None
) -> np.ndarray:
    vectors = np.vstack([_hash_vector(f"actuator={aid}|path=unmapped") for aid in action_ids])
    if priority_to_actuators is None or priority_to_actuators.empty or "actuator_id" not in priority_to_actuators:
        return vectors
    table = priority_to_actuators.copy()
    table["actuator_id"] = table["actuator_id"].astype(str).map(_clean_id)
    for idx, aid in enumerate(action_ids):
        rows = table[table["actuator_id"].eq(aid)]
        if rows.empty:
            continue
        tokens = []
        for _, row in rows.iterrows():
            tokens.append(
                "|".join(
                    [
                        f"priority={row.get('priority_node', '')}",
                        f"actuator={aid}",
                        f"role={row.get('asset_role', '')}",
                        f"direction={row.get('direction', row.get('matched_node_field', ''))}",
                        f"node={row.get('matched_node', '')}",
                        f"distance={row.get('influence_path_length', '')}",
                    ]
                )
            )
        vectors[idx] = np.mean(np.vstack([_hash_vector(token) for token in tokens]), axis=0)
    return vectors


def _action_context(
    action_ids: list[str],
    actuators: pd.DataFrame | None,
    priority_to_actuators: pd.DataFrame | None,
) -> dict[str, object]:
    """Cache static facility/path features shared by all timesteps in a detail."""
    clean_ids = tuple(_clean_id(x) for x in action_ids)
    key = (clean_ids, id(actuators), id(priority_to_actuators))
    cached = _ACTION_CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached
    n = len(clean_ids)
    aligned = _aligned_actuator_frame(list(clean_ids), actuators)
    link_type = aligned.get("link_type", pd.Series([""] * n)).fillna("").astype(str).str.lower()
    storage_role = aligned.get("storage_control_type", pd.Series([""] * n)).fillna("").astype(str).str.lower()
    asset_role = aligned.get("asset_role", pd.Series([""] * n)).fillna("").astype(str).str.lower()
    near_storage = (
        aligned.get("near_storage", pd.Series([False] * n))
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    ids = aligned.get("actuator_id", pd.Series(list(clean_ids))).astype(str).map(_clean_id).tolist()
    inlet_mask = storage_role.str.contains("inlet", na=False).to_numpy() | asset_role.str.contains("inlet", na=False).to_numpy()
    outlet_mask = storage_role.str.contains("outlet", na=False).to_numpy() | asset_role.str.contains("outlet", na=False).to_numpy()
    storage_mask = near_storage | inlet_mask | outlet_mask | asset_role.str.contains("storage", na=False).to_numpy()
    regulator_mask = (
        link_type.str.contains("orifice|weir", regex=True, na=False).to_numpy()
        | asset_role.str.contains("orifice|weir|regulator", regex=True, na=False).to_numpy()
    ) & ~storage_mask
    pump_mask = link_type.str.contains("pump", na=False).to_numpy() | asset_role.str.contains("pump", na=False).to_numpy()
    priority_ids = _priority_actuator_ids(priority_to_actuators)
    priority_mask = np.asarray([aid in priority_ids for aid in ids], dtype=bool)
    if not np.any(priority_mask):
        priority_mask = np.ones(n, dtype=bool)
    cached = {
        "ids": ids,
        "pump_mask": pump_mask,
        "inlet_mask": inlet_mask,
        "outlet_mask": outlet_mask,
        "storage_mask": storage_mask,
        "regulator_mask": regulator_mask,
        "priority_mask": priority_mask,
        "actuator_vectors": np.vstack([_hash_vector(f"actuator={aid}") for aid in clean_ids]),
        "path_vectors": _path_hash_by_actuator(list(clean_ids), priority_to_actuators),
    }
    _ACTION_CONTEXT_CACHE[key] = cached
    return cached


def build_action_feature_map(
    action_ids: list[str],
    action: np.ndarray,
    *,
    sequence: np.ndarray | None = None,
    reference_action: np.ndarray | None = None,
    actuators: pd.DataFrame | None = None,
    priority_to_actuators: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Build action-aware horizon features shared by training and online MPC.

    The earlier Project5 horizon scorer only saw global action mean/min/max/std,
    so changing one influential facility barely moved the feature vector. These
    role-aware summaries expose pump/storage/priority-domain changes while
    staying city-agnostic: roles come from the actuator table and influence
    domains rather than Wuhan-specific names.
    """
    action_ids = [_clean_id(x) for x in action_ids]
    action_arr = np.asarray(action, dtype=float).reshape(-1)
    n = len(action_ids)
    if n <= 0:
        n = int(action_arr.size)
        action_ids = [f"actuator_{i}" for i in range(n)]
    if action_arr.size != n:
        action_arr = np.resize(action_arr, n) if action_arr.size else np.ones(n, dtype=float)
    action_arr = np.nan_to_num(action_arr, nan=1.0, posinf=1.0, neginf=0.0)
    if sequence is None:
        seq = action_arr.reshape(1, -1)
    else:
        seq = np.asarray(sequence, dtype=float)
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)
        if seq.shape[1] != n:
            seq = np.resize(seq, (seq.shape[0], n))
        seq = np.nan_to_num(seq, nan=1.0, posinf=1.0, neginf=0.0)
    if reference_action is None:
        reference_seq = np.repeat(action_arr.reshape(1, -1), seq.shape[0], axis=0)
    else:
        reference_seq = np.asarray(reference_action, dtype=float)
        if reference_seq.ndim == 1:
            reference_seq = np.repeat(reference_seq.reshape(1, -1), seq.shape[0], axis=0)
        elif reference_seq.ndim != 2:
            reference_seq = reference_seq.reshape(1, -1)
        if reference_seq.shape != seq.shape:
            reference_seq = np.resize(reference_seq, seq.shape)
        reference_seq = np.nan_to_num(reference_seq, nan=1.0, posinf=1.0, neginf=0.0)
    reference = reference_seq[0]
    context = _action_context(action_ids, actuators, priority_to_actuators)
    ids = context["ids"]
    pump_mask = context["pump_mask"]
    inlet_mask = context["inlet_mask"]
    outlet_mask = context["outlet_mask"]
    storage_mask = context["storage_mask"]
    regulator_mask = context["regulator_mask"]
    priority_mask = context["priority_mask"]

    delta = action_arr - reference
    seq_delta = seq - reference_seq
    global_mean = _mean(action_arr, default=1.0)
    seq_mean = _mean(seq, default=global_mean)
    delta_mean = _mean(delta, default=0.0)
    features = {
        "action_mean": global_mean,
        "action_min": float(np.nanmin(action_arr)) if action_arr.size else 1.0,
        "action_max": float(np.nanmax(action_arr)) if action_arr.size else 1.0,
        "action_std": float(np.nanstd(action_arr)) if action_arr.size > 1 else 0.0,
        "action_delta_mean": delta_mean,
        "action_delta_min": float(np.nanmin(delta)) if delta.size else 0.0,
        "action_delta_max": float(np.nanmax(delta)) if delta.size else 0.0,
        "action_delta_std": float(np.nanstd(delta)) if delta.size > 1 else 0.0,
        "action_delta_abs_mean": _mean(np.abs(delta), default=0.0),
        "action_delta_abs_max": float(np.nanmax(np.abs(delta))) if delta.size else 0.0,
        "sequence_action_mean": seq_mean,
        "sequence_action_min": float(np.nanmin(seq)) if seq.size else global_mean,
        "sequence_action_max": float(np.nanmax(seq)) if seq.size else global_mean,
        "sequence_action_std": float(np.nanstd(seq)) if seq.size > 1 else 0.0,
        "sequence_delta_mean": _mean(seq_delta, default=0.0),
        "sequence_delta_min": float(np.nanmin(seq_delta)) if seq_delta.size else 0.0,
        "sequence_delta_max": float(np.nanmax(seq_delta)) if seq_delta.size else 0.0,
        "sequence_delta_std": float(np.nanstd(seq_delta)) if seq_delta.size > 1 else 0.0,
        "sequence_delta_abs_mean": _mean(np.abs(seq_delta), default=0.0),
        "sequence_delta_abs_max": float(np.nanmax(np.abs(seq_delta))) if seq_delta.size else 0.0,
        "retain_fraction": _fraction(delta < -1.0e-6),
        "release_fraction": _fraction(delta > 1.0e-6),
    }

    for phase, phase_slice in _sequence_phase_slices(seq.shape[0]).items():
        phase_seq = seq[phase_slice, :]
        phase_delta = seq_delta[phase_slice, :]
        features[f"sequence_{phase}_action_mean"] = _mean(phase_seq, default=seq_mean)
        features[f"sequence_{phase}_delta_mean"] = _mean(phase_delta, default=0.0)
        features[f"sequence_{phase}_delta_abs_mean"] = _mean(np.abs(phase_delta), default=0.0)

    actuator_vectors = context["actuator_vectors"]
    path_vectors = context["path_vectors"]
    phase_weights = {
        "all": np.mean(seq_delta, axis=0) if seq_delta.shape[0] else np.zeros(len(action_ids), dtype=float)
    }
    for phase, phase_slice in _sequence_phase_slices(seq.shape[0]).items():
        phase_delta = seq_delta[phase_slice, :]
        phase_weights[phase] = (
            np.mean(phase_delta, axis=0) if phase_delta.shape[0] else np.zeros(len(action_ids), dtype=float)
        )
    for phase, weights in phase_weights.items():
        for kind, vectors in (("actuator", actuator_vectors), ("path", path_vectors)):
            signature = np.asarray(weights, dtype=float) @ vectors
            for i, value in enumerate(signature):
                features[f"{kind}_hash_{phase}_{i:02d}"] = float(value)

    for name, mask in [
        ("pump", pump_mask),
        ("regulator", regulator_mask),
        ("storage", storage_mask),
        ("storage_inlet", inlet_mask),
        ("storage_outlet", outlet_mask),
        ("priority", priority_mask),
    ]:
        features[f"{name}_action_mean"] = _mean(action_arr, mask, default=global_mean)
        features[f"{name}_sequence_action_mean"] = _mean(seq, mask, default=seq_mean)
        features[f"{name}_delta_mean"] = _mean(delta, mask, default=delta_mean)
        for phase, phase_slice in _sequence_phase_slices(seq.shape[0]).items():
            phase_seq = seq[phase_slice, :]
            phase_delta = seq_delta[phase_slice, :]
            features[f"{name}_sequence_{phase}_action_mean"] = _mean(
                phase_seq,
                mask,
                default=features[f"{name}_sequence_action_mean"],
            )
            features[f"{name}_sequence_{phase}_delta_mean"] = _mean(
                phase_delta,
                mask,
                default=features[f"{name}_delta_mean"],
            )
    features["priority_delta_abs_mean"] = _mean(np.abs(delta), priority_mask, default=features["action_delta_abs_mean"])
    features["priority_retain_fraction"] = _fraction(delta < -1.0e-6, priority_mask, default=features["retain_fraction"])
    features["priority_release_fraction"] = _fraction(delta > 1.0e-6, priority_mask, default=features["release_fraction"])

    # Explicit facility-wise action features are essential for a fixed
    # deployment set.  Hash and role summaries remain useful transferable
    # descriptors, but they can collide or average away the difference between
    # two storage links.  The same function is called by both the offline
    # rollout builder and online MPC predictor, so this does not create a
    # train/serve encoding mismatch.
    for idx, aid in enumerate(ids):
        key = _feature_safe_id(aid)
        features[f"asset_{key}_action"] = float(action_arr[idx])
        features[f"asset_{key}_delta"] = float(delta[idx])
        for phase, phase_slice in _sequence_phase_slices(seq.shape[0]).items():
            features[f"asset_{key}_{phase}_action"] = _mean(
                seq[phase_slice, idx], default=float(action_arr[idx])
            )
            features[f"asset_{key}_{phase}_delta"] = _mean(
                seq_delta[phase_slice, idx], default=float(delta[idx])
            )
    return {str(col): float(value) for col, value in features.items()}
