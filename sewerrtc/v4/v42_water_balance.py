"""V4.2 water balance baseline — physics-based mass conservation model.

Uses V(t+1) = V(t) + Qin*dt - Qout*dt - Qflood*dt to predict system-level
hydraulic metrics over a 12-step horizon.  Calibrated via event-grouped
5-fold CV against the trajectory dataset's observed depth evolution.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZON_STEPS = 12
DT_MIN = 10  # horizon_interval_min from graph schema
DT_SEC = DT_MIN * 60.0
N_NODES = 932
N_FACILITIES = 36

# Output sub-directory (relative to output_root)
WATER_BALANCE_DIR = "models/v42_water_balance"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_trajectory_data(trajectory_dir: Path) -> pd.DataFrame:
    """Load the trajectory manifest parquet."""
    parquet = trajectory_dir / "trajectory_manifest_v42.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"trajectory manifest not found: {parquet}")
    return pd.read_parquet(parquet)


def _load_graph_schema(trajectory_dir: Path) -> dict:
    """Load graph schema JSON."""
    path = trajectory_dir / "graph_schema_v42.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_node_schema(trajectory_dir: Path) -> dict:
    """Load node feature schema JSON (contains is_storage, is_outfall info)."""
    path = trajectory_dir / "node_feature_schema_v42.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_array(raw: str | list) -> np.ndarray:
    """Parse a JSON-encoded array (string or already parsed)."""
    if isinstance(raw, list):
        return np.array(raw, dtype=np.float64)
    return np.array(json.loads(raw), dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_node_masks(node_schema: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for storage and outfall nodes."""
    node_ids = node_schema["node_ids"]
    n = len(node_ids)
    # The node_feature_schema has depth_columns and flood_columns which
    # list ALL nodes.  We need to figure out is_storage / is_outfall from
    # the node naming conventions or from the static features.
    # Since the schema doesn't directly give per-node flags, we use
    # naming heuristics consistent with the SWMM model:
    #   - Storage: names containing "Tank", "ST_", "storage"
    #   - Outfall: names starting with "OF" or "OFJ" or containing "Out"
    is_storage = np.zeros(n, dtype=bool)
    is_outfall = np.zeros(n, dtype=bool)
    for i, nid in enumerate(node_ids):
        nid_lower = nid.lower()
        if "tank" in nid_lower or "st_" in nid_lower or "storage" in nid_lower:
            is_storage[i] = True
        if (
            nid_lower.startswith("of")
            or nid_lower.startswith("ofj")
            or "out" in nid_lower
            or "wwplant" in nid_lower
        ):
            is_outfall[i] = True
    return is_storage, is_outfall


def _extract_sample_features(
    row: pd.Series,
    is_storage: np.ndarray,
    is_outfall: np.ndarray,
) -> dict[str, Any]:
    """Extract features and targets from a single trajectory sample."""
    event_id = row["event_id"]

    # --- Parse trajectory depths (932 nodes × 12 steps) ---
    traj_raw = row["trajectory_depth_candidate"]
    traj = _parse_json_array(traj_raw)
    # Reshape: the trajectory is a flat array of 932*12 values
    # stored as [node0_step0, node0_step1, ..., node0_step11, node1_step0, ...]
    # OR as [step0_allNodes, step1_allNodes, ...]
    # Based on the data, it appears to be stored as a list of 12 sublists
    # each of length 932 (time-major) or 932 sublists of length 12.
    # Let's check: the first element is a list of lists
    if traj.ndim == 1:
        # Try reshape to (12, 932) — time-major
        if traj.size == HORIZON_STEPS * N_NODES:
            traj = traj.reshape(HORIZON_STEPS, N_NODES)
        else:
            # Might be nested JSON
            traj = traj.reshape(-1, N_NODES)
    # If it's a list of lists (from JSON), it may already be (12, 932) or (932, 12)
    if traj.shape == (N_NODES, HORIZON_STEPS):
        traj = traj.T  # -> (12, 932)
    elif traj.shape[0] != HORIZON_STEPS and traj.shape[1] == HORIZON_STEPS:
        traj = traj.T

    # --- Parse history depth (initial state) ---
    hist_raw = row["history_depth"]
    hist = _parse_json_array(hist_raw)
    # history_depth is 7 frames × 932 nodes; take the last frame as initial
    if hist.ndim == 1:
        if hist.size == 7 * N_NODES:
            hist = hist.reshape(7, N_NODES)
        else:
            hist = hist.reshape(-1, N_NODES)
    # Initial depth = last history frame
    if hist.shape[0] >= 7:
        initial_depth = hist[-1]  # (932,)
    else:
        initial_depth = hist[0]

    # --- Parse rainfall forecast ---
    rain_raw = row["rainfall_forecast"]
    rain = _parse_json_array(rain_raw)
    # rainfall_forecast: 12 steps, possibly per-node or system-wide
    if rain.ndim == 1 and rain.size == HORIZON_STEPS:
        rain_per_step = rain  # (12,)
    elif rain.ndim == 1 and rain.size > HORIZON_STEPS:
        # Might be (12 * something); take first 12 or reshape
        rain_per_step = rain[:HORIZON_STEPS]
    elif rain.ndim == 2:
        # (12, n_nodes) or similar — aggregate to system rainfall
        rain_per_step = rain.mean(axis=-1) if rain.shape[0] == HORIZON_STEPS else rain.mean(axis=0)
    else:
        rain_per_step = np.zeros(HORIZON_STEPS)

    # --- Parse action sequence ---
    action_raw = row["candidate_action_seq"]
    actions = _parse_json_array(action_raw)
    # (12, 36) — 12 steps × 36 facilities
    if actions.ndim == 1:
        if actions.size == HORIZON_STEPS * N_FACILITIES:
            actions = actions.reshape(HORIZON_STEPS, N_FACILITIES)
        else:
            actions = actions.reshape(-1, N_FACILITIES)
    if actions.shape[0] != HORIZON_STEPS:
        actions = actions[:HORIZON_STEPS]

    # --- Compute targets from trajectory data ---
    # System storage: sum of depths at storage nodes
    storage_mask = is_storage[:traj.shape[1]] if traj.shape[1] <= len(is_storage) else is_storage
    outfall_mask = is_outfall[:traj.shape[1]] if traj.shape[1] <= len(is_outfall) else is_outfall

    # Ensure traj has the right number of nodes
    n_traj_nodes = traj.shape[1]
    if n_traj_nodes < len(is_storage):
        storage_mask = is_storage[:n_traj_nodes]
        outfall_mask = is_outfall[:n_traj_nodes]
    else:
        storage_mask = np.zeros(n_traj_nodes, dtype=bool)
        outfall_mask = np.zeros(n_traj_nodes, dtype=bool)
        storage_mask[:len(is_storage)] = is_storage
        outfall_mask[:len(is_outfall)] = is_outfall

    # System storage trajectory (sum across storage nodes per step)
    storage_traj = traj[:, storage_mask].sum(axis=1)  # (12,)
    system_storage = float(storage_traj.sum())

    # Outfall flow proxy: sum of depths at outfall nodes (indicates flow)
    outfall_traj = traj[:, outfall_mask].sum(axis=1)  # (12,)
    outfall_flow = float(outfall_traj.sum())

    # Flooding: depth exceeding max capacity.  Use a simple threshold:
    # flooding occurs when depth > some threshold.  Since we don't have
    # max_depth per node here, use the 95th percentile of initial depth
    # as a proxy for capacity.
    depth_capacity = np.percentile(initial_depth, 95) + 1e-6
    flood_excess = np.maximum(traj - depth_capacity, 0)  # (12, n_nodes)
    flood_rate_traj = flood_excess.sum(axis=1)  # (12,)
    total_flooding_rate = float(flood_rate_traj.sum())

    # TFV = integral of flooding rate over time (sum × dt)
    tfv = float(flood_rate_traj.sum() * DT_SEC)

    # Peak = maximum flooding rate
    peak = float(flood_rate_traj.max())

    # --- Features for the model ---
    # Aggregate rainfall
    total_rain = float(rain_per_step.sum())
    max_rain = float(rain_per_step.max())
    mean_rain = float(rain_per_step.mean())

    # Action features
    action_intensity = float(actions.mean())
    action_sum = float(actions.sum())

    # Initial volume (sum of initial depths)
    initial_volume = float(initial_depth.sum())
    initial_storage = float(initial_depth[storage_mask[:len(initial_depth)]].sum())

    return {
        "event_id": event_id,
        # Targets
        "system_storage": system_storage,
        "outfall_flow": outfall_flow,
        "total_flooding_rate": total_flooding_rate,
        "tfv": tfv,
        "peak": peak,
        # Features
        "total_rain": total_rain,
        "max_rain": max_rain,
        "mean_rain": mean_rain,
        "action_intensity": action_intensity,
        "action_sum": action_sum,
        "initial_volume": initial_volume,
        "initial_storage": initial_storage,
        # Per-step features (for trajectory prediction)
        "rain_per_step": rain_per_step,
        "storage_traj": storage_traj,
        "outfall_traj": outfall_traj,
        "flood_rate_traj": flood_rate_traj,
    }


# ---------------------------------------------------------------------------
# Water Balance Model
# ---------------------------------------------------------------------------

class WaterBalanceBaseline:
    """Physics-based water balance baseline model.

    Predicts 5 system-level targets using mass-conservation-inspired
    linear features:
      1. system_storage — total volume in storage units
      2. outfall_flow   — total outflow at outfalls
      3. total_flooding_rate — system-wide flooding rate
      4. tfv — total flood volume
      5. peak — maximum flooding rate
    """

    TARGET_NAMES = [
        "system_storage",
        "outfall_flow",
        "total_flooding_rate",
        "tfv",
        "peak",
    ]

    def __init__(self) -> None:
        self.models: dict[str, Ridge] = {}
        self.coefficients: dict[str, list[float]] = {}
        self.intercepts: dict[str, float] = {}
        self.feature_names: list[str] = []

    def _build_features(self, samples: list[dict]) -> np.ndarray:
        """Build feature matrix from extracted sample features."""
        X = np.zeros((len(samples), 6), dtype=np.float64)
        for i, s in enumerate(samples):
            X[i, 0] = s["initial_volume"]
            X[i, 1] = s["initial_storage"]
            X[i, 2] = s["total_rain"]
            X[i, 3] = s["max_rain"]
            X[i, 4] = s["mean_rain"]
            X[i, 5] = s["action_intensity"]
        return X

    def fit(self, samples: list[dict]) -> None:
        """Calibrate the model on training samples."""
        self.feature_names = [
            "initial_volume",
            "initial_storage",
            "total_rain",
            "max_rain",
            "mean_rain",
            "action_intensity",
        ]
        X = self._build_features(samples)
        for target in self.TARGET_NAMES:
            y = np.array([s[target] for s in samples], dtype=np.float64)
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            self.models[target] = model
            self.coefficients[target] = model.coef_.tolist()
            self.intercepts[target] = float(model.intercept_)

    def predict(self, samples: list[dict]) -> dict[str, np.ndarray]:
        """Predict targets for given samples."""
        X = self._build_features(samples)
        preds = {}
        for target in self.TARGET_NAMES:
            preds[target] = self.models[target].predict(X)
        return preds

    def evaluate(
        self, samples: list[dict], predictions: dict[str, np.ndarray]
    ) -> dict[str, dict[str, float]]:
        """Compute R², MAE, sign accuracy per target."""
        results = {}
        for target in self.TARGET_NAMES:
            y_true = np.array([s[target] for s in samples], dtype=np.float64)
            y_pred = predictions[target]

            r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
            mae = float(mean_absolute_error(y_true, y_pred))

            # Sign accuracy: for delta targets, does the model get the
            # direction right?  Use (value - median) as sign reference.
            median_val = np.median(y_true)
            sign_true = np.sign(y_true - median_val)
            sign_pred = np.sign(y_pred - median_val)
            sign_acc = float(np.mean(sign_true == sign_pred))

            results[target] = {
                "r2": r2,
                "mae": mae,
                "sign_accuracy": sign_acc,
                "n_samples": len(y_true),
                "y_mean": float(np.mean(y_true)),
                "y_std": float(np.std(y_true)),
                "y_pred_mean": float(np.mean(y_pred)),
                "y_pred_std": float(np.std(y_pred)),
            }
        return results


# ---------------------------------------------------------------------------
# Event-grouped cross-validation
# ---------------------------------------------------------------------------

def _event_grouped_cv(
    samples: list[dict],
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run 5-fold CV grouped by event.

    Returns per-fold and aggregate metrics.
    """
    # Group samples by event
    event_samples: dict[str, list[dict]] = {}
    for s in samples:
        eid = s["event_id"]
        if eid not in event_samples:
            event_samples[eid] = []
        event_samples[eid].append(s)

    events = sorted(event_samples.keys())
    n_events = len(events)

    # Split events into folds
    fold_events = np.array_split(np.array(events), n_folds)

    fold_results = []
    for fold_idx in range(n_folds):
        val_events = set(fold_events[fold_idx].tolist())
        train_samples = []
        val_samples = []
        for eid, esamples in event_samples.items():
            if eid in val_events:
                val_samples.extend(esamples)
            else:
                train_samples.extend(esamples)

        if len(train_samples) == 0 or len(val_samples) == 0:
            continue

        model = WaterBalanceBaseline()
        model.fit(train_samples)
        preds = model.predict(val_samples)
        metrics = model.evaluate(val_samples, preds)

        fold_results.append({
            "fold": fold_idx,
            "val_events": sorted(val_events),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "metrics": metrics,
        })

    # Aggregate across folds
    agg_metrics: dict[str, dict[str, float]] = {}
    for target in WaterBalanceBaseline.TARGET_NAMES:
        r2s = [fr["metrics"][target]["r2"] for fr in fold_results]
        maes = [fr["metrics"][target]["mae"] for fr in fold_results]
        sign_accs = [fr["metrics"][target]["sign_accuracy"] for fr in fold_results]
        agg_metrics[target] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "sign_accuracy_mean": float(np.mean(sign_accs)),
            "sign_accuracy_std": float(np.std(sign_accs)),
        }

    return {
        "n_folds": n_folds,
        "n_events": n_events,
        "n_samples": len(samples),
        "fold_results": fold_results,
        "aggregate_metrics": agg_metrics,
    }


# ---------------------------------------------------------------------------
# Public API: train + evaluate
# ---------------------------------------------------------------------------

def train_water_balance_baseline(
    *, project_root: Path, output_root: Path, config: dict
) -> dict:
    """Train (calibrate) water balance model on trajectory dataset.

    Returns CV results dict.
    """
    trajectory_dir = output_root / "v42" / "trajectory_dataset"
    log.info("Loading trajectory data from %s", trajectory_dir)

    manifest = _load_trajectory_data(trajectory_dir)
    graph_schema = _load_graph_schema(trajectory_dir)
    node_schema = _load_node_schema(trajectory_dir)

    is_storage, is_outfall = _extract_node_masks(node_schema)
    log.info(
        "Node masks: %d storage, %d outfall out of %d nodes",
        is_storage.sum(),
        is_outfall.sum(),
        len(is_storage),
    )

    # Extract features for all samples
    log.info("Extracting features from %d samples...", len(manifest))
    samples = []
    for idx, row in manifest.iterrows():
        try:
            s = _extract_sample_features(row, is_storage, is_outfall)
            samples.append(s)
        except Exception as exc:
            log.warning("Failed to extract sample %d: %s", idx, exc)

    log.info("Successfully extracted %d / %d samples", len(samples), len(manifest))

    # Run event-grouped CV
    cv_results = _event_grouped_cv(samples, n_folds=5)
    cv_results["graph_schema"] = graph_schema
    cv_results["n_storage_nodes"] = int(is_storage.sum())
    cv_results["n_outfall_nodes"] = int(is_outfall.sum())

    return cv_results


def evaluate_water_balance_baseline(
    *, project_root: Path, output_root: Path, config: dict, cv_results: dict | None = None
) -> dict:
    """Evaluate water balance model — per-event detailed results.

    Uses the full dataset (train all, evaluate per-event).
    """
    trajectory_dir = output_root / "v42" / "trajectory_dataset"
    manifest = _load_trajectory_data(trajectory_dir)
    node_schema = _load_node_schema(trajectory_dir)
    is_storage, is_outfall = _extract_node_masks(node_schema)

    # Extract features
    samples = []
    for idx, row in manifest.iterrows():
        try:
            s = _extract_sample_features(row, is_storage, is_outfall)
            samples.append(s)
        except Exception:
            pass

    # Train on all data
    model = WaterBalanceBaseline()
    model.fit(samples)
    preds = model.predict(samples)

    # Per-event evaluation
    event_groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        eid = s["event_id"]
        if eid not in event_groups:
            event_groups[eid] = []
        event_groups[eid].append(i)

    per_event_results = {}
    for eid, indices in sorted(event_groups.items()):
        event_samples = [samples[i] for i in indices]
        event_preds = {t: preds[t][indices] for t in WaterBalanceBaseline.TARGET_NAMES}
        metrics = model.evaluate(event_samples, event_preds)
        per_event_results[eid] = {
            "n_samples": len(indices),
            "metrics": metrics,
        }

    # Overall evaluation
    overall_metrics = model.evaluate(samples, preds)

    return {
        "model_coefficients": {
            t: {"coefficients": model.coefficients[t], "intercept": model.intercepts[t]}
            for t in WaterBalanceBaseline.TARGET_NAMES
        },
        "feature_names": model.feature_names,
        "overall_metrics": overall_metrics,
        "per_event_results": per_event_results,
        "n_events": len(event_groups),
        "n_samples": len(samples),
    }
