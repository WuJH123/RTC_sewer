"""Materialize causal GAT-reconstructed 13-frame histories for the fast V4.2 E2E pilot.

For every selected decision state t, this script performs thirteen *real* Step1
calls at t-60, t-55, ..., t.  Each call itself uses the preceding 60 minutes of
sparse observations.  The resulting 13 reconstructed full-network depth frames
become the Step2 history.  Current-frame repetition and SWMM-truth-history
substitution are therefore impossible by construction.

The future rainfall column used by the downstream pilot is replaced with a
causal persistence/decay forecast built only from rainfall observed through t.
The realised future rainfall remains stored only as a diagnostic/label column.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.v42_fast_e2e import FAST_E2E_CONTRACT_ID, MIN_RAINFALL_GROUPS, make_causal_rainfall_forecast
from sewerrtc.v4.v42_step1_dataset import (
    _build_usecols,
    _detail_extract_window,
    _sensor_layout,
    load_graph_assets,
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _read_detail(path: Path, required: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = set(map(str, header.columns))
    missing = [c for c in required if c not in available]
    if missing:
        raise KeyError(f"GAT-history detail missing required columns: {missing[:10]}")
    frame = pd.read_csv(path, usecols=required, low_memory=False)
    return frame.loc[:, required]


def _resolve_history_detail(
    *,
    window_manifest: pd.DataFrame,
    event_id: str,
    rainfall_sha: str,
    checkpoint: float,
    candidate_path: str,
) -> Path | None:
    """Find an existing same-event forcing detail with the earlier causal prefix."""
    q = window_manifest[window_manifest["event_id"].astype(str).eq(str(event_id))].copy()
    if "rainfall_sha256" in q.columns and rainfall_sha:
        q = q[q["rainfall_sha256"].astype(str).eq(str(rainfall_sha))]
    start = float(checkpoint) - 120.0
    end = float(checkpoint) - 60.0
    q["history_start_min"] = pd.to_numeric(q["history_start_min"], errors="coerce")
    q["history_end_min"] = pd.to_numeric(q["history_end_min"], errors="coerce")
    q = q[q.history_start_min.le(start) & q.history_end_min.ge(end)]
    q = q[q.detail_path.astype(str).ne(str(candidate_path))]
    if q.empty:
        return None
    return Path(str(q.sort_values(["history_start_min", "detail_path"]).iloc[0]["detail_path"]))


def _reconstruct_state_history(
    *,
    detail: pd.DataFrame,
    checkpoint_min: float,
    model: TemporalSparseGATReconstructorV42,
    graph,
    sensor_mask_1d: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = [checkpoint_min - 60.0 + 5.0 * i for i in range(13)]
    extracted = []
    for anchor in anchors:
        item = _detail_extract_window(detail, anchor, graph.node_ids, graph.facility_ids)
        if item is None:
            raise ValueError(f"insufficient causal warm-up for Step1 at anchor={anchor:.6f}")
        extracted.append(item)

    mask_history = np.broadcast_to(
        sensor_mask_1d[None, :], (13, graph.n_nodes)
    ).astype(np.float32, copy=True)
    sparse_batch = np.stack(
        [item["depth_history"] * mask_history for item in extracted], axis=0
    ).astype(np.float32)
    mask_batch = np.broadcast_to(
        mask_history[None, :, :], (13, 13, graph.n_nodes)
    ).copy().astype(np.float32)
    rain_batch = np.stack([item["rainfall"] for item in extracted], axis=0).astype(np.float32)
    action_batch = np.stack([item["actions"] for item in extracted], axis=0).astype(np.float32)

    with torch.no_grad():
        out = model(
            sparse_depth_history=torch.from_numpy(sparse_batch).to(device),
            sensor_mask_history=torch.from_numpy(mask_batch).to(device),
            rainfall_history=torch.from_numpy(rain_batch).to(device),
            historical_actions=torch.from_numpy(action_batch).to(device),
            node_static=torch.from_numpy(graph.node_static).to(device),
            link_static=torch.from_numpy(graph.link_static).to(device),
            edge_index=torch.from_numpy(graph.edge_index).to(device),
            action_node_map=torch.from_numpy(graph.action_node_map).to(device),
        )
    history = out.depth_mean.detach().cpu().numpy().astype(np.float32)
    std = out.depth_std.detach().cpu().numpy().astype(np.float32)
    observed_rain_history = extracted[-1]["rainfall"].astype(np.float32)
    if history.shape != (13, graph.n_nodes):
        raise RuntimeError(f"unexpected reconstructed history shape: {history.shape}")
    return history, std, observed_rain_history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--step1-model-dir", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--audit-output", type=Path, required=True)
    ap.add_argument("--sensor-ratio", type=float, default=0.10)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=3)
    ap.add_argument("--min-rainfall-groups", type=int, default=MIN_RAINFALL_GROUPS)
    ap.add_argument(
        "--step1-window-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet",
    )
    args = ap.parse_args()

    frame = _read(args.input_manifest)
    if frame.empty:
        raise ValueError("fast E2E Step2 input manifest is empty")
    required = {
        "state_key",
        "split_group_key",
        "checkpoint_min",
        "source_detail_path_candidate",
        "history_depth",
        "rainfall_forecast",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"fast E2E manifest missing fields: {sorted(missing)}")
    window_manifest = _read(args.step1_window_manifest)
    window_required = {"event_id", "detail_path", "history_start_min", "history_end_min"}
    if not window_required.issubset(window_manifest.columns):
        raise KeyError(f"Step1 window manifest missing fields: {sorted(window_required - set(window_manifest.columns))}")

    graph = load_graph_assets(args.project_root)
    sensor_mask, sensor_indices, sensor_sha = _sensor_layout(
        graph.n_nodes, args.sensor_ratio, args.sensor_layout_seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        gat_layers=args.gat_layers,
    ).to(device)
    model_path = args.step1_model_dir / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    required_cols = _build_usecols(graph.node_ids, graph.facility_ids)
    detail_cache: dict[str, pd.DataFrame] = {}
    output_rows: list[pd.Series] = []
    failures: list[dict[str, str]] = []

    for state_key, group in frame.groupby("state_key", sort=True):
        first = group.iloc[0]
        try:
            checkpoint = float(first["checkpoint_min"])
            candidate_path = str(first["source_detail_path_candidate"])
            rainfall_sha = str(first.get("rainfall_sha256", first.get("split_group_key", "")))
            history_path = _resolve_history_detail(
                window_manifest=window_manifest,
                event_id=str(first.get("event_id", "")),
                rainfall_sha=rainfall_sha,
                checkpoint=checkpoint,
                candidate_path=candidate_path,
            )
            if history_path is None or not history_path.exists():
                raise FileNotFoundError("no same-state history detail covers checkpoint-120..checkpoint-60")
            history_key = str(history_path.resolve())
            if history_key not in detail_cache:
                detail_cache[history_key] = _read_detail(history_path, required_cols)
            history, std, observed_rain = _reconstruct_state_history(
                detail=detail_cache[history_key],
                checkpoint_min=checkpoint,
                model=model,
                graph=graph,
                sensor_mask_1d=sensor_mask,
                device=device,
            )
            causal_forecast = make_causal_rainfall_forecast(observed_rain)
            for _, row in group.iterrows():
                rec = row.copy()
                rec["history_depth_swmm_truth_diagnostic"] = rec["history_depth"]
                rec["history_depth"] = json.dumps(history.tolist(), allow_nan=False)
                rec["gat_depth_std_history_mean"] = float(std.mean())
                rec["gat_depth_std_current_mean"] = float(std[-1].mean())
                rec["rainfall_realized_future_diagnostic"] = rec["rainfall_forecast"]
                rec["rainfall_forecast"] = json.dumps(causal_forecast.tolist(), allow_nan=False)
                rec["state_source"] = "gat_sparse_reconstruction"
                rec["reconstructed_history_contract"] = "13_real_step1_calls_at_5min_spacing"
                rec["current_frame_repetition_used"] = False
                rec["authoritative_swmm_history_used_as_online_input"] = False
                rec["rainfall_input_authority"] = "causal_persistence_decay_from_observed_history"
                rec["realized_future_rainfall_used_online"] = False
                rec["sensor_layout_sha256"] = sensor_sha
                rec["fast_e2e_contract_id"] = FAST_E2E_CONTRACT_ID
                output_rows.append(rec)
        except Exception as exc:
            candidate_path = str(first.get("source_detail_path_candidate", ""))
            elapsed_min_min = None
            elapsed_min_max = None
            try:
                h = pd.read_csv(candidate_path, usecols=["elapsed_min"])
                elapsed = pd.to_numeric(h["elapsed_min"], errors="coerce").dropna()
                elapsed_min_min = float(elapsed.min()) if not elapsed.empty else None
                elapsed_min_max = float(elapsed.max()) if not elapsed.empty else None
            except Exception:
                pass
            failures.append({
                "state_key": str(state_key),
                "checkpoint": str(first.get("checkpoint_min", "")),
                "detail_path": candidate_path,
                "elapsed_min_min": str(elapsed_min_min),
                "elapsed_min_max": str(elapsed_min_max),
                "failed_anchor": str(float(first.get("checkpoint_min", 0.0)) - 60.0),
                "required_history_start": str(float(first.get("checkpoint_min", 0.0)) - 120.0),
                "exception": f"{type(exc).__name__}: {exc}",
            })

    out = pd.DataFrame(output_rows)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    early_audit = {
        "contract_id": FAST_E2E_CONTRACT_ID,
        "stage": "fast_e2e_gat_causal_history",
        "development_only": True,
        "input_rows": int(len(frame)),
        "input_states": int(frame["state_key"].nunique()),
        "output_rows": int(len(out)),
        "output_states": int(out["state_key"].nunique()) if not out.empty else 0,
        "failed_states": int(len(failures)),
        "failure_examples": failures[:200],
        "state_source": "gat_sparse_reconstruction",
        "reconstructed_history_contract": "13_real_step1_calls_at_5min_spacing",
        "current_frame_repetition_used": False,
        "authoritative_swmm_history_used_as_online_input": False,
        "realized_future_rainfall_used_online": False,
    }
    args.audit_output.write_text(json.dumps(early_audit, indent=2, allow_nan=False), encoding="utf-8")
    if out.empty:
        raise RuntimeError("no fast E2E state could build a causal GAT history")
    groups = int(out["split_group_key"].astype(str).nunique())
    if groups < int(args.min_rainfall_groups):
        raise RuntimeError(
            f"causal GAT-history materialization retained only {groups} rainfall groups; "
            f"minimum is {args.min_rainfall_groups}. Choose later checkpoints or a larger source pool."
        )
    state_counts = out.groupby("state_key").size()
    if int(state_counts.min()) < 2:
        raise RuntimeError("GAT-history materialization lost candidate multiplicity within a state")
    _write(out, args.output_manifest)

    audit = {
        "contract_id": FAST_E2E_CONTRACT_ID,
        "stage": "fast_e2e_gat_causal_history",
        "development_only": True,
        "formal_mainline_authorized": False,
        "input_rows": int(len(frame)),
        "output_rows": int(len(out)),
        "input_states": int(frame["state_key"].nunique()),
        "output_states": int(out["state_key"].nunique()),
        "output_rainfall_groups": groups,
        "minimum_required_rainfall_groups": int(args.min_rainfall_groups),
        "sensor_ratio": float(args.sensor_ratio),
        "sensor_count": int(len(sensor_indices)),
        "sensor_layout_sha256": sensor_sha,
        "state_source": "gat_sparse_reconstruction",
        "reconstructed_history_contract": "13_real_step1_calls_at_5min_spacing",
        "current_frame_repetition_used": False,
        "authoritative_swmm_history_used_as_online_input": False,
        "rainfall_input_authority": "causal_persistence_decay_from_observed_history",
        "realized_future_rainfall_used_online": False,
        "future_SWMM_trajectories_remain_supervision_only": True,
        "failed_states": int(len(failures)),
        "failure_examples": failures[:50],
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
