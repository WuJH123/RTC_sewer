"""Streaming bounded partial Step-2 fine-tune; no SWMM and no full wide load."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v42_step2_head_only_repair import _metrics, _pair_loss
from scripts.train_v42_step2_fast import _forward, _graph_indices, _tensorise
from scripts.train_v42_step2_formal_f2 import _rank
from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, build_surrogate_action_node_map


BRANCHES = ("candidate", "no_control", "dynamic_internal", "hold_previous")
ARRAY_COLUMNS = [
    "history_depth", "history_actions_readback", "rainfall_forecast",
    "action_candidate_readback", "action_no_control_readback", "action_dynamic_internal_readback",
    "action_hold_previous_readback", "pfv_delta", "tfv_delta", "peak_delta",
    "source_detail_path_dynamic_internal", "checkpoint_min", "state_key", "split_group_key",
]
for _branch in BRANCHES:
    ARRAY_COLUMNS.extend([f"trajectory_depth_{_branch}", f"trajectory_flood_{_branch}"])


def _arr(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _causal_action_cache(meta: pd.DataFrame) -> dict[tuple[str, float], str]:
    result: dict[tuple[str, float], str] = {}
    for raw_path, raw_cp in meta[["source_detail_path_dynamic_internal", "checkpoint_min"]].drop_duplicates().itertuples(index=False):
        path = Path(str(raw_path)); cp = float(raw_cp)
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            header = [str(x).strip() for x in next(csv.reader(handle))]
        settings = [column for column in header if column.casefold().startswith("setting:")]
        if len(settings) != 36:
            raise RuntimeError(f"{path}: expected 36 setting columns, got {len(settings)}")
        detail = pd.read_csv(path, usecols=["elapsed_min", *settings])
        elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)
        indices = np.flatnonzero(np.isclose(elapsed, cp, atol=1.0e-8, rtol=0.0))
        if len(indices) != 1:
            raise RuntimeError(f"{path}: checkpoint {cp:g} has {len(indices)} rows")
        current = detail.iloc[int(indices[0])][settings].to_numpy(np.float32)
        if not np.isfinite(current).all():
            raise RuntimeError(f"{path}: non-finite causal Internal readback")
        result[(str(path.resolve()), cp)] = json.dumps(np.repeat(current[None, :], 12, axis=0).tolist(), separators=(",", ":"))
    return result


def _metadata(args: argparse.Namespace) -> pd.DataFrame:
    columns = ["state_key", "split_group_key", "action_candidate_readback", "source_detail_path_dynamic_internal", "checkpoint_min"]
    meta = pd.read_parquet(args.manifest, columns=columns)
    meta["action_key"] = meta["action_candidate_readback"].map(lambda value: action_sha256(_arr(value)))
    bank = pd.read_parquet(args.experience_bank, columns=["state_key", "candidate_action_sha256", "canonical_candidate_action_sha256", "pfv_budget_metric_m3", "tfv_candidate_m3", "tfv_internal_m3"])
    bank["action_key"] = bank["canonical_candidate_action_sha256"].fillna(bank["candidate_action_sha256"]).astype(str)
    lookup = bank.drop_duplicates(["state_key", "action_key"], keep="last").set_index(["state_key", "action_key"])
    keys = pd.MultiIndex.from_arrays([meta.state_key.astype(str), meta.action_key.astype(str)])
    labels = lookup.reindex(keys)
    if labels.pfv_budget_metric_m3.isna().any():
        raise RuntimeError("experience-bank label alignment failed")
    meta["pfv_budget_metric_m3"] = labels.pfv_budget_metric_m3.to_numpy(float)
    meta["tfv_delta_m3"] = (labels.tfv_candidate_m3 - labels.tfv_internal_m3).to_numpy(float)
    groups = _rank(sorted(meta.split_group_key.astype(str).unique()), 42)
    meta["split"] = np.where(meta.split_group_key.isin(groups[16:]), "train", np.where(meta.split_group_key.isin(groups[:8]), "validation", "holdout"))
    if tuple(meta.loc[meta.split.eq("train"), "split_group_key"].nunique() for _ in [0]) != (65,):
        raise RuntimeError("unexpected train group count")
    return meta


def _trajectory_loss(pred: dict, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    terms = []
    for branch in BRANCHES:
        terms.extend([
            nn.functional.smooth_l1_loss(pred["branches"][branch]["node_depth"], batch[f"depth_{branch}"].to(device)),
            nn.functional.smooth_l1_loss(pred["branches"][branch]["node_flooding_rate"], batch[f"flood_{branch}"].to(device)),
        ])
    return torch.stack(terms).mean()


def _iter_batches(args: argparse.Namespace, meta: pd.DataFrame, causal: dict[tuple[str, float], str], split: str | None):
    parquet = pq.ParquetFile(args.manifest)
    offset = 0
    for record in parquet.iter_batches(batch_size=int(args.batch_size), columns=ARRAY_COLUMNS, use_threads=True):
        frame = record.to_pandas()
        indices = np.arange(offset, offset + len(frame), dtype=int)
        offset += len(frame)
        if split is not None:
            keep = meta.iloc[indices].split.eq(split).to_numpy()
            if not keep.any():
                continue
            frame = frame.iloc[np.flatnonzero(keep)].reset_index(drop=True)
            indices = indices[keep]
        frame["action_dynamic_internal_readback"] = [
            causal[(str(path.resolve()), float(cp))]
            for path, cp in zip(frame.source_detail_path_dynamic_internal.map(Path), frame.checkpoint_min)
        ]
        yield indices, frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--experience-bank", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    meta = _metadata(args); causal = _causal_action_cache(meta)
    groups = _rank(sorted(meta.split_group_key.astype(str).unique()), 42)
    train_groups, validation_groups, holdout_groups = groups[16:], groups[:8], groups[8:16]
    graph = _load_graph_topology(args.project_root); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_tensors = (
        torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device), torch.from_numpy(graph["node_static"].astype(np.float32)).to(device),
        torch.from_numpy(build_surrogate_action_node_map(graph).astype(np.float32)).to(device), _graph_indices(graph, "is_storage", device), _graph_indices(graph, "is_outfall", device),
    )
    priority = torch.as_tensor(get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device)
    model = MultiReferenceHydraulicSurrogate(n_nodes=int(graph["n_nodes"]), n_facilities=int(graph["n_facilities"]), state_feature_dim=1, static_feature_dim=int(graph["node_static"].shape[1]), hidden_dim=64, gat_heads=4, gat_layers=3, horizon=12).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for module_name in ("action_encoder", "rollout_gat", "rollout_norm", "rollout_skip", "dynamics", "depth_head", "flood_head", "action_effect_flood_head"):
        for parameter in getattr(model, module_name).parameters(): parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2.0e-4, weight_decay=1.0e-4)
    scales = torch.as_tensor([max(float(np.median(np.abs(meta.pfv_budget_metric_m3))), 1.0), max(float(np.median(np.abs(meta.tfv_delta_m3))), 1.0)], dtype=torch.float32, device=device)
    state_codes = {state: i for i, state in enumerate(sorted(meta.state_key.astype(str).unique()))}
    state_ids = torch.as_tensor([state_codes[str(value)] for value in meta.state_key], dtype=torch.long)
    best_key = None; best_state = None; stale = 0; history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train(); losses = []
        for indices, frame in _iter_batches(args, meta, causal, "train"):
            batch = {key: value.to(device) for key, value in _tensorise(frame).items()}
            pred = _forward(model, batch, graph_tensors, priority, device)
            predicted = torch.stack([pred["pfv_delta"] - 0.05 * pred["kpi_no_control"]["pfv_m3"], pred["tfv_delta"]], dim=1)
            target = torch.as_tensor(meta.iloc[indices][["pfv_budget_metric_m3", "tfv_delta_m3"]].to_numpy(np.float32), device=device)
            rank, direction = _pair_loss(predicted, target, state_ids.index_select(0, torch.as_tensor(indices)).to(device))
            loss = nn.functional.smooth_l1_loss(predicted / scales, target / scales) + 0.5 * rank + 0.5 * direction + 0.1 * _trajectory_loss(pred, batch, device)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 2.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); prediction_rows = []; label_rows = []
        with torch.inference_mode():
            for indices, frame in _iter_batches(args, meta, causal, "validation"):
                batch = {key: value.to(device) for key, value in _tensorise(frame).items()}; pred = _forward(model, batch, graph_tensors, priority, device)
                prediction_rows.append(torch.stack([pred["pfv_delta"] - 0.05 * pred["kpi_no_control"]["pfv_m3"], pred["tfv_delta"]], dim=1).cpu().numpy()); label_rows.append(meta.iloc[indices])
        val_labels = pd.concat(label_rows, ignore_index=True); val_pred = np.concatenate(prediction_rows); val_report = _metrics(torch.empty(0), val_labels, val_pred)
        key_values = [val_report.get("median_pfv_spearman"), val_report.get("median_tfv_spearman"), val_report.get("median_pfv_pairwise"), val_report.get("median_tfv_pairwise")]; key = tuple(-float(value) if value is not None else 1.0 for value in key_values)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": val_report}); print(json.dumps(history[-1], allow_nan=False), flush=True)
        if best_key is None or key < best_key:
            best_key = key; best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= int(args.patience): break
    if best_state is None: raise RuntimeError("no partial checkpoint produced")
    model.load_state_dict(best_state); model.eval()
    def evaluate(split: str) -> dict[str, object]:
        prediction_rows = []; label_rows = []
        with torch.inference_mode():
            for indices, frame in _iter_batches(args, meta, causal, split):
                batch = {key: value.to(device) for key, value in _tensorise(frame).items()}; pred = _forward(model, batch, graph_tensors, priority, device)
                prediction_rows.append(torch.stack([pred["pfv_delta"] - 0.05 * pred["kpi_no_control"]["pfv_m3"], pred["tfv_delta"]], dim=1).cpu().numpy()); label_rows.append(meta.iloc[indices])
        return _metrics(torch.empty(0), pd.concat(label_rows, ignore_index=True), np.concatenate(prediction_rows))
    model_path = out / "PARTIAL_FINE_TUNE_V1_seed42.pt"; torch.save(model.state_dict(), model_path)
    report = {
        "audit_id": "STEP2_PARTIAL_FINE_TUNE_V1_SEED42_STREAMING", "status": "development_only", "new_swmm_started": False,
        "calibration_groups_used_for_training": [], "train_groups": train_groups, "validation_groups": validation_groups, "holdout_groups": holdout_groups,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source_model": str(args.model.resolve()), "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "batch_size": int(args.batch_size), "max_epochs": int(args.epochs), "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable)),
        "frozen_modules": ["history_encoder", "rain_encoder", "state_to_hidden"], "updated_modules": ["action_encoder", "rollout_gat", "rollout_norm", "rollout_skip", "dynamics", "depth_head", "flood_head", "action_effect_flood_head"],
        "history": history, "train": evaluate("train"), "validation": evaluate("validation"), "holdout": evaluate("holdout"),
        "model_path": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "development_gate": {"local_direction": 0.70, "pairwise": 0.75, "spearman": 0.60, "top5_recall": 0.70},
    }
    (out / "STEP2_PARTIAL_FINE_TUNE_SEED42_REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out / "STEP2_PARTIAL_FINE_TUNE_SEED42_REPORT.json"), "device": str(device), "epochs": len(history)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
