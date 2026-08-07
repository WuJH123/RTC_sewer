"""Bounded seed42 comparison for the candidate-relative Step-2 V2 repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.models.candidate_relative_differentiable_control_v2 import (
    CandidateRelativeDifferentiableControlSurrogateV2,
)


def _arr(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2 or np.allclose(actual, actual[0]) or np.allclose(predicted, predicted[0]):
        return None
    value = float(spearmanr(actual, predicted).statistic)
    return value if np.isfinite(value) else None


def _pair_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2:
        return None
    a = actual[:, None] - actual[None, :]
    p = predicted[:, None] - predicted[None, :]
    mask = np.triu(np.ones_like(a, dtype=bool), 1) & (np.abs(a) > 1.0e-7)
    if not mask.any():
        return None
    return float((np.sign(a[mask]) == np.sign(p[mask])).mean())


def _decode_frame(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    state = np.stack([_arr(x) for x in frame.state_signature_json]).astype(np.float32)
    action = np.stack([_arr(x) for x in frame.candidate_action_json]).astype(np.float32)
    if action.shape[1:] != (12, 36):
        raise ValueError(f"unexpected action shape {action.shape}")
    current = action[:, 3, :].copy()
    no_control = np.ones_like(current, dtype=np.float32)
    internal = current.copy()
    aux = np.stack([_arr(x) for x in frame.trajectory_aux_json]).astype(np.float32)
    aux_mask = np.stack([np.asarray(json.loads(str(x)), dtype=bool) for x in frame.trajectory_aux_mask_json])
    return {
        "state": state,
        "action": action,
        "current": current,
        "no_control": no_control,
        "internal": internal,
        "g": frame.g_pfv.to_numpy(np.float32),
        "tfv": frame.delta_tfv.to_numpy(np.float32),
        "aux": aux,
        "aux_mask": aux_mask,
        "row_id": frame.row_id.to_numpy(np.int64),
    }


def _normalisation(frame: pd.DataFrame, train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    state_center = np.median(train["state"], axis=0).astype(np.float32)
    state_scale = np.maximum(np.percentile(train["state"], 75, axis=0) - np.percentile(train["state"], 25, axis=0), 1.0e-5).astype(np.float32)
    g_scale = max(float(np.median(np.abs(train["g"]))), 1.0)
    tfv_scale = max(float(np.median(np.abs(train["tfv"]))), 1.0)
    aux_scale = np.ones(4, dtype=np.float32)
    for j in range(4):
        values = train["aux"][train["aux_mask"][:, j], :, j]
        if values.size:
            aux_scale[j] = max(float(np.median(np.abs(values))), 1.0e-4)
    return {"state_center": state_center, "state_scale": state_scale, "g_scale": np.float32(g_scale), "tfv_scale": np.float32(tfv_scale), "aux_scale": aux_scale}


def _to_tensors(data: dict[str, np.ndarray], norm: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state": torch.as_tensor((data["state"] - norm["state_center"]) / norm["state_scale"], device=device),
        "action": torch.as_tensor(data["action"], device=device),
        "current": torch.as_tensor(data["current"], device=device),
        "no_control": torch.as_tensor(data["no_control"], device=device),
        "internal": torch.as_tensor(data["internal"], device=device),
        "g": torch.as_tensor(data["g"] / norm["g_scale"], device=device),
        "tfv": torch.as_tensor(data["tfv"] / norm["tfv_scale"], device=device),
        "aux": torch.as_tensor(data["aux"] / norm["aux_scale"][None, None, :], device=device),
        "aux_mask": torch.as_tensor(data["aux_mask"], device=device),
    }


def _forward(model: nn.Module, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(tensors["state"], tensors["action"], tensors["current"], tensors["no_control"], tensors["internal"])


def _pair_loss(pred: torch.Tensor, target: torch.Tensor, row_i: torch.Tensor, row_j: torch.Tensor) -> torch.Tensor:
    if row_i.numel() == 0:
        return pred.sum() * 0.0
    pdiff = pred.index_select(0, row_j) - pred.index_select(0, row_i)
    tdiff = target.index_select(0, row_j) - target.index_select(0, row_i)
    informative = tdiff.abs() > 1.0e-6
    if not bool(informative.any()):
        return pred.sum() * 0.0
    return nn.functional.softplus(-tdiff[informative].sign() * pdiff[informative]).mean()


def _batch_loss(model: nn.Module, tensors: dict[str, torch.Tensor], *, pairs: pd.DataFrame, local_pairs: pd.DataFrame, row_to_pos: dict[int, int], variant: str, norm: dict[str, np.ndarray]) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = _forward(model, tensors)
    loss_g = nn.functional.smooth_l1_loss(prediction["mean_g_pfv"], tensors["g"])
    loss_t = nn.functional.smooth_l1_loss(prediction["delta_tfv"], tensors["tfv"])
    loss = loss_g + loss_t
    terms = {"pfv": float(loss_g.detach()), "tfv": float(loss_t.detach()), "pair": 0.0, "direction": 0.0, "trajectory": 0.0}
    if variant != "baseline" and not pairs.empty:
        pi = torch.as_tensor([row_to_pos[int(x)] for x in pairs.row_i], dtype=torch.long, device=tensors["state"].device)
        pj = torch.as_tensor([row_to_pos[int(x)] for x in pairs.row_j], dtype=torch.long, device=tensors["state"].device)
        rank = 0.5 * (_pair_loss(prediction["mean_g_pfv"], tensors["g"], pi, pj) + _pair_loss(prediction["delta_tfv"], tensors["tfv"], pi, pj))
        loss = loss + 0.5 * rank
        terms["pair"] = float(rank.detach())
    if variant == "gradient" and not local_pairs.empty:
        selected = local_pairs.sample(n=min(16, len(local_pairs)), random_state=42)
        pi = torch.as_tensor([row_to_pos[int(x)] for x in selected.row_i], dtype=torch.long, device=tensors["state"].device)
        pj = torch.as_tensor([row_to_pos[int(x)] for x in selected.row_j], dtype=torch.long, device=tensors["state"].device)
        action = tensors["action"].index_select(0, pi).detach().clone().requires_grad_(True)
        state = tensors["state"].index_select(0, pi)
        current = tensors["current"].index_select(0, pi)
        no_control = tensors["no_control"].index_select(0, pi)
        internal = tensors["internal"].index_select(0, pi)
        with torch.backends.cudnn.flags(enabled=False):
            tfv = model(state, action, current, no_control, internal)["delta_tfv"]
            grad = torch.autograd.grad(tfv.sum(), action, create_graph=True)[0]
        delta_action = tensors["action"].index_select(0, pj) - tensors["action"].index_select(0, pi)
        directional_prediction = (grad * delta_action).sum(dim=(1, 2))
        actual = tensors["tfv"].index_select(0, pj) - tensors["tfv"].index_select(0, pi)
        informative = actual.abs() > 1.0e-6
        if bool(informative.any()):
            direction = nn.functional.softplus(-actual[informative].sign() * directional_prediction[informative]).mean()
            loss = loss + 0.25 * direction
            terms["direction"] = float(direction.detach())
    aux_pred = prediction["trajectory_residual"]
    mask = tensors["aux_mask"][:, None, :].expand_as(aux_pred)
    if bool(mask.any()):
        trajectory = nn.functional.smooth_l1_loss(aux_pred[mask], tensors["aux"][mask])
        loss = loss + 0.05 * trajectory
        terms["trajectory"] = float(trajectory.detach())
    return loss, terms


def _predict(model: nn.Module, tensors: dict[str, torch.Tensor], norm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        out = _forward(model, tensors)
    return {
        "g": out["mean_g_pfv"].detach().cpu().numpy() * float(norm["g_scale"]),
        "tfv": out["delta_tfv"].detach().cpu().numpy() * float(norm["tfv_scale"]),
        "aux": out["trajectory_residual"].detach().cpu().numpy() * norm["aux_scale"][None, None, :],
    }


def _gradient_direction_metrics(model: nn.Module, tensors: dict[str, torch.Tensor], frame: pd.DataFrame, local_pairs: pd.DataFrame, norm: dict[str, np.ndarray], max_pairs: int = 128) -> dict[str, float | None]:
    if local_pairs.empty:
        return {"tfv": None, "pfv": None, "count": 0}
    work = local_pairs.sample(n=min(max_pairs, len(local_pairs)), random_state=123).reset_index(drop=True)
    was_training = model.training
    # cuDNN RNNs do not expose backward in eval mode.  This diagnostic needs a
    # parameter gradient only with respect to the action input; model weights
    # remain untouched and dropout is not used by V2.
    model.train()
    row_to_pos = {int(row): i for i, row in enumerate(frame.row_id.to_numpy(np.int64))}
    actual_tfv = []
    actual_pfv = []
    predicted_tfv = []
    predicted_pfv = []
    for row in work.itertuples(index=False):
        i = row_to_pos.get(int(row.row_i)); j = row_to_pos.get(int(row.row_j))
        if i is None or j is None:
            continue
        action = tensors["action"][i:i + 1].detach().clone().requires_grad_(True)
        pred = model(tensors["state"][i:i + 1], action, tensors["current"][i:i + 1], tensors["no_control"][i:i + 1], tensors["internal"][i:i + 1])
        grad_t = torch.autograd.grad(pred["delta_tfv"].sum(), action, retain_graph=True)[0]
        grad_g = torch.autograd.grad(pred["mean_g_pfv"].sum(), action)[0]
        delta = tensors["action"][j:j + 1] - tensors["action"][i:i + 1]
        predicted_tfv.append(float((grad_t * delta).sum().detach().cpu()) * float(norm["tfv_scale"]))
        predicted_pfv.append(float((grad_g * delta).sum().detach().cpu()) * float(norm["g_scale"]))
        actual_tfv.append(float(frame.iloc[j].delta_tfv - frame.iloc[i].delta_tfv))
        actual_pfv.append(float(frame.iloc[j].g_pfv - frame.iloc[i].g_pfv))
    result = {
        "tfv": float((np.sign(actual_tfv) == np.sign(predicted_tfv)).mean()) if actual_tfv else None,
        "pfv": float((np.sign(actual_pfv) == np.sign(predicted_pfv)).mean()) if actual_pfv else None,
        "count": int(len(actual_tfv)),
    }
    model.train(was_training)
    return result


def _decision_metrics(frame: pd.DataFrame, prediction: dict[str, np.ndarray]) -> dict[str, object]:
    work = frame.reset_index(drop=True)
    predicted_safe = prediction["g"] <= 100.0
    actual_safe = work.g_pfv.to_numpy(float) <= 100.0
    selected = []
    regrets = []
    captures = []
    for state, positions in work.groupby("state_key", sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        safe_idx = idx[actual_safe[idx]]
        predicted_idx = idx[predicted_safe[idx]]
        if not len(safe_idx):
            continue
        best = float(work.iloc[safe_idx].delta_tfv.min())
        if len(predicted_idx):
            chosen = int(predicted_idx[np.argmin(prediction["tfv"][predicted_idx])])
            selected.append(chosen)
            regrets.append(float(work.iloc[chosen].delta_tfv - best))
            denominator = float(work.iloc[safe_idx].delta_tfv.max() - best)
            numerator = float(work.iloc[chosen].delta_tfv - best)
            captures.append(1.0 - numerator / denominator if denominator > 1.0e-6 else 1.0)
    false_safe = predicted_safe & ~actual_safe
    return {
        "predicted_safe_count": int(predicted_safe.sum()),
        "actual_safe_count": int(actual_safe.sum()),
        "false_safe_fraction": float(false_safe.sum() / max(int(predicted_safe.sum()), 1)),
        "selected_state_count": int(len(selected)),
        "selected_actual_safe_rate": float(np.mean(actual_safe[selected])) if selected else None,
        "empty_safe_set_rate": float(1.0 - len(selected) / max(work.state_key.nunique(), 1)),
        "selection_regret_mean_m3": float(np.mean(regrets)) if regrets else None,
        "selection_regret_median_m3": float(np.median(regrets)) if regrets else None,
        "oracle_capture_mean": float(np.mean(captures)) if captures else None,
    }


def _metrics(model: nn.Module, frame: pd.DataFrame, tensors: dict[str, torch.Tensor], pairs: pd.DataFrame, local_pairs: pd.DataFrame, norm: dict[str, np.ndarray]) -> dict[str, object]:
    prediction = _predict(model, tensors, norm)
    rows = []
    for state, positions in frame.groupby("state_key", sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        rows.append({
            "state_key": str(state),
            "pfv_spearman": _safe_spearman(frame.iloc[idx].g_pfv.to_numpy(float), prediction["g"][idx]),
            "tfv_spearman": _safe_spearman(frame.iloc[idx].delta_tfv.to_numpy(float), prediction["tfv"][idx]),
            "pfv_pairwise": _pair_accuracy(frame.iloc[idx].g_pfv.to_numpy(float), prediction["g"][idx]),
            "tfv_pairwise": _pair_accuracy(frame.iloc[idx].delta_tfv.to_numpy(float), prediction["tfv"][idx]),
        })
    def median(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row[key] is not None]
        return float(np.median(values)) if values else None
    pair_metrics = _gradient_direction_metrics(model, tensors, frame, local_pairs, norm)
    decision = _decision_metrics(frame, prediction)
    aux_rows = []
    aux_mask = tensors["aux_mask"].detach().cpu().numpy()
    aux_pred = prediction["aux"]
    aux_true = tensors["aux"].detach().cpu().numpy() * norm["aux_scale"][None, None, :]
    names = ("depth_mean", "flood_total", "storage_mean", "facility_flow_mean")
    for j, name in enumerate(names):
        valid = aux_mask[:, j][:, None].repeat(12, axis=1)
        if valid.any():
            error = aux_pred[:, :, j][valid] - aux_true[:, :, j][valid]
            aux_rows.append({"metric": name, "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "count": int(error.size)})
    return {
        "rows": int(len(frame)),
        "states": int(frame.state_key.nunique()),
        "median_pfv_spearman": median("pfv_spearman"),
        "median_tfv_spearman": median("tfv_spearman"),
        "median_pfv_pairwise": median("pfv_pairwise"),
        "median_tfv_pairwise": median("tfv_pairwise"),
        "gradient_direction": pair_metrics,
        "decision_quality": decision,
        "trajectory_auxiliary": aux_rows,
    }


def _train_variant(name: str, frame: pd.DataFrame, train: dict[str, np.ndarray], validation: dict[str, np.ndarray], holdout: dict[str, np.ndarray], pairs: pd.DataFrame, local_pairs: pd.DataFrame, norm: dict[str, np.ndarray], output_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    torch.manual_seed(42); np.random.seed(42)
    model = CandidateRelativeDifferentiableControlSurrogateV2(raw_action_baseline=name == "baseline").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    train_t = _to_tensors(train, norm, device)
    val_t = _to_tensors(validation, norm, device)
    hold_t = _to_tensors(holdout, norm, device)
    row_to_pos = {int(row): i for i, row in enumerate(train["row_id"])}
    train_pairs = pairs[pairs.row_i.isin(row_to_pos) & pairs.row_j.isin(row_to_pos)].reset_index(drop=True)
    train_local = local_pairs[local_pairs.row_i.isin(row_to_pos) & local_pairs.row_j.isin(row_to_pos)].reset_index(drop=True)
    history = []
    initial_terms = None
    initial_grad_norm = None
    best_state = None; best_key = None; stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = np.random.RandomState(42 + epoch).permutation(len(train["row_id"]))
        losses = []
        for start in range(0, len(order), int(args.batch_size)):
            idx = torch.as_tensor(order[start:start + int(args.batch_size)], dtype=torch.long, device=device)
            batch = {key: value.index_select(0, idx) for key, value in train_t.items()}
            positions = {int(train["row_id"][int(pos)]): int(k) for k, pos in enumerate(order[start:start + int(args.batch_size)])}
            batch_pairs = train_pairs[train_pairs.row_i.isin(positions) & train_pairs.row_j.isin(positions)]
            batch_local = train_local[train_local.row_i.isin(positions) & train_local.row_j.isin(positions)]
            remapped = positions
            optimizer.zero_grad(set_to_none=True)
            loss, terms = _batch_loss(model, batch, pairs=batch_pairs, local_pairs=batch_local, row_to_pos=remapped, variant=name, norm=norm)
            loss.backward()
            if initial_terms is None:
                initial_terms = terms
                initial_grad_norm = float(
                    math.sqrt(
                        sum(
                            float(parameter.grad.detach().pow(2).sum())
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        )
                    )
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = _metrics(model, frame[frame.split.eq("validation")].reset_index(drop=True), val_t, pairs, local_pairs, norm)
        key = (
            -(val_metrics["median_tfv_pairwise"] or 0.0),
            -(val_metrics["median_pfv_pairwise"] or 0.0),
            val_metrics["decision_quality"]["selection_regret_median_m3"] or 1.0e30,
        )
        if best_key is None or key < best_key:
            best_key = key; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": val_metrics})
        if stale >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    model_path = output_dir / f"{name}_seed42.pt"
    torch.save(model.state_dict(), model_path)
    report = {
        "variant": name,
        "seed": 42,
        "device": str(device),
        "runtime_s": float(elapsed),
        "epochs_completed": len(history),
        "batch_size": int(args.batch_size),
        "model_path": str(model_path),
        "history": history,
        "loss_scale_audit": {
            "raw_terms_first_batch": initial_terms or {},
            "raw_total_first_batch": float(sum((initial_terms or {}).values())),
            "effective_contribution_first_batch": {
                key: float(value / max(sum((initial_terms or {}).values()), 1.0e-12))
                for key, value in (initial_terms or {}).items()
            },
            "gradient_norm_first_batch": initial_grad_norm,
        },
        "validation": _metrics(model, frame[frame.split.eq("validation")].reset_index(drop=True), val_t, pairs, local_pairs, norm),
        "holdout": _metrics(model, frame[frame.split.eq("holdout")].reset_index(drop=True), hold_t, pairs, local_pairs, norm),
        "train": _metrics(model, frame[frame.split.eq("train")].reset_index(drop=True), train_t, pairs, local_pairs, norm),
    }
    (output_dir / f"{name}_seed42_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.0e-3)
    args = ap.parse_args()
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.dataset_dir / "STATE_ACTION_GROUP_DATASET.parquet")
    pairs = pd.read_parquet(args.dataset_dir / "SAME_STATE_ACTION_PAIRS.parquet")
    local_pairs = pd.read_parquet(args.dataset_dir / "LOCAL_ACTION_EFFECT_PAIRS.parquet")
    decoded = _decode_frame(frame)
    train_mask = frame.split.eq("train").to_numpy(); val_mask = frame.split.eq("validation").to_numpy(); hold_mask = frame.split.eq("holdout").to_numpy()
    train = {key: value[train_mask] for key, value in decoded.items()}; validation = {key: value[val_mask] for key, value in decoded.items()}; holdout = {key: value[hold_mask] for key, value in decoded.items()}
    norm = _normalisation(frame, train)
    (output_dir / "LOSS_SCALE_AUDIT.json").write_text(json.dumps({"state_scale_median": float(np.median(norm["state_scale"])), "g_scale": float(norm["g_scale"]), "tfv_scale": float(norm["tfv_scale"]), "aux_scale": norm["aux_scale"].tolist(), "recipe_frozen_once": True}, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reports = {}
    for variant in ("baseline", "v2", "gradient"):
        reports[variant] = _train_variant(variant, frame, train, validation, holdout, pairs, local_pairs, norm, output_dir, args, device)
    (output_dir / "LOSS_SCALE_AUDIT.json").write_text(
        json.dumps(
            {
                "normalisation": {
                    "state_scale_median": float(np.median(norm["state_scale"])),
                    "g_scale": float(norm["g_scale"]),
                    "tfv_scale": float(norm["tfv_scale"]),
                    "aux_scale": norm["aux_scale"].tolist(),
                },
                "variants": {name: report["loss_scale_audit"] for name, report in reports.items()},
                "recipe_frozen_once": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    legacy_path = args.dataset_dir.parent / "STEP2_GRADIENT_TRUTH_AUDIT_FAST8_COMBINED.json"
    legacy_reference = None
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy_reference = {key: legacy.get(key) for key in ("median_tfv_local_direction_agreement", "median_pfv_local_direction_agreement", "median_tfv_pairwise_accuracy", "median_pfv_pairwise_accuracy", "median_tfv_spearman", "median_pfv_spearman", "mean_top5_good_action_recall", "mean_predicted_safe_actual_unsafe_fraction")}
    gpu_peak = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else None
    eligible = {
        name: report
        for name, report in reports.items()
        if float(report["holdout"]["decision_quality"]["false_safe_fraction"]) <= 0.05
    }
    if eligible:
        selected_variant = max(
            eligible,
            key=lambda item: (
                float(eligible[item]["holdout"]["median_tfv_pairwise"] or -1.0),
                float(eligible[item]["holdout"]["median_pfv_pairwise"] or -1.0),
                -float(eligible[item]["holdout"]["decision_quality"]["selection_regret_median_m3"] or 1.0e30),
            ),
        )
    else:
        selected_variant = max(reports, key=lambda item: float(reports[item]["holdout"]["median_tfv_pairwise"] or -1.0))
    selected = reports[selected_variant]["holdout"]
    ready = bool(
        selected["decision_quality"]["false_safe_fraction"] <= 0.05
        and (selected["median_tfv_pairwise"] or 0.0) > 0.55
        and (selected["median_pfv_pairwise"] or 0.0) > 0.55
        and (selected["gradient_direction"]["tfv"] or 0.0) > 0.55
        and (selected["gradient_direction"]["pfv"] or 0.0) > 0.55
    )
    audit = {
        "audit_id": "STEP2_V2_DECISION_AUDIT_V1",
        "development_only": True,
        "formal_mainline_authorized": False,
        "architecture": "CANDIDATE_RELATIVE_DIFFERENTIABLE_CONTROL_SURROGATE_V2",
        "data_summary": str(args.dataset_dir / "DATASET_SUMMARY.json"),
        "seed": 42,
        "device": str(device),
        "gpu_peak_allocated_gb": gpu_peak,
        "variants": reports,
        "selected_development_variant": selected_variant,
        "READY_FOR_GRADIENT_SEARCH": ready,
        "readiness_blocker": None if ready else "gradient_direction_and_same_state_decision_fidelity_not_stable_on_holdout",
        "legacy_step2_reference": legacy_reference,
        "no_future_hydraulic_truth_as_input": True,
        "realized_future_rainfall_as_input": False,
        "pfv_admission_authority": "authoritative_label_only_for_development_decision_audit",
        "next_authorized_stage": "offline_gradient_search_only_if_decision_quality_and_PFV_false_safe_are_acceptable",
    }
    (output_dir / "STEP2_V2_DECISION_AUDIT.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "device": str(device), "gpu_peak_allocated_gb": gpu_peak, "variants": list(reports), "output": str(output_dir / "STEP2_V2_DECISION_AUDIT.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
