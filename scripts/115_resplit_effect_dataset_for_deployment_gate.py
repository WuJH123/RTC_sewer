from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.raw_joint_training import aggregate_effect_targets


STRONG_KINDS = {"strong_counterfactual", "strong_single_or_pair"}


def _aggregate_targets(data: np.lib.npyio.NpzFile) -> np.ndarray:
    import torch

    reference = torch.as_tensor(data["reference_risk_rate_seq"], dtype=torch.float32)
    delta = torch.as_tensor(data["delta_risk_rate_seq"], dtype=torch.float32)
    return aggregate_effect_targets(reference, delta).numpy()


def _event_stats(
    data: np.lib.npyio.NpzFile,
    targets: np.ndarray,
    *,
    pfv_abs_margin: float,
    pfv_rel_margin: float,
    tfv_tol: float,
    peak_tol: float,
) -> list[dict[str, object]]:
    event_ids = data["event_ids"].astype(str)
    kinds = data["candidate_kind"].astype(str) if "candidate_kind" in data.files else np.asarray([""] * len(event_ids))
    reference_pfv = data["reference_risk_rate_seq"][:, :, 0].sum(axis=1) * 300.0
    pfv_margin = np.maximum(float(pfv_abs_margin), np.maximum(reference_pfv, 0.0) * float(pfv_rel_margin))
    rows: list[dict[str, object]] = []
    for event_id in sorted(set(event_ids)):
        idx = np.flatnonzero(event_ids == event_id)
        deploy = idx[~np.isin(kinds[idx], sorted(STRONG_KINDS))]
        deploy_targets = targets[deploy] if len(deploy) else np.empty((0, 3), dtype=float)
        tfv_active = np.abs(deploy_targets[:, 1]) > float(tfv_tol) if len(deploy) else np.asarray([], dtype=bool)
        peak_active = np.abs(deploy_targets[:, 2]) > float(peak_tol) if len(deploy) else np.asarray([], dtype=bool)
        row_targets = targets[idx]
        pfv_unsafe = row_targets[:, 0] > pfv_margin[idx]
        peak_unsafe = row_targets[:, 2] > float(peak_tol)
        rows.append(
            {
                "event_id": event_id,
                "rows_total": int(len(idx)),
                "deployment_rows": int(len(deploy)),
                "deployment_tfv_support": int(tfv_active.sum()),
                "deployment_peak_support": int(peak_active.sum()),
                "deployment_tfv_improved": int((deploy_targets[tfv_active, 1] < 0).sum()) if len(deploy) else 0,
                "deployment_tfv_worsened": int((deploy_targets[tfv_active, 1] > 0).sum()) if len(deploy) else 0,
                "deployment_peak_improved": int((deploy_targets[peak_active, 2] < 0).sum()) if len(deploy) else 0,
                "deployment_peak_worsened": int((deploy_targets[peak_active, 2] > 0).sum()) if len(deploy) else 0,
                "pfv_unsafe_rows": int(pfv_unsafe.sum()),
                "peak_unsafe_rows": int(peak_unsafe.sum()),
            }
        )
    return rows


def _select_validation_events(rows: list[dict[str, object]], count: int) -> list[str]:
    def score(row: dict[str, object]) -> tuple[float, int, int, int]:
        tfv_balance = min(int(row["deployment_tfv_improved"]), int(row["deployment_tfv_worsened"]))
        peak_balance = min(int(row["deployment_peak_improved"]), int(row["deployment_peak_worsened"]))
        support = int(row["deployment_tfv_support"]) + int(row["deployment_peak_support"])
        unsafe = int(row["pfv_unsafe_rows"]) + int(row["peak_unsafe_rows"])
        deployment = int(row["deployment_rows"])
        # Balance matters most; raw support and unsafe rows are secondary.
        return (2.0 * tfv_balance + 2.5 * peak_balance + 0.30 * support + 0.20 * unsafe, deployment, support, unsafe)

    candidates = [
        row for row in rows
        if int(row["deployment_rows"]) > 0
        and int(row["deployment_tfv_support"]) >= 6
        and int(row["deployment_peak_support"]) >= 6
    ]
    candidates.sort(key=score, reverse=True)
    selected: list[str] = []
    pfv_unsafe = peak_unsafe = 0
    pfv_candidates = sorted(
        [row for row in candidates if int(row["pfv_unsafe_rows"]) > 0],
        key=lambda row: (
            int(row["pfv_unsafe_rows"]),
            int(row["deployment_peak_support"]) + int(row["deployment_tfv_support"]),
            int(row["peak_unsafe_rows"]),
        ),
        reverse=True,
    )
    for row in pfv_candidates:
        selected.append(str(row["event_id"]))
        pfv_unsafe += int(row["pfv_unsafe_rows"])
        peak_unsafe += int(row["peak_unsafe_rows"])
        if pfv_unsafe >= 8 or len(selected) >= int(count):
            break
    for row in candidates:
        if str(row["event_id"]) in selected:
            continue
        selected.append(str(row["event_id"]))
        pfv_unsafe += int(row["pfv_unsafe_rows"])
        peak_unsafe += int(row["peak_unsafe_rows"])
        if len(selected) >= int(count) and pfv_unsafe >= 8 and peak_unsafe >= 8:
            break
    if len(selected) < int(count):
        for row in candidates:
            event_id = str(row["event_id"])
            if event_id not in selected:
                selected.append(event_id)
            if len(selected) >= int(count):
                break
    return selected[: int(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an event-group split aligned with deployment-direction gating.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_hierarchical_residual_v8_gateupdate.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--validation-events", type=int, default=10)
    parser.add_argument("--pfv-abs-margin-m3", type=float, default=100.0)
    parser.add_argument("--pfv-rel-margin", type=float, default=0.02)
    parser.add_argument("--tfv-direction-tolerance-m3", type=float, default=100.0)
    parser.add_argument("--peak-direction-tolerance", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    dataset_path = root / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    out_npz = root / args.out_npz if not Path(args.out_npz).is_absolute() else Path(args.out_npz)
    out_report = root / args.out_report if not Path(args.out_report).is_absolute() else Path(args.out_report)
    ensure_dir(out_npz.parent)
    ensure_dir(out_report.parent)

    data = np.load(dataset_path, allow_pickle=True)
    targets = _aggregate_targets(data)
    rows = _event_stats(
        data,
        targets,
        pfv_abs_margin=float(args.pfv_abs_margin_m3),
        pfv_rel_margin=float(args.pfv_rel_margin),
        tfv_tol=float(args.tfv_direction_tolerance_m3),
        peak_tol=float(args.peak_direction_tolerance),
    )
    selected = _select_validation_events(rows, int(args.validation_events))
    if not selected:
        raise SystemExit("no deployment-supported validation events could be selected")
    event_ids = data["event_ids"].astype(str)
    split = np.where(np.isin(event_ids, selected), "validation", "train").astype("U16")
    payload = {key: data[key] for key in data.files}
    payload["split"] = split
    np.savez_compressed(out_npz, **payload)

    selected_rows = [row for row in rows if str(row["event_id"]) in set(selected)]
    report = {
        "source_dataset": str(dataset_path),
        "out_npz": str(out_npz),
        "validation_events": selected,
        "validation_event_count": len(selected),
        "train_event_count": int(len(set(event_ids)) - len(selected)),
        "validation_rows": int((split == "validation").sum()),
        "train_rows": int((split == "train").sum()),
        "selected_event_stats": selected_rows,
        "selection_policy": "event-group split optimized for deployment direction support plus unsafe safety support",
        "strong_candidate_kinds_retained_for_safety_only": sorted(STRONG_KINDS),
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
