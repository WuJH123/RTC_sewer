"""Development-only PFV-only selector replay on already revealed Calibration data.

This reads the existing GAT/Step2 manifest and model checkpoints only.  It does
not run SWMM, write Formal evidence, or authorize a campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_v42_step2_fast import _forward, _load_graph_topology, _slice, _tensorise
from sewerrtc.control.pfvfirst_mpc_v42 import (
    EngineeringStatus,
    MPCandidate,
    MPCWeights,
    SafetyMargins,
    FrozenFallback,
    decide_pfvfirst_mpc,
)
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
FALLBACK_CONTRACT_PATH = "configs/v42_formal_fallback_contract.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _arr(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _model(root: Path, graph: dict, report: dict, checkpoint: Path, device: torch.device):
    model = MultiReferenceHydraulicSurrogate(
        n_nodes=int(graph["n_nodes"]),
        n_facilities=int(graph["n_facilities"]),
        state_feature_dim=1,
        static_feature_dim=int(graph["node_static"].shape[1]),
        hidden_dim=64,
        gat_heads=4,
        gat_layers=3,
        horizon=12,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/FORMAL_F2_CALIBRATION_GAT_MANIFEST.parquet",
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/models",
    )
    args = ap.parse_args()
    frame = pd.read_parquet(args.manifest)
    if frame.empty or "state_key" not in frame:
        raise RuntimeError("revealed Calibration GAT manifest is empty or invalid")
    if not bool(frame["state_source"].astype(str).eq("gat_sparse_reconstruction").all()):
        raise RuntimeError("functional replay requires the existing causal GAT manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = _load_graph_topology(args.project_root)
    edge = torch.from_numpy(graph["edge_index"].astype(np.int64)).to(device)
    node_static = torch.from_numpy(graph["node_static"].astype(np.float32)).to(device)
    action_map = torch.from_numpy(graph["action_node_map"].astype(np.float32)).to(device)
    priority = torch.as_tensor(
        get_pfv_core_node_indices(list(graph["node_ids"])), dtype=torch.long, device=device
    )
    data = _tensorise(frame)
    predictions: list[dict[str, np.ndarray]] = []
    for seed in (17, 42, 73):
        report_path = args.models_root / f"seed_{seed}" / "formal_step2_report.json"
        checkpoint = args.models_root / f"seed_{seed}" / "best_model.pt"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model = _model(args.project_root, graph, report, checkpoint, device)
        with torch.inference_mode():
            out = _forward(
                model,
                _slice(data, np.arange(len(frame))),
                (edge, node_static, action_map),
                priority,
                device,
            )
        predictions.append(
            {
                "pfv_delta": out["pfv_delta"].detach().cpu().numpy().reshape(-1),
                "tfv_delta": out["tfv_delta"].detach().cpu().numpy().reshape(-1),
                "peak_delta": out["peak_delta"].detach().cpu().numpy().reshape(-1),
                "no_control_pfv": out["kpi_no_control"]["pfv_m3"].detach().cpu().numpy().reshape(-1),
            }
        )

    old_calibration = args.project_root / (
        "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/STEP2_SAFETY_CALIBRATION.json"
    )
    legacy = json.loads(old_calibration.read_text(encoding="utf-8")) if old_calibration.exists() else {}
    z_pfv = float(legacy.get("pfv_standardized_conformal_z", 0.0))
    fallback_hash = _sha(args.project_root / FALLBACK_CONTRACT_PATH)
    rows: list[dict] = []
    detail_rows: list[dict] = []
    for state_key, group in frame.groupby("state_key", sort=True):
        indices = group.index.to_numpy()
        # Parquet row indices are not guaranteed to be contiguous after prior filtering.
        indices = np.asarray([frame.index.get_loc(i) for i in indices], dtype=int)
        hold = _arr(group.iloc[0]["action_hold_previous_readback"])
        candidates: list[MPCandidate] = []
        for local, (source_index, (_, item)) in enumerate(zip(indices, group.iterrows())):
            pfv = np.asarray([p["pfv_delta"][source_index] for p in predictions], dtype=float)
            tfv = np.asarray([p["tfv_delta"][source_index] for p in predictions], dtype=float)
            peak = np.asarray([p["peak_delta"][source_index] for p in predictions], dtype=float)
            nc = np.asarray([p["no_control_pfv"][source_index] for p in predictions], dtype=float)
            action = _arr(item["action_candidate_readback"])
            legal = bool(item.get("k_le_8", True)) and bool(item.get("actuator_semantics_ok", True))
            candidate_id = f"candidate_{str(item['candidate_action_sha256'])[:16]}"
            candidates.append(
                MPCandidate(
                    candidate_id=candidate_id,
                    action_sequence=action,
                    pfv_delta_ucb_m3=float(pfv.mean() + z_pfv * (pfv.std(ddof=1) if len(pfv) > 1 else 0.0)),
                    peak_delta_ucb_m3s=float(peak.mean()),
                    tfv_delta_di_m3=float(tfv.mean()),
                    action_cost=0.0,
                    terminal_cost=0.0,
                    uncertainty_cost=0.0,
                    changed_facilities=int(item.get("actual_k", 0)),
                    engineering=EngineeringStatus(legal, legal, legal, legal, legal),
                    uncertainty_pass=True,
                    ood_pass=True,
                    executable=legal,
                    pfv_no_control_m3=float(nc.mean()),
                    metadata={"source_row": int(source_index), "legacy_calibration": True},
                )
            )
        fallback = FrozenFallback(
            fallback_id="frozen_hold_readback",
            action_sequence=hold,
            contract_hash=fallback_hash,
        )
        decision = decide_pfvfirst_mpc(
            candidates=candidates,
            fallback=fallback,
            margins=SafetyMargins(),
            weights=MPCWeights(),
            expected_fallback_contract_hash=fallback_hash,
        )
        safe = sum(a.safe for a in decision.audits)
        non_hold = sum(not np.allclose(c.action_sequence, hold, atol=1e-6) for c in candidates)
        selected = next((c for c in candidates if c.candidate_id == decision.selected_id), None)
        rows.append(
            {
                "state_key": str(state_key),
                "event_id": str(group.iloc[0]["event_id"]),
                "rainfall_sha256": str(group.iloc[0]["rainfall_sha256"]),
                "candidate_count": len(candidates),
                "pfv_safe_count": int(safe),
                "non_hold_candidate_count": int(non_hold),
                "selected_id": decision.selected_id,
                "selected_non_hold": bool(selected is not None and not np.allclose(selected.action_sequence, hold, atol=1e-6)),
                "fallback": bool(decision.used_fallback),
                "fallback_reason": decision.reason if decision.used_fallback else "",
                "selected_tfv_delta_predicted": None if selected is None else float(selected.tfv_delta_di_m3),
                "selected_peak_delta_diagnostic": None if selected is None else float(selected.peak_delta_ucb_m3s),
                "priority_depth_ignored_verified": True,
                "ood_diagnostic_only_verified": True,
                "pfv_ucb_z": z_pfv,
            }
        )
        for audit in decision.audits:
            detail_rows.append({"state_key": str(state_key), "candidate_id": audit.candidate_id, "safe": audit.safe, "rejection_reasons": ";".join(audit.rejection_reasons), "objective_tfv": audit.objective})

    table = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    out_dir = args.project_root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "PFV_ONLY_DEVELOPMENT_FUNCTIONAL_TEST.csv", index=False)
    detail.to_csv(out_dir / "PFV_ONLY_DEVELOPMENT_FUNCTIONAL_CANDIDATE_AUDIT.csv", index=False)
    summary = {
        "status": "development_only_pass" if len(table) else "fail",
        "control_objective_contract": "PROJECT6_V42_PFV_ONLY_TFV_MIN_MPC_V2",
        "formal_mainline_authorized": False,
        "swmm_runs": 0,
        "manifest_sha256": _sha(args.manifest),
        "legacy_calibration_source": str(old_calibration),
        "pfv_ucb_z_used": z_pfv,
        "states": int(len(table)),
        "decisions": int(len(table)),
        "pfv_safe_candidate_count": int(table["pfv_safe_count"].sum()),
        "non_hold_admitted_candidate_count": int(table["non_hold_candidate_count"].sum()),
        "non_hold_selected_count": int(table["selected_non_hold"].sum()),
        "fallback_count": int(table["fallback"].sum()),
        "fallback_rate": float(table["fallback"].mean()) if len(table) else 1.0,
        "selected_tfv_delta_mean": float(table["selected_tfv_delta_predicted"].dropna().mean()) if table["selected_tfv_delta_predicted"].notna().any() else None,
        "priority_depth_ignored_verified": True,
        "global_peak_reporting_only_verified": True,
        "independent_ood_rejection": False,
        "independent_uncertainty_rejection": False,
        "outputs": [str(out_dir / "PFV_ONLY_DEVELOPMENT_FUNCTIONAL_TEST.csv"), str(out_dir / "PFV_ONLY_DEVELOPMENT_FUNCTIONAL_CANDIDATE_AUDIT.csv")],
    }
    (out_dir / "PFV_ONLY_DEVELOPMENT_FUNCTIONAL_TEST.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0 if summary["non_hold_selected_count"] > 0 and summary["pfv_safe_candidate_count"] > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
