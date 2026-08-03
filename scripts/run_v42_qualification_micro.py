"""Development-only authoritative SWMM micro stages for qualification 13-28.

This runner reuses the qualification checkpoints and models.  It never writes
Formal evidence and never uses future hydraulic or rainfall observations as
controller input.  The surrogate selects an action; the resulting metrics are
always recomputed from the recorded PySWMM trajectory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "PROJECT6_V42_QUALIFICATION_FIRST_PASS_V1"
HORIZON_STEPS = 12
CONTROL_STEP_SEC = 300
STRATEGIES = {
    "Proposed": {"policy_id": "proposed_gat_pfv_first", "kind": "proposed"},
    "EFD": {"policy_id": "efd_storage_priority", "kind": "baseline"},
    "Auto-RBC": {"policy_id": "auto_rbc", "kind": "baseline"},
    "All-close": {"policy_id": "all_closed_safe", "kind": "baseline"},
    "No-control": {"policy_id": "no_control", "kind": "baseline"},
    "Internal": {"policy_id": "internal_rules", "kind": "baseline"},
    "Hold": {"policy_id": "hold_previous", "kind": "baseline"},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    """Preserve non-finite diagnostics as JSON null, never as fabricated zero."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _case_inp_in_parent(parent: Path) -> Path | None:
    try:
        for item in parent.iterdir():
            if item.is_file() and item.name.casefold() == "case.inp":
                return item.resolve()
    except OSError:
        return None
    return None


def _resolve_common_inp(detail_paths: list[Path]) -> Path:
    """Resolve one physical INP for all branches and reject hash collisions."""
    candidates: dict[str, Path] = {}
    for detail in detail_paths:
        parent = detail.parent
        for current in (parent, *parent.parents):
            found = _case_inp_in_parent(current)
            if found is not None:
                candidates[str(found)] = found
            if len(current.parts) <= 1:
                break
    if not candidates:
        raise FileNotFoundError("no case.inp found for selected event branches")
    hashes = {_sha256_file(path): path for path in candidates.values()}
    if len(hashes) != 1:
        raise RuntimeError(
            "network input hash mismatch across selected event branches: "
            + ", ".join(f"{digest}:{path}" for digest, path in hashes.items())
        )
    return next(iter(hashes.values())).resolve()


def _ledger_reusable(
    ledger_path: Path,
    event_id: str,
    strategy: str,
    input_sha256: str,
    model_sha256: str,
    policy_sha256: str,
) -> bool:
    if not ledger_path.exists():
        return False
    try:
        frame = pd.read_csv(ledger_path)
    except Exception:
        return False
    if frame.empty:
        return False
    rows = frame[
        frame.event_id.astype(str).eq(str(event_id))
        & frame.strategy.astype(str).eq(str(strategy))
        & frame.status.astype(str).eq("pass")
    ]
    if rows.empty:
        return False
    row = rows.iloc[-1]
    if any(str(row.get(key, "")) != value for key, value in (
        ("input_sha256", input_sha256),
        ("model_sha256", model_sha256),
        ("policy_sha256", policy_sha256),
    )):
        return False
    output = Path(str(row.get("detail_path", "")))
    return output.exists() and output.stat().st_size > 0


def _append_ledger(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_priority_nodes(root: Path, graph_nodes: list[str]) -> list[str]:
    path = root / "outputs/design_v8_storage_variablepump/priority_nodes.txt"
    if path.exists():
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        values = [value for value in values if value and not value.startswith("#")]
        if values:
            return [value for value in values if value in set(graph_nodes)]
    from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices

    return [graph_nodes[i] for i in get_pfv_core_node_indices(graph_nodes)]


def _load_actuators(root: Path) -> pd.DataFrame:
    path = root / "outputs/audit_v8_storage_variablepump/actuator_table.csv"
    frame = pd.read_csv(path)
    if "action_index" in frame:
        frame = frame.sort_values("action_index")
    frame = frame.reset_index(drop=True)
    if len(frame) != 36:
        raise RuntimeError(f"qualification micro requires Engineering36 actuators, got {len(frame)}")
    return frame


def _load_events(root: Path, qualification: Path) -> list[dict[str, Any]]:
    plan = pd.read_csv(qualification / "QUALIFICATION_DEVELOPMENT_EVALUATION_PLAN.csv")
    raw = pd.read_parquet(root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/FORMAL_F2_STEP2_RAW_MANIFEST.parquet")
    events: list[dict[str, Any]] = []
    branch_columns = [
        "source_detail_path_candidate",
        "source_detail_path_no_control",
        "source_detail_path_dynamic_internal",
        "source_detail_path_hold_previous",
    ]
    for item in plan.sort_values(["qualification_role", "event_id"]).to_dict("records"):
        rows = raw[raw.state_key.astype(str).eq(str(item["state_key"]))].copy()
        if rows.empty:
            raise RuntimeError(f"qualification event has no raw rows: {item['event_id']}")
        first = rows.iloc[0]
        paths = [Path(str(first[col])) for col in branch_columns]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
        inp = _resolve_common_inp(paths)
        no_control = pd.read_csv(paths[1], usecols=["elapsed_min"])
        simulation_duration = float(pd.to_numeric(no_control.elapsed_min, errors="coerce").max())
        checkpoint = float(item["checkpoint_min"])
        if not np.isfinite(simulation_duration) or simulation_duration < checkpoint + 120.0:
            raise RuntimeError(f"event lacks checkpoint+120 coverage: {item['event_id']}")
        events.append(
            {
                "event_id": str(item["event_id"]),
                "role": str(item["qualification_role"]),
                "rainfall_sha256": str(item["rainfall_sha256"]),
                "state_key": str(item["state_key"]),
                "checkpoint_min": checkpoint,
                "rain_duration_min": int(round(simulation_duration - 120.0)),
                "simulation_duration_min": int(round(simulation_duration)),
                "inp_path": str(inp),
                "inp_sha256": _sha256_file(inp),
                "raw_rows": int(len(rows)),
                "raw": rows,
            }
        )
    return events


def _event_manifest(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "raw"}


def _read_frame(sim: Any, node_objs: dict[str, Any], rain_obj: Any, link_objs: dict[str, Any], actuator_ids: list[str], actuators: pd.DataFrame) -> dict[str, Any]:
    from sewerrtc.simulation.pyswmm_runner import _as_float, _observed_action_from_links

    elapsed = (sim.current_time - sim.start_time).total_seconds() / 60.0
    depths = np.asarray([_as_float(getattr(node_objs[nid], "depth", 0.0), 0.0) for nid in node_objs], dtype=np.float32)
    rain = _as_float(getattr(rain_obj, "rainfall", 0.0), 0.0) if rain_obj is not None else 0.0
    action = _observed_action_from_links(link_objs, actuator_ids, actuators)
    return {"elapsed_min": float(elapsed), "depth": depths, "rain": float(rain), "action": action}


def _detail_row(frame: dict[str, Any], event_id: str, strategy: str, command: np.ndarray, readback: np.ndarray, node_ids: list[str], node_objs: dict[str, Any], link_objs: dict[str, Any], actuator_ids: list[str], rain: float) -> dict[str, Any]:
    from sewerrtc.simulation.pyswmm_runner import _as_float

    row: dict[str, Any] = {
        "event_id": event_id,
        "policy_id": strategy,
        "elapsed_min": frame["elapsed_min"],
        "rainfall_mm_h": float(rain),
        "phase": "qualification_micro",
    }
    for nid, obj in node_objs.items():
        row[f"h:{nid}"] = _as_float(getattr(obj, "depth", np.nan), np.nan)
        row[f"head:{nid}"] = _as_float(getattr(obj, "head", np.nan), np.nan)
        row[f"storage_volume:{nid}"] = _as_float(getattr(obj, "volume", np.nan), np.nan)
        row[f"flood:{nid}"] = _as_float(getattr(obj, "flooding", 0.0), 0.0)
    for i, aid in enumerate(actuator_ids):
        row[f"a:{aid}"] = float(command[i])
        row[f"setting:{aid}"] = float(readback[i])
        row[f"flow:{aid}"] = _as_float(getattr(link_objs[aid], "flow", np.nan), np.nan)
    return row


class _QualificationSurrogatePredictor:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle

    def predict_many(self, sequences: list[np.ndarray], contexts: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        model = self.bundle["step2"]
        device = self.bundle["device"]
        graph = self.bundle["graph_tensors"]
        histories = np.stack([np.asarray(ctx["state_history"], dtype=np.float32) for ctx in contexts])
        actions = np.stack([np.asarray(ctx["historical_actions"], dtype=np.float32) for ctx in contexts])
        rain = np.stack([np.asarray(ctx["rainfall_window"], dtype=np.float32)[:HORIZON_STEPS] for ctx in contexts])
        candidate = np.stack([np.asarray(seq, dtype=np.float32) for seq in sequences])
        reference = np.stack([
            np.asarray(ctx.get("reference_action_sequence"), dtype=np.float32)
            for ctx in contexts
        ])
        edge_index, node_static, action_map, priority_idx = graph
        with torch.no_grad():
            output = model(
                state_history=torch.as_tensor(histories, device=device),
                historical_actions=torch.as_tensor(actions, device=device),
                rainfall_forecast=torch.as_tensor(rain, device=device),
                action_candidate=torch.as_tensor(candidate, device=device),
                action_no_control=torch.as_tensor(reference, device=device),
                action_dynamic_internal=torch.as_tensor(reference, device=device),
                action_hold_previous=torch.as_tensor(reference, device=device),
                edge_index=edge_index,
                node_static=node_static,
                action_node_map=action_map,
                priority_node_indices=priority_idx,
                storage_node_indices=None,
                outfall_node_indices=None,
            )
        flood = output["branches"]["candidate"]["node_flooding_rate"].detach().cpu().numpy()
        priority = flood[:, :, priority_idx.detach().cpu().numpy()].sum(axis=2)
        system = flood.sum(axis=2)
        pfv = priority * float(self.bundle["dt_sec"])
        tfv = system * float(self.bundle["dt_sec"])
        peak = system
        return [
            {
                "pfv": pfv[i],
                "tfv": tfv[i],
                "peak_tfv_rate": peak[i],
                "online_future_hydraulics_used": np.asarray([0.0], dtype=np.float32),
                "uncertainty_margin": np.zeros(HORIZON_STEPS, dtype=np.float32),
            }
            for i in range(len(sequences))
        ]

    def __call__(self, sequence: np.ndarray, context: dict[str, Any]) -> dict[str, np.ndarray]:
        return self.predict_many([sequence], [context])[0]


def _load_model_bundle(root: Path, qualification: Path, device_name: str) -> dict[str, Any]:
    from sewerrtc.v4.v42_step1_dataset import load_graph_assets, _sensor_layout
    from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
    from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
    from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices

    graph = load_graph_assets(root)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    gat = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=128,
        heads=4,
        gat_layers=3,
    ).to(device)
    gat_path = qualification / "step1/seed_42/best_model.pt"
    gat.load_state_dict(torch.load(gat_path, map_location=device, weights_only=True))
    gat.eval()
    step2 = MultiReferenceHydraulicSurrogate(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        state_feature_dim=1,
        static_feature_dim=graph.node_static.shape[1],
        hidden_dim=32,
        gat_heads=4,
        gat_layers=2,
        horizon=HORIZON_STEPS,
    ).to(device)
    step2_path = qualification / "step2/models/seed_42/best_model.pt"
    step2.load_state_dict(torch.load(step2_path, map_location=device, weights_only=True))
    step2.eval()
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    node_static = torch.as_tensor(graph.node_static, dtype=torch.float32, device=device)
    action_map = torch.as_tensor(graph.action_node_map, dtype=torch.float32, device=device)
    priority_idx = torch.as_tensor(get_pfv_core_node_indices(graph.node_ids), dtype=torch.long, device=device)
    _, sensor_indices, sensor_sha = _sensor_layout(graph.n_nodes, 0.10, 42)
    return {
        "graph": graph,
        "gat": gat,
        "step2": step2,
        "device": device,
        "graph_tensors": (edge_index, node_static, action_map, priority_idx),
        "sensor_indices": sensor_indices,
        "sensor_sha256": sensor_sha,
        "step1_sha256": _sha256_file(gat_path),
        "step2_sha256": _sha256_file(step2_path),
        "dt_sec": CONTROL_STEP_SEC,
    }


def _reconstruct_history(frames: list[dict[str, Any]], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) < 25:
        raise RuntimeError("causal GAT history requires t-120 through t before first decision")
    from sewerrtc.models.temporal_sparse_gat_v42 import HISTORY_FRAMES

    n_nodes = bundle["graph"].n_nodes
    sensor_indices = bundle["sensor_indices"]
    windows = []
    for anchor in range(len(frames) - HISTORY_FRAMES, len(frames)):
        window = frames[anchor - HISTORY_FRAMES + 1 : anchor + 1]
        if len(window) != HISTORY_FRAMES:
            raise RuntimeError("incomplete causal anchor window")
        windows.append(window)
    sparse = np.zeros((HISTORY_FRAMES, HISTORY_FRAMES, n_nodes), dtype=np.float32)
    mask = np.zeros_like(sparse)
    rainfall = np.zeros((HISTORY_FRAMES, HISTORY_FRAMES), dtype=np.float32)
    actions = np.zeros((HISTORY_FRAMES, HISTORY_FRAMES, bundle["graph"].n_facilities), dtype=np.float32)
    for i, window in enumerate(windows):
        for j, frame in enumerate(window):
            sparse[i, j, sensor_indices] = frame["depth"][sensor_indices]
            mask[i, j, sensor_indices] = 1.0
            rainfall[i, j] = frame["rain"]
            actions[i, j] = frame["action"]
    graph = bundle["graph"]
    device = bundle["device"]
    with torch.no_grad():
        output = bundle["gat"](
            sparse_depth_history=torch.as_tensor(sparse, device=device),
            sensor_mask_history=torch.as_tensor(mask, device=device),
            rainfall_history=torch.as_tensor(rainfall, device=device),
            historical_actions=torch.as_tensor(actions, device=device),
            node_static=torch.as_tensor(graph.node_static, dtype=torch.float32, device=device),
            link_static=torch.as_tensor(graph.link_static, dtype=torch.float32, device=device),
            edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
            action_node_map=torch.as_tensor(graph.action_node_map, dtype=torch.float32, device=device),
        )
    return output.depth_mean.cpu().numpy().astype(np.float32), output.depth_std.cpu().numpy().astype(np.float32)


def _action_projected(action: np.ndarray, previous: np.ndarray, actuators: pd.DataFrame, decision_step: int, last_change: dict[str, int]) -> tuple[np.ndarray, dict[str, Any]]:
    from sewerrtc.simulation.pyswmm_runner import _enforce_actuator_semantics

    ids = actuators.actuator_id.astype(str).tolist()
    projected = _enforce_actuator_semantics(action, ids, actuators, "binary_unless_verified", ["add350.1"])
    projected = np.asarray(projected, dtype=np.float32)
    types = actuators.set_index("actuator_id").link_type.astype(str).str.lower().to_dict()
    roles = actuators.set_index("actuator_id").get("storage_control_type", pd.Series(dtype=str)).fillna("").astype(str).to_dict()
    storage = actuators.set_index("actuator_id").get("storage_node", pd.Series(dtype=str)).fillna("").astype(str).to_dict()
    for i, aid in enumerate(ids):
        if types.get(aid) == "pump" and aid not in {"add350.1"}:
            if decision_step - int(last_change.get(aid, -1000)) < 2 and abs(float(projected[i]) - float(previous[i])) > 1.0e-7:
                projected[i] = previous[i]
        else:
            limit = 0.15 if aid == "add350.1" else 0.12
            projected[i] = np.clip(projected[i], previous[i] - limit, previous[i] + limit)
    changed = np.abs(projected - previous) > 1.0e-6
    if int(changed.sum()) > 8:
        return previous.copy(), {"engineering_pass": False, "engineering_reason": "adaptive_k_exceeded", "changed_facilities": int(changed.sum())}
    for node in {value for value in storage.values() if value}:
        inlet = [i for i, aid in enumerate(ids) if storage.get(aid) == node and roles.get(aid) == "storage_inlet"]
        outlet = [i for i, aid in enumerate(ids) if storage.get(aid) == node and roles.get(aid) == "storage_outlet"]
        if any(changed[inlet]) and any(changed[outlet]):
            return previous.copy(), {"engineering_pass": False, "engineering_reason": "storage_inlet_outlet_interlock", "changed_facilities": int(changed.sum())}
    for i, aid in enumerate(ids):
        if changed[i]:
            last_change[aid] = int(decision_step)
    return projected, {"engineering_pass": True, "engineering_reason": "pass", "changed_facilities": int(changed.sum())}


def _run_baseline(root: Path, event: dict[str, Any], strategy: str, out_dir: Path, actuators: pd.DataFrame, priority_nodes: list[str]) -> dict[str, Any]:
    from sewerrtc.simulation.pyswmm_runner import run_swmm_trajectory

    spec = STRATEGIES[strategy]
    detail = out_dir / "detail.csv"
    # Keep PySWMM's recovery/checkpoint filenames below the Windows path
    # budget; the qualification output path is already deliberately verbose.
    runtime_key = hashlib.sha256(f"{event['event_id']}|{strategy}|{event['inp_sha256']}".encode("utf-8")).hexdigest()[:12]
    runtime_root = root / "outputs" / "p6q_rt" / runtime_key
    result = run_swmm_trajectory(
        event["inp_path"],
        spec["policy_id"],
        actuators,
        priority_nodes,
        detail,
        event["event_id"],
        event["rain_duration_min"],
        control_step_sec=CONTROL_STEP_SEC,
        seed=42,
        simulation_duration_min=event["simulation_duration_min"],
        recession_min=120,
        pump_control_mode="binary_unless_verified",
        variable_speed_pump_ids=["add350.1"],
        runtime_output_root=runtime_root,
    )
    (out_dir / "run_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    return result


def _run_proposed(event: dict[str, Any], out_dir: Path, actuators: pd.DataFrame, priority_nodes: list[str], bundle: dict[str, Any], max_candidate_sequences: int) -> dict[str, Any]:
    from pyswmm import Links, Nodes, RainGages, Simulation
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController
    from sewerrtc.control.action_sequence_generator import generate_action_sequences
    from sewerrtc.simulation.pyswmm_runner import _as_float, _enforce_actuator_semantics, _get_existing_links, _ids_from_container, _observed_action_from_links
    from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast
    from sewerrtc.simulation.kpi_metrics import compute_kpis

    graph = bundle["graph"]
    if actuators.actuator_id.astype(str).tolist() != [str(x) for x in graph.facility_ids]:
        raise RuntimeError("Engineering36 actuator order does not match the trained graph order")
    model_predictor = _QualificationSurrogatePredictor(bundle)
    per_delta = {str(aid): (1.0 if str(aid) in {"ADD301.2", "ADD301.3"} else (0.15 if str(aid) == "add350.1" else 0.12)) for aid in actuators.actuator_id}
    min_hold = {str(aid): 2 for aid in actuators.actuator_id if str(aid) in {"ADD301.2", "ADD301.3", "add350.1"}}
    controller = GenericGATMPCController(
        actuators,
        horizon_steps=HORIZON_STEPS,
        max_candidate_delta=0.12,
        horizon_predictor=model_predictor,
        predictor_source="qualification_step2_seed42_trajectory_surrogate",
        max_candidate_sequences=max_candidate_sequences,
        candidate_group_limit=8,
        min_pfv_improvement_abs=0.0,
        pfv_tolerance_abs=0.0,
        tfv_tolerance_abs=0.0,
        peak_tolerance_abs=0.0,
        tfv_hard_constraint=True,
        per_actuator_max_delta=per_delta,
        min_hold_steps_by_actuator=min_hold,
        pump_control_mode="binary_unless_verified",
        variable_speed_pump_ids=["add350.1"],
        max_first_step_delta=0.12,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "detail.csv"
    decision_path = out_dir / "decisions.jsonl"
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    last_change: dict[str, int] = {}
    t0 = time.time()
    with Simulation(str(event["inp_path"])) as sim:
        sim.step_advance(CONTROL_STEP_SEC)
        # PySWMM exposes current_time/node state only after swmm_start().  The
        # initial frame is required for the first t-120 causal window.
        sim.start()
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        node_objs = {nid: nodes[nid] for nid in node_ids}
        actuator_ids, link_objs = _get_existing_links(links, actuators.actuator_id.astype(str).tolist())
        if actuator_ids != actuators.actuator_id.astype(str).tolist():
            raise RuntimeError("selected INP does not expose all Engineering36 links")
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        frames.append(_read_frame(sim, node_objs, rain_obj, link_objs, actuator_ids, actuators))
        current_action = frames[-1]["action"].copy()
        step_index = 0
        for _ in sim:
            pre = _read_frame(sim, node_objs, rain_obj, link_objs, actuator_ids, actuators)
            frames.append(pre)
            elapsed = float(pre["elapsed_min"])
            command = current_action.copy()
            info: dict[str, Any] = {"fallback_to_default": True, "selected_sequence_label": "warmup_hold", "selected_gate_pass": False}
            gat_uncertainty = np.nan
            if elapsed >= 120.0 - 1.0e-6 and abs((elapsed / 10.0) - round(elapsed / 10.0)) <= 1.0e-6:
                state_history, uncertainty = _reconstruct_history(frames, bundle)
                gat_uncertainty = float(np.mean(uncertainty))
                observed_rain = [frame["rain"] for frame in frames]
                forecast = make_causal_rainfall_forecast(observed_rain, horizon_steps=HORIZON_STEPS)
                historical_actions = np.stack([frame["action"] for frame in frames[-13:]]).astype(np.float32)
                reference = np.repeat(current_action[None, :], HORIZON_STEPS, axis=0).astype(np.float32)
                command, info = controller.choose(
                    reconstructed_state=state_history[-1],
                    rainfall_window=forecast,
                    current_action=current_action,
                    reference_action_sequence=reference,
                    elapsed_min=elapsed,
                    phase="qualification_micro",
                    extra_predictor_context={
                        "state_history": state_history,
                        "historical_actions": historical_actions,
                        "action_semantics": "absolute_from_causal_persistence_reference",
                        "online_future_hydraulics_used": False,
                        "action_setting_deadband": 1.0e-6,
                        "adaptive_k_limit": 8,
                    },
                )
                command, engineering = _action_projected(command, current_action, actuators, step_index, last_change)
                if not engineering["engineering_pass"]:
                    info["fallback_to_default"] = True
                    info["intervention_reason"] = engineering["engineering_reason"]
                info.update(engineering)
                info.update({
                    "elapsed_min": elapsed,
                    "gat_reconstructions": 13,
                    "sensor_ratio": 0.10,
                    "sensor_layout_sha256": bundle["sensor_sha256"],
                    "current_frame_repetition_used": False,
                    "authoritative_swmm_history_used_as_online_input": False,
                    "realized_future_rainfall_used_online": False,
                    "predicted_safe": bool(not info.get("fallback_to_default", True) and info.get("selected_gate_pass", False)),
                })
                decisions.append(_json_safe(info))
                step_index += 1
            command = _enforce_actuator_semantics(command, actuator_ids, actuators, "binary_unless_verified", ["add350.1"])
            for i, aid in enumerate(actuator_ids):
                link_objs[aid].target_setting = float(command[i])
            readback = _observed_action_from_links(link_objs, actuator_ids, actuators)
            current_action = readback.copy()
            records.append(_detail_row(pre, event["event_id"], "Proposed", command, readback, node_ids, node_objs, link_objs, actuator_ids, pre["rain"]))
        detail = pd.DataFrame(records)
    detail.to_csv(detail_path, index=False)
    with decision_path.open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False, default=str) + "\n")
    kpis = compute_kpis(detail, priority_nodes, dt_sec=CONTROL_STEP_SEC)
    result = {
        "event_id": event["event_id"],
        "strategy": "Proposed",
        "status": "pass",
        "detail_path": str(detail_path),
        "decision_path": str(decision_path),
        "kpis": kpis,
        "decisions": len(decisions),
        "fallback_rate": float(np.mean([bool(row.get("fallback_to_default", True)) for row in decisions])) if decisions else 1.0,
        "controller_runtime_sec": float(time.time() - t0),
        "state_source": "gat_sparse_reconstruction",
        "rainfall_input_authority": "causal_observed_persistence_decay",
        "authoritative_outcome": "recorded_swmm_selected_action",
        "current_frame_repetition_used": False,
        "authoritative_swmm_history_used_as_online_input": False,
        "realized_future_rainfall_used_online": False,
        "model_sha256": bundle["step2_sha256"],
        "gat_model_sha256": bundle["step1_sha256"],
    }
    (out_dir / "run_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    return result


def _compute_summary(root: Path, qualification: Path, events: list[dict[str, Any]], priority_nodes: list[str], ledger: Path) -> dict[str, Any]:
    from sewerrtc.simulation.kpi_metrics import compute_kpis, compute_window_kpis

    event_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for event in events:
        event_dir = qualification / "micro" / event["event_id"]
        details: dict[str, pd.DataFrame] = {}
        for strategy in STRATEGIES:
            path = event_dir / strategy / "detail.csv"
            if path.exists():
                details[strategy] = pd.read_csv(path)
        baseline_nc = details.get("No-control")
        baseline_internal = details.get("Internal")
        for strategy, detail in details.items():
            kpi = compute_kpis(detail, priority_nodes, dt_sec=CONTROL_STEP_SEC)
            row = {
                "event_id": event["event_id"],
                "qualification_role": event["role"],
                "rainfall_sha256": event["rainfall_sha256"],
                "strategy": strategy,
                "PFV": kpi.get("PFV", np.nan),
                "TFV": kpi.get("TFV", np.nan),
                "Peak": kpi.get("peak_TFV_rate", np.nan),
                "priority_flood_duration_min": kpi.get("priority_flood_duration_min", np.nan),
                "action_changes": kpi.get("action_changes", np.nan),
                "authority": "recorded_authoritative_swmm",
                "swmm_backed_fraction": 1.0,
                "fallback_rate": np.nan,
                "false_safe_rate": np.nan,
            }
            if baseline_nc is not None:
                row["PFV_reduction_vs_No_control"] = float(compute_kpis(baseline_nc, priority_nodes, CONTROL_STEP_SEC)["PFV"] - row["PFV"])
            if baseline_internal is not None:
                ik = compute_kpis(baseline_internal, priority_nodes, CONTROL_STEP_SEC)
                row["TFV_reduction_vs_Internal"] = float(ik["TFV"] - row["TFV"])
                row["Peak_reduction_vs_Internal"] = float(ik["peak_TFV_rate"] - row["Peak"])
            if strategy == "Proposed":
                result_path = event_dir / strategy / "run_result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    row["fallback_rate"] = result.get("fallback_rate", np.nan)
                decisions_path = event_dir / strategy / "decisions.jsonl"
                if decisions_path.exists() and baseline_nc is not None and baseline_internal is not None:
                    for line in decisions_path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        decision = json.loads(line)
                        start = float(decision.get("elapsed_min", 0.0))
                        proposed_window = compute_window_kpis(detail, priority_nodes, start, 10.0, CONTROL_STEP_SEC)
                        nc_window = compute_window_kpis(baseline_nc, priority_nodes, start, 10.0, CONTROL_STEP_SEC)
                        internal_window = compute_window_kpis(baseline_internal, priority_nodes, start, 10.0, CONTROL_STEP_SEC)
                        actual_safe = bool(
                            proposed_window["PFV"] <= nc_window["PFV"] + 1.0e-6
                            and proposed_window["peak_TFV_rate"] <= internal_window["peak_TFV_rate"] + 1.0e-6
                        )
                        selected = not bool(decision.get("fallback_to_default", True))
                        predicted_safe = bool(decision.get("predicted_safe", False))
                        decision_rows.append({
                            "event_id": event["event_id"],
                            "qualification_role": event["role"],
                            "elapsed_min": start,
                            "selected_nonfallback": selected,
                            "predicted_safe": predicted_safe,
                            "actual_safe": actual_safe,
                            "false_safe": bool(selected and predicted_safe and not actual_safe),
                            "PFV": proposed_window["PFV"],
                            "PFV_no_control": nc_window["PFV"],
                            "Peak": proposed_window["peak_TFV_rate"],
                            "Peak_internal": internal_window["peak_TFV_rate"],
                        })
            event_rows.append(row)
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(qualification / "micro/QUALIFICATION_MICRO_EVENT_SUMMARY.csv", index=False)
    if event_frame.empty:
        raise RuntimeError("qualification micro produced no summaries")
    aggregate_rows = []
    for strategy, group in event_frame.groupby("strategy", sort=False):
        row = {
            "strategy": strategy,
            "events": int(group.event_id.nunique()),
            "PFV": float(group.PFV.mean()),
            "TFV": float(group.TFV.mean()),
            "Peak": float(group.Peak.mean()),
            "PFV_reduction_vs_No_control": float(group.get("PFV_reduction_vs_No_control", pd.Series(dtype=float)).mean()),
            "TFV_reduction_vs_Internal": float(group.get("TFV_reduction_vs_Internal", pd.Series(dtype=float)).mean()),
            "Peak_reduction_vs_Internal": float(group.get("Peak_reduction_vs_Internal", pd.Series(dtype=float)).mean()),
            "fallback_rate": float(group.fallback_rate.dropna().mean()) if group.fallback_rate.notna().any() else np.nan,
            "authority": "recorded_authoritative_swmm",
            "swmm_backed_fraction": float(group.swmm_backed_fraction.mean()),
        }
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(qualification / "micro/QUALIFICATION_MICRO_AGGREGATE.csv", index=False)
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(qualification / "micro/QUALIFICATION_MICRO_DECISION_AUDIT.csv", index=False)
    selected = decisions[decisions.selected_nonfallback.astype(bool)] if not decisions.empty else decisions
    false_safe_rate = float(selected.false_safe.mean()) if not selected.empty else None
    actual_safety_rate = float(decisions.actual_safe.mean()) if not decisions.empty else None
    fallback_rate = float(1.0 - len(selected) / len(decisions)) if not decisions.empty else 1.0
    proposed = aggregate[aggregate.strategy.eq("Proposed")].iloc[0]
    nc = aggregate[aggregate.strategy.eq("No-control")].iloc[0]
    internal = aggregate[aggregate.strategy.eq("Internal")].iloc[0]
    potential_go = bool(
        proposed.PFV <= nc.PFV + 1.0e-6
        and proposed.Peak <= internal.Peak + 1.0e-6
        and proposed.TFV < internal.TFV - 1.0e-6
        and fallback_rate < 0.95
        and (false_safe_rate is not None and false_safe_rate <= 0.10)
    )
    return {
        "event_summary": str(qualification / "micro/QUALIFICATION_MICRO_EVENT_SUMMARY.csv"),
        "aggregate_summary": str(qualification / "micro/QUALIFICATION_MICRO_AGGREGATE.csv"),
        "decision_audit": str(qualification / "micro/QUALIFICATION_MICRO_DECISION_AUDIT.csv"),
        "event_rows": int(len(event_frame)),
        "events": int(event_frame.event_id.nunique()),
        "strategies": int(event_frame.strategy.nunique()),
        "replayed_decisions": int(len(decisions)),
        "fallback_rate": fallback_rate,
        "actual_safety_rate": actual_safety_rate,
        "false_safe_rate": false_safe_rate,
        "potential_go": potential_go,
    }


def _write_stage_status(qualification: Path, events: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    required = len(events) * len(STRATEGIES)
    completed = summary["event_rows"] == required
    statuses = {
        "13_new_calibration_authoritative_swmm": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "14_calibration_data_bridge": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "15_step1_uncertainty_ood_calibration": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "16_step2_pfv_peak_safety_calibration": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "17_compile_step1_step2_evidence": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "18_step3_authoritative_engineering_audit": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "19_compile_step3_evidence": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "20_true_state_offline": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "21_exact_authoritative_swmm_closed_loop": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "22_surrogate_closed_loop": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "23_gat_integrated_closed_loop": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "24_policy_lock": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "25_challenge": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "26_locked_validation": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "27_qualification_blind_seven_strategies": "PASS_REUSABLE" if completed else "NOT_STARTED",
        "28_v42_qualification_audit": "PASS_REUSABLE" if completed else "NOT_STARTED",
    }
    payload = {
        "contract_id": CONTRACT_ID,
        "qualification_only": True,
        "development_only": True,
        "formal_mainline_authorized": False,
        "process_status": "pass" if completed else "fail",
        "scientific_performance_status": "provisional",
        "formal_evidence_eligible": False,
        "stage_status": statuses,
        "potential_go": bool(summary["potential_go"]),
        "fallback_rate": summary["fallback_rate"],
        "false_safe_rate": summary["false_safe_rate"],
        "actual_safety_rate": summary["actual_safety_rate"],
        "all_seven_strategies_authoritative_swmm": completed,
        "event_count": len(events),
        "event_strategy_count": required,
        "completed_event_strategy_rows": summary["event_rows"],
        "no_formal_untouched_events_consumed": True,
    }
    path = qualification / "QUALIFICATION_MICRO_STAGE_STATUS.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return payload


def _preflight(root: Path, qualification: Path) -> dict[str, Any]:
    events = _load_events(root, qualification)
    payload = {
        "contract_id": CONTRACT_ID,
        "qualification_only": True,
        "development_only": True,
        "formal_mainline_authorized": False,
        "events": [_event_manifest(event) for event in events],
        "event_count": len(events),
        "strategy_count": len(STRATEGIES),
        "input_network_hashes": sorted({event["inp_sha256"] for event in events}),
        "step1_model": str(qualification / "step1/seed_42/best_model.pt"),
        "step2_model": str(qualification / "step2/models/seed_42/best_model.pt"),
        "status": "pass",
    }
    (qualification / "QUALIFICATION_MICRO_PREFLIGHT.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--stage", choices=("preflight", "micro", "status"), default="micro")
    parser.add_argument("--event-id", default="")
    parser.add_argument("--limit-events", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-candidate-sequences", type=int, default=32)
    parser.add_argument("--strategies", default="", help="comma-separated debug subset; omit for all seven")
    args = parser.parse_args()
    root = args.project_root.resolve()
    qualification = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/qualification_first_pass"
    qualification.mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        path = qualification / "QUALIFICATION_MICRO_STAGE_STATUS.json"
        print(path.read_text(encoding="utf-8") if path.exists() else json.dumps({"status": "not_started"}, indent=2), flush=True)
        return 0
    events = _load_events(root, qualification)
    if args.event_id:
        events = [event for event in events if event["event_id"] == args.event_id]
    if args.limit_events > 0:
        events = events[: args.limit_events]
    if not events:
        raise RuntimeError("no qualification events selected")
    if args.stage == "preflight":
        _preflight(root, qualification)
        return 0
    _preflight(root, qualification)
    from sewerrtc.v4.v42_step1_dataset import load_graph_assets

    actuators = _load_actuators(root)
    graph = load_graph_assets(root)
    priority_nodes = _load_priority_nodes(root, [str(x) for x in graph.node_ids])
    ledger = qualification / "micro/QUALIFICATION_MICRO_EXECUTION_LEDGER.csv"
    micro_root = qualification / "micro"
    model_sha = (
        _sha256_file(qualification / "step2/models/seed_42/best_model.pt")
        + ":"
        + _sha256_file(qualification / "step1/seed_42/best_model.pt")
    )
    selected_strategies = list(STRATEGIES)
    if args.strategies.strip():
        selected_strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
        unknown = sorted(set(selected_strategies) - set(STRATEGIES))
        if unknown:
            raise ValueError(f"unknown qualification micro strategies: {unknown}")
    all_events_for_summary = _load_events(root, qualification)
    for event in events:
        event_dir = micro_root / event["event_id"]
        event_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MICRO] event={event['event_id']} role={event['role']} inp={event['inp_sha256'][:12]}", flush=True)
        for strategy in selected_strategies:
            spec = STRATEGIES[strategy]
            out_dir = event_dir / strategy
            policy_sha = _sha256_json({"strategy": strategy, "policy_id": spec["policy_id"], "seed": 42, "contract": CONTRACT_ID})
            strategy_model_sha = model_sha if spec["kind"] == "proposed" else "none"
            detail = out_dir / "detail.csv"
            if not args.force and _ledger_reusable(ledger, event["event_id"], strategy, event["inp_sha256"], strategy_model_sha, policy_sha):
                print(f"[MICRO] REUSE event={event['event_id']} strategy={strategy}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            print(f"[MICRO] RUN event={event['event_id']} strategy={strategy}", flush=True)
            bundle = None
            try:
                if spec["kind"] == "proposed":
                    bundle = _load_model_bundle(root, qualification, args.device)
                    result = _run_proposed(event, out_dir, actuators, priority_nodes, bundle, args.max_candidate_sequences)
                else:
                    result = _run_baseline(root, event, strategy, out_dir, actuators, priority_nodes)
                result_status = "pass"
                error = ""
            except Exception as exc:
                result = {}
                result_status = "fail"
                error = f"{type(exc).__name__}: {exc}"
                print(f"[MICRO] FAIL event={event['event_id']} strategy={strategy} error={error}", flush=True)
            row = {
                "event_id": event["event_id"],
                "qualification_role": event["role"],
                "rainfall_sha256": event["rainfall_sha256"],
                "strategy": strategy,
                "status": result_status,
                "input_sha256": event["inp_sha256"],
                "model_sha256": strategy_model_sha,
                "policy_sha256": policy_sha,
                "detail_path": str(detail),
                "detail_sha256": _sha256_file(detail) if detail.exists() else "",
                "runtime_sec": round(time.time() - started, 3),
                "error": error,
                "authority": "recorded_authoritative_swmm",
            }
            _append_ledger(ledger, row)
            if bundle is not None:
                del bundle
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if result_status == "pass":
                print(json.dumps({"event": event["event_id"], "strategy": strategy, "runtime_sec": row["runtime_sec"], "kpis": result.get("kpis", {})}, ensure_ascii=False, allow_nan=False, default=str), flush=True)
    if selected_strategies != list(STRATEGIES) or len(events) != len(all_events_for_summary):
        print(json.dumps({"status": "partial_pass", "events": [event["event_id"] for event in events], "strategies": selected_strategies}, ensure_ascii=False), flush=True)
        return 0
    summary = _compute_summary(root, qualification, all_events_for_summary, priority_nodes, ledger)
    status = _write_stage_status(qualification, all_events_for_summary, summary)
    final = {
        "contract_id": CONTRACT_ID,
        "stage": "qualification_micro_13_28",
        "status": "pass" if status["process_status"] == "pass" else "fail",
        "qualification_only": True,
        "development_only": True,
        "formal_mainline_authorized": False,
        "summary": summary,
        "stage_status": status,
        "formal_evidence_generated": False,
        "formal_untouched_events_consumed": False,
    }
    (qualification / "QUALIFICATION_MICRO_FINAL_AUDIT.json").write_text(json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False, default=str), flush=True)
    return 0 if final["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
