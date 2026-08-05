"""Authoritative Project6 V4.2 Formal closed-loop runtime.

This module is deliberately separate from the development/qualification micro
runner.  It uses the frozen V4.2 contracts directly:

* sparse-GAT (or true-state diagnostic) causal history;
 * shared four-reference trajectory surrogate;
 * canonical ``decide_pfvfirst_mpc`` PFV-constrained TFV-minimising selector;
* Engineering36 projected/read-back execution;
* explicit Formal baselines: No-control=all-open, All-close=all-zero,
  Internal=native SWMM rules, Hold=initial readback, EFD and Auto-RBC;
* no realised future hydraulic truth/rainfall in online decisions;
* authoritative outcome metrics are always recomputed from the SWMM trajectory.

The module does not select evaluation rainfalls.  The frozen Formal ledger and
``evaluation_inputs`` manifests are the only event authorities.  Missing local
INP inputs fail closed instead of silently falling back to a development case.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from sewerrtc.control.action_sequence_generator import generate_action_sequences
from sewerrtc.control.formal_baselines_f2 import (
    all_close_action,
    auto_rbc_action,
    equal_filling_degree_action,
    hold_previous_action,
    no_control_all_open_action,
)
from sewerrtc.control.pfvfirst_mpc_v42 import (
    EngineeringStatus,
    FrozenFallback,
    MPCandidate,
    MPCWeights,
    SafetyMargins,
    decide_pfvfirst_mpc,
)
from sewerrtc.network.influence_domain import build_priority_influence_domains
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import (
    _as_float,
    _enforce_actuator_semantics,
    _get_existing_links,
    _ids_from_container,
    _node_max_depth,
    _observed_action_from_links,
)
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast
from sewerrtc.v4.v42_node_safety import priority_depth_limits_m
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import (
    SURROGATE_ACTION_MAP_CONTRACT,
    _parse_inp_topology,
    build_surrogate_action_node_map,
)


FORMAL_OBJECTIVE_CONTRACT = "PROJECT6_V42_PFV_ONLY_TFV_MIN_MPC_V2"
FORMAL_FALLBACK_CONTRACT = "configs/v42_formal_fallback_contract.json"
FORMAL_STRATEGIES = (
    "Proposed",
    "EFD",
    "Auto-RBC",
    "All-close",
    "No-control",
    "Internal",
    "Hold",
)
STATE_STEP_SEC = 300
CONTROL_INTERVAL_MIN = 10.0
HORIZON_STEPS = 12
CONTROLLABLE_PREFIX_STEPS = 3
MAX_CHANGED_FACILITIES = 8
SENSOR_RATIO = 0.10
SENSOR_LAYOUT_SEED = 42


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


@dataclass(frozen=True)
class FormalEventInput:
    role: str
    event_id: str
    rainfall_sha256: str
    inp_path: Path
    rain_duration_min: int
    simulation_duration_min: int

    @property
    def input_sha256(self) -> str:
        return sha256_file(self.inp_path)


@dataclass
class FormalModelBundle:
    project_root: Path
    graph: Any
    device: torch.device
    gat: TemporalSparseGATReconstructorV42
    step2_models: list[MultiReferenceHydraulicSurrogate]
    step2_reports: list[dict[str, Any]]
    step1_report: dict[str, Any]
    step1_calibration: dict[str, Any]
    step2_calibration: dict[str, Any]
    sensor_indices: np.ndarray
    sensor_layout_sha256: str
    priority_indices: list[int]
    priority_to_actuators: pd.DataFrame
    priority_depth_limits: np.ndarray
    edge_index: torch.Tensor
    node_static: torch.Tensor
    link_static: torch.Tensor
    action_node_map: torch.Tensor
    surrogate_action_node_map: torch.Tensor
    fallback_contract_sha256: str

    @property
    def gat_model_sha256(self) -> str:
        return str(self.step1_report["gat_model_sha256"])

    @property
    def surrogate_model_sha256(self) -> str:
        # The paper evidence is anchored to the primary seed42 model.  The
        # ensemble members remain separately listed in Step2 calibration.
        primary = next(
            (r for r in self.step2_reports if int(r.get("seed", -1)) == 42),
            self.step2_reports[0],
        )
        return str(primary["surrogate_model_sha256"])


def _build_priority_influence_map(
    project_root: str | Path, graph: Any, actuators: pd.DataFrame
) -> pd.DataFrame:
    """Use only physically local actuator domains for Formal candidate generation."""
    inp_path = Path(project_root) / "data/wuhan_v8_storage_retrofit.inp"
    _, links = _parse_inp_topology(inp_path)
    endpoint_by_link = {
        str(row["link_id"]).casefold(): {
            "from_node": str(row["from_node"]),
            "to_node": str(row["to_node"]),
        }
        for _, row in links.iterrows()
        if str(row.get("link_id", "")).strip()
    }
    enriched = actuators.copy()
    for column in ("from_node", "to_node"):
        values = []
        for _, row in enriched.iterrows():
            current = str(row.get(column, ""))
            if current and current.casefold() not in {"nan", "none"}:
                values.append(current)
                continue
            link_id = str(row.get("link_id", row.get("actuator_id", ""))).casefold()
            values.append(str(endpoint_by_link.get(link_id, {}).get(column, "")))
        enriched[column] = values
    priority_nodes = [
        str(graph.node_ids[i])
        for i in get_pfv_core_node_indices(list(graph.node_ids))
    ]
    _, influence = build_priority_influence_domains(
        links,
        enriched,
        priority_nodes,
        include_global_storage_controls=False,
        include_global_regulators=False,
        include_global_pumps=False,
    )
    if influence.empty:
        raise RuntimeError("Formal candidate generation requires a non-empty priority influence map")
    return influence.reset_index(drop=True)


def load_formal_event_inputs(
    project_root: str | Path,
    *,
    role: str,
    evaluation_plan: str | Path | None = None,
    input_manifest: str | Path | None = None,
) -> list[FormalEventInput]:
    """Resolve one frozen held-out role to local authoritative INP files.

    ``evaluation_inputs/<role>_case_manifest.csv`` is a deployment manifest,
    not a scientific selector.  Its rainfall/event identities must equal the
    already-frozen plan.  Required columns are ``event_id``, ``rainfall_sha256``,
    ``inp_path``, ``rain_duration_min`` and ``simulation_duration_min``.
    """
    root = Path(project_root)
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    plan_path = Path(evaluation_plan) if evaluation_plan else formal / "evaluation_plan" / f"{role}_plan.json"
    manifest_path = Path(input_manifest) if input_manifest else formal / "evaluation_inputs" / f"{role}_case_manifest.csv"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Formal local event-input manifest missing: {manifest_path}. "
            "Create it from the frozen evaluation plan; do not substitute a development event."
        )
    plan = _read_json(plan_path)
    planned_rows = plan.get("events", [])
    planned = {
        (str(x.get("event_id", "")), str(x.get("rainfall_sha256", "")))
        for x in planned_rows
    }
    if not planned or any(not a or not b for a, b in planned):
        raise RuntimeError(f"frozen {role} plan has missing event/rainfall identity")
    frame = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "event_id",
        "rainfall_sha256",
        "inp_path",
        "rain_duration_min",
        "simulation_duration_min",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Formal event-input manifest missing columns: {missing}")
    observed = set(
        zip(frame["event_id"].astype(str), frame["rainfall_sha256"].astype(str))
    )
    if observed != planned:
        raise RuntimeError(
            f"Formal {role} event-input manifest differs from frozen plan: "
            f"missing={len(planned-observed)} extra={len(observed-planned)}"
        )
    if frame.duplicated(["event_id", "rainfall_sha256"]).any():
        raise RuntimeError(f"Formal {role} event-input manifest contains duplicate identities")
    result: list[FormalEventInput] = []
    for row in frame.to_dict("records"):
        inp = Path(str(row["inp_path"]))
        if not inp.is_absolute():
            inp = root / inp
        if not inp.exists():
            raise FileNotFoundError(inp)
        rain_duration = int(row["rain_duration_min"])
        simulation_duration = int(row["simulation_duration_min"])
        if simulation_duration < rain_duration or simulation_duration < 240:
            raise RuntimeError(
                f"Formal event {row['event_id']} has insufficient simulation duration"
            )
        result.append(
            FormalEventInput(
                role=role,
                event_id=str(row["event_id"]),
                rainfall_sha256=str(row["rainfall_sha256"]),
                inp_path=inp.resolve(),
                rain_duration_min=rain_duration,
                simulation_duration_min=simulation_duration,
            )
        )
    return sorted(result, key=lambda x: (x.rainfall_sha256, x.event_id))


@lru_cache(maxsize=4)
def load_actuators(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root)
    path = root / "outputs/audit_v8_storage_variablepump/actuator_table.csv"
    frame = pd.read_csv(path, low_memory=False)
    if "action_index" in frame:
        frame = frame.sort_values("action_index")
    frame = frame.reset_index(drop=True)
    if len(frame) != 36 or frame["actuator_id"].astype(str).nunique() != 36:
        raise RuntimeError("Formal runtime requires exactly 36 unique Engineering36 actuators")
    return frame


@lru_cache(maxsize=4)
def _load_baseline_graph_assets(project_root: str | Path) -> dict[str, Any]:
    """Load only the numpy/pandas graph fields needed by SWMM baselines."""
    from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology

    return _load_graph_topology(Path(project_root))


@lru_cache(maxsize=4)
def load_model_bundle(
    project_root: str | Path,
    device_name: str = "auto",
    *,
    step2_calibration_path: str | Path | None = None,
) -> FormalModelBundle:
    import torch

    from sewerrtc.models.temporal_sparse_gat_v42 import (
        TemporalSparseGATReconstructorV42,
    )
    from sewerrtc.v4.models_v42.hydraulic_multi_reference import (
        MultiReferenceHydraulicSurrogate,
    )
    from sewerrtc.v4.v42_step1_dataset import _sensor_layout, load_graph_assets

    root = Path(project_root)
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    graph = load_graph_assets(root)
    device = torch.device(
        device_name
        if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    step1_dir = formal / "step1/seed_42"
    step1_report = _read_json(step1_dir / "formal_step1_report.json")
    gat = TemporalSparseGATReconstructorV42(
        n_nodes=graph.n_nodes,
        n_facilities=graph.n_facilities,
        node_static_dim=graph.node_static.shape[1],
        link_static_dim=graph.link_static.shape[1],
        hidden_dim=128,
        heads=4,
        gat_layers=3,
    ).to(device)
    gat.load_state_dict(
        torch.load(step1_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    gat.eval()

    causal_model_root = formal / "step2/models_action_causal_controlaware_v2"
    preferred_model_root = formal / "step2/models_action_diffusion_v1"
    legacy_model_root = formal / "step2/models"
    model_root = (
        causal_model_root
        if all(
            (causal_model_root / f"seed_{seed}" / "formal_step2_report.json").exists()
            for seed in (17, 42, 73)
        )
        else (
            preferred_model_root
            if all(
                (preferred_model_root / f"seed_{seed}" / "formal_step2_report.json").exists()
                for seed in (17, 42, 73)
            )
            else legacy_model_root
        )
    )
    step2_models: list[MultiReferenceHydraulicSurrogate] = []
    step2_reports: list[dict[str, Any]] = []
    for seed in (17, 42, 73):
        model_dir = model_root / f"seed_{seed}"
        report = _read_json(model_dir / "formal_step2_report.json")
        if report.get("surrogate_action_map_contract") != SURROGATE_ACTION_MAP_CONTRACT:
            raise RuntimeError(
                "Formal Step2 checkpoint uses the endpoint-only action map; "
                "retrain with the action-influence map before Core RTC"
            )
        # Core RTC supplies only the checkpoint native-rule readback persisted
        # through H12.  Old reports without this provenance used future action
        # schedules and must fail closed before SWMM execution.
        if report.get("dynamic_internal_action_input_contract") != (
            "causal_current_native_rule_readback_persistence"
        ) or report.get("future_dynamic_internal_action_input_used") is not False:
            raise RuntimeError(
                "Formal Step2 checkpoint is not compatible with the causal "
                "Dynamic-Internal input contract; retrain before Core RTC: "
                f"{model_dir}"
            )
        step2_reports.append(report)
        model = MultiReferenceHydraulicSurrogate(
            n_nodes=graph.n_nodes,
            n_facilities=graph.n_facilities,
            state_feature_dim=1,
            static_feature_dim=graph.node_static.shape[1],
            hidden_dim=64,
            gat_heads=4,
            gat_layers=3,
            horizon=HORIZON_STEPS,
        ).to(device)
        model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
        )
        model.eval()
        step2_models.append(model)

    contracts = {str(x.get("step2_target_contract", "")) for x in step2_reports}
    if len(contracts) != 1 or "" in contracts:
        raise RuntimeError(f"Formal Step2 ensemble target contracts differ: {contracts}")
    step1_calibration = _read_json(
        formal / "calibration/STEP1_UNCERTAINTY_OOD_CALIBRATION.json"
    )
    if step2_calibration_path is None:
        calibration_path = (
            formal / "calibration/PFV_ONLY_SAFETY_CALIBRATION_CAUSAL_V2.json"
            if model_root == causal_model_root
            else formal / "calibration/PFV_ONLY_SAFETY_CALIBRATION.json"
        )
        if not calibration_path.exists():
            # Current simple-core runs reuse the frozen Fresh Calibration12
            # artifact when the legacy Formal path is absent.
            fresh_path = formal / "pfv_only_v2/FRESH_PFV_ONLY_SAFETY_CALIBRATION.json"
            if fresh_path.exists():
                calibration_path = fresh_path
    else:
        calibration_path = Path(step2_calibration_path)
    step2_calibration = _read_json(calibration_path)
    if step2_calibration_path is not None and (
        step2_calibration.get("development_only") is not True
        or step2_calibration.get("formal_mainline_authorized") is True
    ):
        raise RuntimeError(
            "an explicit Step2 calibration override must be marked development_only "
            "and cannot authorize Formal mainline"
        )
    if step1_calibration.get("status") != "pass" or step2_calibration.get("status") != "pass":
        raise RuntimeError("Formal runtime requires passed Step1 and Step2 calibration")
    if int(step1_calibration.get("calibration_rainfall_group_count", 0)) != 12:
        raise RuntimeError("Formal runtime requires complete Calibration12 for Step1")
    if int(step2_calibration.get("calibration_rainfall_group_count", 0)) != 12:
        raise RuntimeError("Formal runtime requires complete Calibration12 for Step2")
    expected_model_hashes = {
        str(seed): str(report.get("surrogate_model_sha256", ""))
        for seed, report in zip((17, 42, 73), step2_reports)
    }
    calibration_model_hashes = {
        str(seed): str(value)
        for seed, value in dict(step2_calibration.get("model_hashes", {})).items()
    }
    if expected_model_hashes != calibration_model_hashes:
        raise RuntimeError(
            "Formal Step2 safety calibration does not match the selected model "
            f"bundle: {calibration_path}"
        )

    _, sensor_indices, sensor_sha = _sensor_layout(
        graph.n_nodes, SENSOR_RATIO, SENSOR_LAYOUT_SEED
    )
    if str(step1_report.get("sensor_layout_sha256", "")) != sensor_sha:
        raise RuntimeError("Formal runtime sensor layout differs from Formal Step1")
    priority_indices = get_pfv_core_node_indices(list(graph.node_ids))
    surrogate_action_map = build_surrogate_action_node_map(graph)
    if not np.count_nonzero(surrogate_action_map[:, priority_indices].sum(axis=0)) == len(priority_indices):
        raise RuntimeError("surrogate action map does not cover every PFV_CORE8 node")
    priority_to_actuators = _build_priority_influence_map(root, graph, load_actuators(root))
    limits = priority_depth_limits_m(root, priority_indices)
    fallback_path = root / FORMAL_FALLBACK_CONTRACT
    if not fallback_path.exists():
        raise FileNotFoundError(fallback_path)
    return FormalModelBundle(
        project_root=root,
        graph=graph,
        device=device,
        gat=gat,
        step2_models=step2_models,
        step2_reports=step2_reports,
        step1_report=step1_report,
        step1_calibration=step1_calibration,
        step2_calibration=step2_calibration,
        sensor_indices=sensor_indices,
        sensor_layout_sha256=sensor_sha,
        priority_indices=priority_indices,
        priority_to_actuators=priority_to_actuators,
        priority_depth_limits=np.asarray(limits, dtype=np.float64),
        edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
        node_static=torch.as_tensor(graph.node_static, dtype=torch.float32, device=device),
        link_static=torch.as_tensor(graph.link_static, dtype=torch.float32, device=device),
        action_node_map=torch.as_tensor(
            graph.action_node_map, dtype=torch.float32, device=device
        ),
        surrogate_action_node_map=torch.as_tensor(
            surrogate_action_map, dtype=torch.float32, device=device
        ),
        fallback_contract_sha256=sha256_file(fallback_path),
    )


def _frame(
    sim: Any,
    node_objs: Mapping[str, Any],
    rain_obj: Any,
    link_objs: Mapping[str, Any],
    actuator_ids: list[str],
    actuators: pd.DataFrame,
) -> dict[str, Any]:
    elapsed = (sim.current_time - sim.start_time).total_seconds() / 60.0
    return {
        "elapsed_min": float(elapsed),
        "depth": np.asarray(
            [_as_float(getattr(node_objs[n], "depth", np.nan), np.nan) for n in node_objs],
            dtype=np.float32,
        ),
        "rain": _as_float(getattr(rain_obj, "rainfall", 0.0), 0.0)
        if rain_obj is not None
        else 0.0,
        "action": _observed_action_from_links(
            dict(link_objs), actuator_ids, actuators
        ).astype(np.float32),
    }


def _record_row(
    *,
    frame: Mapping[str, Any],
    event_id: str,
    strategy: str,
    command: np.ndarray,
    readback: np.ndarray,
    node_objs: Mapping[str, Any],
    link_objs: Mapping[str, Any],
    actuator_ids: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": event_id,
        "policy_id": strategy,
        "elapsed_min": float(frame["elapsed_min"]),
        "rainfall_mm_h": float(frame["rain"]),
        "phase": "formal_authoritative",
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


def _native_controls_contract(inp_path: str | Path) -> dict[str, Any]:
    """Return a stable audit of the source INP native-control section."""
    lines = Path(inp_path).read_text(encoding="utf-8", errors="replace").splitlines()
    in_controls = False
    section: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_controls = stripped[1:-1].strip().upper() == "CONTROLS"
            continue
        if in_controls:
            section.append(raw.rstrip())
    normalized = "\n".join(section).strip()
    rules = [line for line in section if line.strip() and not line.lstrip().startswith(";")]
    return {
        "section_present": any(
            line.strip().upper() == "[CONTROLS]" for line in lines
        ),
        "rule_count": len(rules),
        "contract_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def _audit_baseline_contract(
    detail: pd.DataFrame, strategy: str, inp_path: str | Path
) -> dict[str, Any]:
    """Audit actual baseline command/readback instead of metadata claims."""
    command_cols = [str(c) for c in detail.columns if str(c).startswith("a:")]
    readback_cols = [str(c) for c in detail.columns if str(c).startswith("setting:")]
    command_ids = [c[2:] for c in command_cols]
    readback_ids = [c[8:] for c in readback_cols]
    columns_match = command_ids == readback_ids and bool(command_ids)
    command = (
        detail[command_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if columns_match
        else np.empty((0, 0), dtype=float)
    )
    readback = (
        detail[readback_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if columns_match
        else np.empty((0, 0), dtype=float)
    )
    readback_finite = bool(readback.size and np.isfinite(readback).all())
    target_write_verified = bool(
        columns_match
        and command.size
        and np.isfinite(command).all()
        and readback_finite
        and np.allclose(command, readback, atol=1.0e-6, rtol=0.0)
    )
    expected = None
    if strategy == "No-control":
        expected = 1.0
    elif strategy == "All-close":
        expected = 0.0
    physical_setting_verified = bool(
        expected is not None
        and readback_finite
        and np.allclose(readback, expected, atol=1.0e-6, rtol=0.0)
    )
    native = _native_controls_contract(inp_path)
    internal_native_rules_preserved = bool(
        strategy != "Internal" or native["section_present"]
    )
    baseline_contract_pass = bool(
        readback_finite
        and internal_native_rules_preserved
        and (
            physical_setting_verified
            if expected is not None
            else strategy == "Internal"
        )
    )
    return {
        "readback_finite": readback_finite,
        "target_write_verified": target_write_verified
        if strategy != "Internal"
        else None,
        "physical_setting_verified": physical_setting_verified
        if expected is not None
        else None,
        "no_control_all_open_contract": physical_setting_verified
        if strategy == "No-control"
        else None,
        "all_close_zero_contract": physical_setting_verified
        if strategy == "All-close"
        else None,
        "internal_native_rules_preserved": internal_native_rules_preserved,
        "source_inp_sha256": sha256_file(inp_path),
        "native_controls_section_present": native["section_present"],
        "native_rule_count": native["rule_count"],
        "native_rule_contract_sha256": native["contract_sha256"],
        "baseline_contract_pass": baseline_contract_pass,
    }


def _is_decision_time(elapsed_min: float) -> bool:
    return elapsed_min >= 120.0 - 1.0e-6 and abs(
        elapsed_min / CONTROL_INTERVAL_MIN - round(elapsed_min / CONTROL_INTERVAL_MIN)
    ) <= 1.0e-6


def reconstruct_history(
    frames: Sequence[Mapping[str, Any]],
    bundle: FormalModelBundle,
    *,
    state_source: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return 13 states at t-60..t and their uncertainty.

    Sparse-GAT mode needs 25 five-minute plant frames (t-120..t) because each
    of the thirteen reconstructed anchors uses its own preceding 60-min sensor
    history.  True-state mode is diagnostic only and uses the last 13 SWMM
    states directly.
    """
    import torch

    if state_source == "true_state":
        if len(frames) < 13:
            raise RuntimeError("true-state diagnostic requires 13 historical frames")
        history = np.stack([np.asarray(x["depth"], np.float32) for x in frames[-13:]])
        return history, np.zeros_like(history), 0.0
    if state_source != "gat_sparse_reconstruction":
        raise ValueError(f"unsupported Formal state source: {state_source}")
    if len(frames) < 25:
        last_elapsed = frames[-1].get("elapsed_min") if frames else None
        raise RuntimeError(
            "sparse-GAT causal history requires t-120..t plant frames; "
            f"observed_frames={len(frames)} last_elapsed_min={last_elapsed}"
        )
    anchors = list(range(len(frames) - 13, len(frames)))
    n_nodes = bundle.graph.n_nodes
    sparse = np.zeros((13, 13, n_nodes), np.float32)
    mask = np.zeros_like(sparse)
    rainfall = np.zeros((13, 13), np.float32)
    actions = np.zeros((13, 13, bundle.graph.n_facilities), np.float32)
    for i, anchor in enumerate(anchors):
        window = frames[anchor - 12 : anchor + 1]
        if len(window) != 13:
            raise RuntimeError("incomplete 13-frame causal anchor window")
        for j, item in enumerate(window):
            depth = np.asarray(item["depth"], np.float32)
            sparse[i, j, bundle.sensor_indices] = depth[bundle.sensor_indices]
            mask[i, j, bundle.sensor_indices] = 1.0
            rainfall[i, j] = float(item["rain"])
            actions[i, j] = np.asarray(item["action"], np.float32)
    with torch.inference_mode():
        out = bundle.gat(
            sparse_depth_history=torch.as_tensor(sparse, device=bundle.device),
            sensor_mask_history=torch.as_tensor(mask, device=bundle.device),
            rainfall_history=torch.as_tensor(rainfall, device=bundle.device),
            historical_actions=torch.as_tensor(actions, device=bundle.device),
            node_static=bundle.node_static,
            link_static=bundle.link_static,
            edge_index=bundle.edge_index,
            action_node_map=bundle.action_node_map,
        )
    mean = out.depth_mean.detach().cpu().numpy().astype(np.float32)
    std = out.depth_std.detach().cpu().numpy().astype(np.float32)
    unobserved = np.ones(n_nodes, dtype=bool)
    unobserved[bundle.sensor_indices] = False
    ood_score = float(np.mean(std[:, unobserved])) if np.any(unobserved) else 0.0
    return mean, std, ood_score


def _binary_ids(actuators: pd.DataFrame) -> set[str]:
    return {"ADD301.2", "ADD301.3"}


def project_candidate_sequence(
    sequence: np.ndarray,
    current_action: np.ndarray,
    actuators: pd.DataFrame,
) -> tuple[np.ndarray, EngineeringStatus, int, bool]:
    """Project a candidate into the frozen three-step Engineering36 envelope."""
    ids = actuators["actuator_id"].astype(str).tolist()
    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    seq = np.asarray(sequence, dtype=np.float32).copy()
    if seq.shape != (HORIZON_STEPS, len(ids)) or current.size != len(ids):
        return seq, EngineeringStatus(False, False, False, False, False), 999, False
    if not np.isfinite(seq).all() or not np.isfinite(current).all():
        return seq, EngineeringStatus(False, False, False, False, False), 999, False
    seq = np.clip(seq, 0.0, 1.0)
    # Only the first 30 min may be controlled. The remaining H120 action is the
    # frozen hold anchor, so the optimizer cannot claim benefit from actions it
    # is not allowed to execute/replan yet.
    seq[CONTROLLABLE_PREFIX_STEPS:] = current[None, :]
    for k in range(CONTROLLABLE_PREFIX_STEPS):
        seq[k] = _enforce_actuator_semantics(
            seq[k], ids, actuators, "binary_unless_verified", ["add350.1"]
        )
    bounds = bool(np.all((seq >= -1e-7) & (seq <= 1.0 + 1e-7)))
    binary = True
    for aid in _binary_ids(actuators):
        if aid in ids:
            col = seq[:CONTROLLABLE_PREFIX_STEPS, ids.index(aid)]
            binary = binary and bool(np.all(np.isclose(col, 0.0) | np.isclose(col, 1.0)))
    rate = True
    previous = current.copy()
    for k in range(CONTROLLABLE_PREFIX_STEPS):
        delta = np.abs(seq[k] - previous)
        for i, aid in enumerate(ids):
            if aid in _binary_ids(actuators):
                continue
            limit = 0.15 if aid == "add350.1" else 0.12
            if float(delta[i]) > limit + 1e-6:
                rate = False
        previous = seq[k]
    changed = np.any(
        np.abs(seq[:CONTROLLABLE_PREFIX_STEPS] - current[None, :]) > 1e-6,
        axis=0,
    )
    k_count = int(changed.sum())
    # Dwell: binary facilities may change at most once inside the controllable
    # 30-min prefix. Runtime memory still protects cross-decision minimum dwell.
    dwell = True
    for aid in _binary_ids(actuators):
        if aid in ids:
            col = np.concatenate([[current[ids.index(aid)]], seq[:3, ids.index(aid)]])
            dwell = dwell and int(np.sum(np.abs(np.diff(col)) > 1e-6)) <= 1
    # Storage inlet/outlet interlock: do not move both directions for one tank
    # in the same candidate prefix.
    interlock = True
    if "storage_node" in actuators and "storage_control_type" in actuators:
        work = actuators.reset_index(drop=True)
        for node in sorted(set(work["storage_node"].fillna("").astype(str)) - {""}):
            inlet = work.index[
                work["storage_node"].astype(str).eq(node)
                & work["storage_control_type"].astype(str).eq("storage_inlet")
            ].to_numpy(int)
            outlet = work.index[
                work["storage_node"].astype(str).eq(node)
                & work["storage_control_type"].astype(str).eq("storage_outlet")
            ].to_numpy(int)
            if inlet.size and outlet.size and changed[inlet].any() and changed[outlet].any():
                interlock = False
                break
    engineering = EngineeringStatus(
        bounds=bool(bounds and binary),
        rate=rate,
        ramp=rate,
        dwell=dwell,
        interlock=interlock,
    )
    executable = bool(engineering.passed and k_count <= MAX_CHANGED_FACILITIES)
    return seq.astype(np.float32), engineering, k_count, executable


@lru_cache(maxsize=4)
def _uncertainty_normalizers(manifest_path_value: str) -> tuple[float, float, float]:
    manifest_path = Path(manifest_path_value)
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Step2 calibration manifest is required to reconstruct the frozen uncertainty-score normalization"
        )
    frame = pd.read_parquet(manifest_path) if manifest_path.suffix.lower() == ".parquet" else pd.read_csv(manifest_path)
    scales = []
    for key in ("pfv_delta", "tfv_delta", "peak_delta"):
        if key not in frame:
            raise KeyError(f"calibration manifest missing {key}")
        scale = float(np.std(pd.to_numeric(frame[key], errors="raise").to_numpy(float)))
        scales.append(max(scale, 1e-6))
    return tuple(scales)  # type: ignore[return-value]


def predict_and_decide(
    *,
    bundle: FormalModelBundle,
    actuators: pd.DataFrame,
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    internal_current_action: np.ndarray,
    gat_ood_score: float,
    max_candidate_sequences: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run ensemble H120 prediction and the canonical PFV-budgeted selector."""
    import torch

    ids = actuators["actuator_id"].astype(str).tolist()
    base = np.asarray(current_action, np.float32)
    generated = generate_action_sequences(
        base,
        actuators,
        HORIZON_STEPS,
        max_delta=0.12,
        include_hold=True,
        max_sequences=max_candidate_sequences,
        group_limit=8,
        reference_sequence=np.repeat(base[None, :], HORIZON_STEPS, axis=0),
        priority_to_actuators=bundle.priority_to_actuators,
    )
    projected: list[tuple[str, np.ndarray, EngineeringStatus, int, bool]] = []
    for item in generated:
        seq, engineering, k_count, executable = project_candidate_sequence(
            np.asarray(item["sequence"], np.float32), base, actuators
        )
        projected.append((str(item.get("label", f"candidate_{len(projected)}")), seq, engineering, k_count, executable))
    if not projected:
        seq = np.repeat(base[None, :], HORIZON_STEPS, axis=0)
        projected = [("hold_only", seq, EngineeringStatus(True, True, True, True, True), 0, True)]

    candidate = np.stack([x[1] for x in projected])
    n = len(candidate)
    nc = np.repeat(np.ones((1, HORIZON_STEPS, len(ids)), np.float32), n, axis=0)
    internal = np.repeat(
        np.asarray(internal_current_action, np.float32)[None, None, :],
        n,
        axis=0,
    )
    internal = np.repeat(internal, HORIZON_STEPS, axis=1)
    hold = np.repeat(base[None, None, :], n, axis=0)
    hold = np.repeat(hold, HORIZON_STEPS, axis=1)
    history = np.repeat(np.asarray(state_history, np.float32)[None, :, :], n, axis=0)
    hist_actions = np.repeat(
        np.asarray(historical_actions, np.float32)[None, :, :], n, axis=0
    )
    rain = np.repeat(
        np.asarray(rainfall_forecast, np.float32)[None, :HORIZON_STEPS], n, axis=0
    )
    # Create the batch tensors once per decision. Recreating the same H120
    # inputs for each ensemble member adds needless CPU->GPU copies.
    state_history_t = torch.as_tensor(history, device=bundle.device)
    historical_actions_t = torch.as_tensor(hist_actions, device=bundle.device)
    rainfall_t = torch.as_tensor(rain, device=bundle.device)
    candidate_t = torch.as_tensor(candidate, device=bundle.device)
    no_control_t = torch.as_tensor(nc, device=bundle.device)
    internal_t = torch.as_tensor(internal, device=bundle.device)
    hold_t = torch.as_tensor(hold, device=bundle.device)
    priority = torch.as_tensor(bundle.priority_indices, dtype=torch.long, device=bundle.device)
    predictions: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for model in bundle.step2_models:
            out = model(
                state_history=state_history_t,
                historical_actions=historical_actions_t,
                rainfall_forecast=rainfall_t,
                action_candidate=candidate_t,
                action_no_control=no_control_t,
                action_dynamic_internal=internal_t,
                action_hold_previous=hold_t,
                edge_index=bundle.edge_index,
                node_static=bundle.node_static,
                action_node_map=bundle.surrogate_action_node_map,
                priority_node_indices=priority,
            )
            predictions.append(
                {
                    "pfv_delta": out["pfv_delta"].detach().cpu().numpy(),
                    "tfv_delta": out["tfv_delta"].detach().cpu().numpy(),
                    "peak_delta": out["peak_delta"].detach().cpu().numpy(),
                    "no_control_pfv": out["kpi_no_control"]["pfv_m3"].detach().cpu().numpy(),
                    "depth": out["branches"]["candidate"]["node_depth"][:, :, priority].detach().cpu().numpy(),
                }
            )
    stack = {
        key: np.stack([p[key] for p in predictions], axis=0)
        for key in predictions[0]
    }
    mean = {key: value.mean(axis=0) for key, value in stack.items()}
    std = {key: value.std(axis=0, ddof=1) for key, value in stack.items()}
    z = float(bundle.step2_calibration.get("confidence_z", np.nan))
    if not np.isfinite(z):
        raise RuntimeError("Formal Step2 calibration confidence_z is not finite")
    scales = _uncertainty_normalizers(
        str(bundle.step2_calibration.get("calibration_manifest", ""))
    )
    # V2 uses uncertainty only for the one-sided PFV UCB. TFV/Peak uncertainty
    # remains available as diagnostic output but cannot reject or rank actions.
    uncertainty_score = np.abs(std["pfv_delta"] / scales[0])
    uncertainty_limit = float(bundle.step2_calibration.get("uncertainty_limit_99", np.nan))
    ood_limit = float(bundle.step1_calibration.get("ood_limit_99", np.nan))

    candidates: list[MPCandidate] = []
    for i, (label, seq, engineering, k_count, executable) in enumerate(projected):
        depth_ucb = mean["depth"][i] + z * std["depth"][i]
        candidates.append(
            MPCandidate(
                candidate_id=label,
                action_sequence=seq,
                pfv_delta_ucb_m3=float(mean["pfv_delta"][i] + z * std["pfv_delta"][i]),
                peak_delta_ucb_m3s=float(mean["peak_delta"][i] + z * std["peak_delta"][i]),
                tfv_delta_di_m3=float(mean["tfv_delta"][i]),
                action_cost=float(np.mean(np.abs(seq[:3] - base[None, :]))),
                terminal_cost=float(np.mean(np.abs(seq[2] - base))),
                uncertainty_cost=float(uncertainty_score[i]),
                changed_facilities=k_count,
                engineering=engineering,
                uncertainty_pass=bool(uncertainty_score[i] <= uncertainty_limit),
                ood_pass=bool(gat_ood_score <= ood_limit),
                executable=executable,
                pfv_no_control_m3=float(mean["no_control_pfv"][i]),
                priority_depth_ucb_m=tuple(depth_ucb.reshape(-1).astype(float)),
                priority_depth_limit_m=tuple(
                    np.tile(bundle.priority_depth_limits, HORIZON_STEPS).astype(float)
                ),
                metadata={
                    "ensemble_seed_count": len(bundle.step2_models),
                    "confidence_z": z,
                    "dynamic_internal_action_forecast": "causal_current_native_rule_setting_persistence",
                    "no_control_action_forecast": "all_engineering36_open",
                    "hold_action_forecast": "current_readback_persistence",
                },
            )
        )
    fallback_seq = np.repeat(base[None, :], HORIZON_STEPS, axis=0)
    decision = decide_pfvfirst_mpc(
        candidates=candidates,
        fallback=FrozenFallback(
            fallback_id="frozen_hold_readback",
            action_sequence=fallback_seq,
            contract_hash=bundle.fallback_contract_sha256,
            legal=True,
        ),
        margins=SafetyMargins(),
        weights=MPCWeights(),
        expected_fallback_contract_hash=bundle.fallback_contract_sha256,
    )
    audit_rows = [
        {
            "candidate_id": a.candidate_id,
            "safe": a.safe,
            "rejection_reasons": list(a.rejection_reasons),
            "objective": a.objective,
            "pfv_allowance_m3": a.pfv_allowance_m3,
            "maximum_priority_depth_exceedance_m": a.maximum_priority_depth_exceedance_m,
        }
        for a in decision.audits
    ]
    return decision.execute_action.astype(np.float32), {
        "selected_id": decision.selected_id,
        "used_fallback": decision.used_fallback,
        "reason": decision.reason,
        "selected_objective_score": decision.objective,
        "candidate_audits": audit_rows,
        "gat_ood_score": float(gat_ood_score),
        "ood_limit_99": ood_limit,
        "uncertainty_limit_99": uncertainty_limit,
        "canonical_pfvfirst_mpc_v42": True,
        "control_objective_contract": FORMAL_OBJECTIVE_CONTRACT,
        "pfv_budget_applied": True,
        "uncertainty_used_for_pfv_ucb": True,
        "priority_depth_hard_gate": False,
        "global_peak_hard_gate": False,
        "global_peak_objective_term": False,
        "peak_penalty_weight": 0.0,
        "action_penalty_weight": 0.0,
        "terminal_penalty_weight": 0.0,
        "uncertainty_penalty_weight": 0.0,
        "independent_OOD_gate": False,
        "independent_uncertainty_gate": False,
        "objective": "minimize_TFV_subject_to_PFV_budget",
        "future_hydraulic_truth_used_online": False,
        "realized_future_rainfall_used_online": False,
        "dynamic_internal_future_truth_used_online": False,
    }


def _node_full_depths(
    node_ids: list[str],
    node_objs: Mapping[str, Any],
    *,
    inp_path: str | Path | None = None,
) -> np.ndarray:
    """Return deterministic physical capacities for filling-degree baselines.

    PySWMM versions do not expose the same node capacity attribute set.  The
    frozen INP is the authority when available; live attributes are only the
    compatibility fallback for callers without an INP path.
    """
    if inp_path is not None:
        parsed_nodes, _ = _parse_inp_topology(Path(inp_path))
        parsed = {
            str(row.node_id): float(row.max_depth)
            for row in parsed_nodes.itertuples(index=False)
            if np.isfinite(float(row.max_depth)) and float(row.max_depth) > 1.0e-6
        }
        if all(str(node_id) in parsed for node_id in node_ids):
            return np.asarray([parsed[str(node_id)] for node_id in node_ids], dtype=np.float32)
    return np.asarray([_node_max_depth(node_objs[n]) for n in node_ids], dtype=np.float32)


def _baseline_desired_action(
    *,
    strategy: str,
    current_action: np.ndarray,
    initial_hold: np.ndarray,
    node_depth: np.ndarray,
    node_full_depth: np.ndarray,
    action_node_map: np.ndarray,
    binary_indices: list[int],
) -> np.ndarray:
    if strategy == "No-control":
        return no_control_all_open_action(36).desired_action
    if strategy == "All-close":
        return all_close_action(36).desired_action
    if strategy == "Hold":
        return hold_previous_action(initial_hold).desired_action
    if strategy == "EFD":
        return equal_filling_degree_action(
            node_depth=node_depth,
            node_full_depth=node_full_depth,
            action_node_map=action_node_map,
            anchor_action=current_action,
            binary_indices=binary_indices,
        ).desired_action
    if strategy == "Auto-RBC":
        return auto_rbc_action(
            node_depth=node_depth,
            node_full_depth=node_full_depth,
            action_node_map=action_node_map,
            anchor_action=current_action,
            binary_indices=binary_indices,
        ).desired_action
    raise KeyError(strategy)


def run_baseline_event(
    event: FormalEventInput,
    *,
    strategy: str,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one explicit Formal baseline against authoritative SWMM."""
    if strategy not in FORMAL_STRATEGIES or strategy == "Proposed":
        raise ValueError(f"unsupported Formal baseline: {strategy}")
    from pyswmm import Links, Nodes, RainGages, Simulation

    root = Path(project_root)
    actuators = load_actuators(root)
    graph = _load_baseline_graph_assets(root)
    ids = actuators["actuator_id"].astype(str).tolist()
    if ids != [str(x) for x in graph["facility_ids"]]:
        raise RuntimeError("Formal baseline actuator order differs from graph order")
    priority_nodes = [
        str(graph["node_ids"][i])
        for i in get_pfv_core_node_indices(list(graph["node_ids"]))
    ]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "detail.csv"
    records: list[dict[str, Any]] = []
    start = time.time()
    with Simulation(str(event.inp_path)) as sim:
        sim.step_advance(STATE_STEP_SEC)
        sim.start()
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        node_objs = {nid: nodes[nid] for nid in node_ids}
        actuator_ids, link_objs = _get_existing_links(links, ids)
        if actuator_ids != ids:
            raise RuntimeError("Formal baseline INP does not expose Engineering36")
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        current = _observed_action_from_links(link_objs, ids, actuators)
        initial_hold = current.copy()
        binary_indices = [ids.index(x) for x in ("ADD301.2", "ADD301.3") if x in ids]
        full_depth = _node_full_depths(node_ids, node_objs, inp_path=event.inp_path)
        command = current.copy()
        # Force explicit non-native Formal baselines before the first hydraulic
        # advance. Internal alone leaves the frozen SWMM rule engine untouched.
        if strategy != "Internal":
            command = _baseline_desired_action(
                strategy=strategy,
                current_action=current,
                initial_hold=initial_hold,
                node_depth=np.asarray([_as_float(node_objs[n].depth, 0.0) for n in node_ids]),
                node_full_depth=full_depth,
                action_node_map=np.asarray(graph["action_node_map"], np.float32),
                binary_indices=binary_indices,
            )
            command = _enforce_actuator_semantics(
                command, ids, actuators, "binary_unless_verified", ["add350.1"]
            )
            for i, aid in enumerate(ids):
                link_objs[aid].target_setting = float(command[i])
        for _ in sim:
            pre = _frame(sim, node_objs, rain_obj, link_objs, ids, actuators)
            elapsed = float(pre["elapsed_min"])
            if strategy == "Internal":
                command = _observed_action_from_links(link_objs, ids, actuators)
            elif abs(elapsed / CONTROL_INTERVAL_MIN - round(elapsed / CONTROL_INTERVAL_MIN)) <= 1e-6:
                desired = _baseline_desired_action(
                    strategy=strategy,
                    current_action=current,
                    initial_hold=initial_hold,
                    node_depth=np.asarray(pre["depth"], np.float32),
                    node_full_depth=full_depth,
                    action_node_map=np.asarray(graph["action_node_map"], np.float32),
                    binary_indices=binary_indices,
                )
                desired = _enforce_actuator_semantics(
                    desired, ids, actuators, "binary_unless_verified", ["add350.1"]
                )
                # EFD/Auto-RBC share the same K/rate projection as deployable
                # baselines. Literal No-control/All-close/Hold retain their
                # explicit baseline definitions.
                if strategy in {"EFD", "Auto-RBC"}:
                    projected, _, _, executable = project_candidate_sequence(
                        np.repeat(desired[None, :], HORIZON_STEPS, axis=0),
                        current,
                        actuators,
                    )
                    command = projected[0] if executable else current.copy()
                else:
                    command = desired
                for i, aid in enumerate(ids):
                    link_objs[aid].target_setting = float(command[i])
            readback = _observed_action_from_links(link_objs, ids, actuators)
            current = readback.copy()
            records.append(
                _record_row(
                    frame=pre,
                    event_id=event.event_id,
                    strategy=strategy,
                    command=np.asarray(command),
                    readback=readback,
                    node_objs=node_objs,
                    link_objs=link_objs,
                    actuator_ids=ids,
                )
            )
    detail = pd.DataFrame(records)
    detail.to_csv(detail_path, index=False)
    kpis = compute_kpis(detail, priority_nodes, dt_sec=STATE_STEP_SEC)
    baseline_contract = _audit_baseline_contract(detail, strategy, event.inp_path)
    result = {
        "status": "pass",
        "event_id": event.event_id,
        "rainfall_sha256": event.rainfall_sha256,
        "strategy": strategy,
        "authority": "authoritative_swmm",
        "detail_path": str(detail_path),
        "detail_sha256": sha256_file(detail_path),
        "input_sha256": event.input_sha256,
        "kpis": kpis,
        "runtime_sec": time.time() - start,
        **baseline_contract,
    }
    (out_dir / "run_result.json").write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result


def run_proposed_event(
    event: FormalEventInput,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    state_source: str = "gat_sparse_reconstruction",
    device: str = "auto",
    max_candidate_sequences: int = 64,
    step2_calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the canonical V4.2 policy against an authoritative SWMM plant.

    A second native-rule SWMM instance is advanced only to the current time and
    supplies the *current* Dynamic-Internal setting.  The H120 internal action
    input is causal persistence of that current native-rule setting; no future
    shadow state or action is exposed to the controller.  The final Internal
    baseline is evaluated separately with the full native rule engine.
    """
    from pyswmm import Links, Nodes, RainGages, Simulation

    root = Path(project_root)
    actuators = load_actuators(root)
    bundle = load_model_bundle(root, device, step2_calibration_path=step2_calibration_path)
    ids = actuators["actuator_id"].astype(str).tolist()
    if ids != [str(x) for x in bundle.graph.facility_ids]:
        raise RuntimeError("Formal Proposed actuator order differs from trained graph")
    priority_nodes = [str(bundle.graph.node_ids[i]) for i in bundle.priority_indices]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "detail.csv"
    decision_path = out_dir / "decisions.jsonl"
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    start = time.time()
    with Simulation(str(event.inp_path)) as sim, Simulation(str(event.inp_path)) as internal_sim:
        sim.step_advance(STATE_STEP_SEC)
        internal_sim.step_advance(STATE_STEP_SEC)
        sim.start()
        internal_sim.start()
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        internal_links = Links(internal_sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        node_objs = {nid: nodes[nid] for nid in node_ids}
        actuator_ids, link_objs = _get_existing_links(links, ids)
        internal_ids, internal_link_objs = _get_existing_links(internal_links, ids)
        if actuator_ids != ids or internal_ids != ids:
            raise RuntimeError("Formal Proposed/main-shadow INPs do not expose Engineering36")
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        current_action = _observed_action_from_links(link_objs, ids, actuators)
        command = current_action.copy()
        internal_iterator = iter(internal_sim)
        initial = _frame(sim, node_objs, rain_obj, link_objs, ids, actuators)
        if abs(float(initial["elapsed_min"])) <= 1.0e-6:
            # Keep the real t=0 plant frame so the first t=120 decision has
            # the complete causal t-120..t history (25 five-minute frames).
            frames.append(initial)
        for _ in sim:
            try:
                next(internal_iterator)
            except StopIteration:
                raise RuntimeError("Dynamic-Internal causal shadow ended before Proposed plant")
            pre = _frame(sim, node_objs, rain_obj, link_objs, ids, actuators)
            frames.append(pre)
            elapsed = float(pre["elapsed_min"])
            if _is_decision_time(elapsed):
                history, uncertainty, ood_score = reconstruct_history(
                    frames, bundle, state_source=state_source
                )
                historical_actions = np.stack(
                    [np.asarray(x["action"], np.float32) for x in frames[-13:]]
                )
                observed_rain = [float(x["rain"]) for x in frames]
                rainfall_forecast = make_causal_rainfall_forecast(
                    observed_rain, horizon_steps=HORIZON_STEPS
                )
                internal_current = _observed_action_from_links(
                    internal_link_objs, ids, actuators
                )
                command, info = predict_and_decide(
                    bundle=bundle,
                    actuators=actuators,
                    state_history=history,
                    historical_actions=historical_actions,
                    rainfall_forecast=rainfall_forecast,
                    current_action=current_action,
                    internal_current_action=internal_current,
                    gat_ood_score=ood_score,
                    max_candidate_sequences=max_candidate_sequences,
                )
                info.update(
                    {
                        "event_id": event.event_id,
                        "rainfall_sha256": event.rainfall_sha256,
                        "elapsed_min": elapsed,
                        "state_source": state_source,
                        "gat_uncertainty_mean": float(np.mean(uncertainty)),
                        "reconstructed_history_ready_before_mpc": True,
                        "reconstructed_history_contract": "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1"
                        if state_source == "gat_sparse_reconstruction"
                        else "true_state_diagnostic",
                        "current_frame_repetition_used": False,
                        "authoritative_swmm_history_used_as_online_input": state_source == "true_state",
                        "gat_uncertainty_used": state_source == "gat_sparse_reconstruction",
                        "ood_gate_used": False,
                        "ood_diagnostic_used": state_source == "gat_sparse_reconstruction",
                        "sensor_layout_sha256": bundle.sensor_layout_sha256,
                    }
                )
                decisions.append(info)
                for i, aid in enumerate(ids):
                    link_objs[aid].target_setting = float(command[i])
            readback = _observed_action_from_links(link_objs, ids, actuators)
            current_action = readback.copy()
            records.append(
                _record_row(
                    frame=pre,
                    event_id=event.event_id,
                    strategy="Proposed",
                    command=np.asarray(command),
                    readback=readback,
                    node_objs=node_objs,
                    link_objs=link_objs,
                    actuator_ids=ids,
                )
            )
    detail = pd.DataFrame(records)
    detail.to_csv(detail_path, index=False)
    with decision_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(
                json.dumps(_json_safe(item), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    kpis = compute_kpis(detail, priority_nodes, dt_sec=STATE_STEP_SEC)
    result = {
        "status": "pass",
        "event_id": event.event_id,
        "rainfall_sha256": event.rainfall_sha256,
        "strategy": "Proposed",
        "authority": "authoritative_swmm",
        "state_source": state_source,
        "detail_path": str(detail_path),
        "detail_sha256": sha256_file(detail_path),
        "decision_path": str(decision_path),
        "input_sha256": event.input_sha256,
        "kpis": kpis,
        "decision_count": len(decisions),
        "fallback_rate": float(
            np.mean([bool(x.get("used_fallback", True)) for x in decisions])
        )
        if decisions
        else 1.0,
        "canonical_pfvfirst_mpc_v42": True,
        "control_objective_contract": FORMAL_OBJECTIVE_CONTRACT,
        "gat_model_sha256": bundle.gat_model_sha256,
        "surrogate_model_sha256": bundle.surrogate_model_sha256,
        "fallback_contract_sha256": bundle.fallback_contract_sha256,
        "online_future_hydraulic_truth_used": False,
        "realized_future_rainfall_used_online": False,
        "dynamic_internal_online_forecast": "causal_current_native_rule_setting_persistence",
        "runtime_sec": time.time() - start,
    }
    (out_dir / "run_result.json").write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result


def policy_lock_payload(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    step1 = _read_json(root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/evidence.json")
    step2 = _read_json(root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step2_surrogate/evidence.json")
    preferred_step2_root = formal / "step2/models_action_diffusion_v1"
    legacy_step2_root = formal / "step2/models"
    step2_root = (
        preferred_step2_root
        if all(
            (preferred_step2_root / f"seed_{seed}" / "formal_step2_report.json").exists()
            for seed in (17, 42, 73)
        )
        else legacy_step2_root
    )
    step2_reports = [
        _read_json(step2_root / f"seed_{seed}" / "formal_step2_report.json")
        for seed in (17, 42, 73)
    ]
    primary_step2 = next(
        (report for report in step2_reports if int(report.get("seed", -1)) == 42),
        step2_reports[0],
    )
    controller_files = [
        root / "sewerrtc/control/pfvfirst_mpc_v42.py",
        root / "sewerrtc/v4/v42_formal_runtime.py",
        root / "configs/v42_formal_fallback_contract.json",
        root / "docs/contracts/PROJECT6_V42_PAPER_WORKFLOW_CONTRACT.json",
    ]
    policy_sha = hashlib.sha256(
        "\n".join(sha256_file(x) for x in controller_files).encode("utf-8")
    ).hexdigest()
    training_groups: set[str] = set()
    for path in sorted((formal / "step1").glob("seed_*/formal_step1_report.json")):
        report = _read_json(path)
        training_groups.update(map(str, report.get("train_rainfall_groups", [])))
        training_groups.update(map(str, report.get("validation_rainfall_groups", [])))
        training_groups.update(map(str, report.get("model_calibration_rainfall_groups", [])))
    for report in step2_reports:
        training_groups.update(map(str, report.get("train_rainfall_groups", [])))
        training_groups.update(map(str, report.get("validation_rainfall_groups", [])))
        training_groups.update(map(str, report.get("calibration_rainfall_groups", [])))
    fallback = root / FORMAL_FALLBACK_CONTRACT
    objective_contract = root / "configs/v42_control_objective.json"
    candidate_generator = root / "sewerrtc/control/action_sequence_generator.py"
    engineering_projector = root / "sewerrtc/v4/v42_formal_runtime.py"
    production_runtime = root / "scripts/run_v42_formal_production_f2.py"
    surrogate_loop = root / "sewerrtc/v4/v42_formal_surrogate_closed_loop.py"
    pfv_calibration = formal / "calibration/PFV_ONLY_SAFETY_CALIBRATION.json"
    rainfall_forecast = root / "sewerrtc/v4/v42_formal_runtime.py"
    sensor_layout = root / "sewerrtc/v4/v42_step1_dataset.py"
    return {
        "policy_sha256": policy_sha,
        "model_sha256": str(
            primary_step2.get("surrogate_model_sha256")
            or step2["surrogate_model_sha256"]
        ),
        "step2_model_root": str(step2_root),
        "surrogate_action_map_contract": str(
            primary_step2.get("surrogate_action_map_contract", "")
        ),
        "gat_model_sha256": str(step1["gat_model_sha256"]),
        "fallback_contract_sha256": sha256_file(fallback),
        "training_rainfall_sha256s": sorted(training_groups),
        "control_objective_contract": FORMAL_OBJECTIVE_CONTRACT,
        "control_objective_contract_sha256": sha256_file(objective_contract),
        "candidate_generator_sha256": sha256_file(candidate_generator),
        "engineering_projector_sha256": sha256_file(engineering_projector),
        "production_runtime_sha256": sha256_file(production_runtime),
        "surrogate_loop_sha256": sha256_file(surrogate_loop),
        # Empty is intentional before the new PFV-only calibration exists;
        # Policy Lock must fail closed until this field is populated.
        "pfv_calibration_sha256": sha256_file(pfv_calibration) if pfv_calibration.exists() else "",
        "rainfall_forecast_sha256": sha256_file(rainfall_forecast),
        "sensor_layout_sha256": sha256_file(sensor_layout),
    }
