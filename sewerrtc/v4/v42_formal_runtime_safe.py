"""Safety wrapper around the V4.2 Formal authoritative runtime.

For Proposed and every non-Internal baseline, Engineering36 must be controlled by
the evaluated policy, not simultaneously by the INP's native [CONTROLS] rules.
This module creates a short-lived rule-free runtime INP while preserving the
physical network/rainfall definition. ``Internal`` alone runs the original INP
with its native rules. Proposed uses *both*: rule-free plant + native-rule causal
shadow advanced only to the current decision time.

Before the first decision (120 min, required by the 13 causal GAT anchors), the
rule-free Proposed plant replays the *current* native-Internal readback from the
causal shadow at every 5-min step. Runtime execution additionally enforces the
cross-decision minimum dwell guard and verifies every PySWMM ``target_setting``
write before accepting it as Formal evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import (
    _as_float,
    _get_existing_links,
    _ids_from_container,
    _observed_action_from_links,
    physical_network_sha256,
)
from sewerrtc.v4.v42_formal_runtime import (
    FORMAL_OBJECTIVE_CONTRACT,
    HORIZON_STEPS,
    STATE_STEP_SEC,
    FormalEventInput,
    _frame,
    _is_decision_time,
    _json_safe,
    _record_row,
    load_actuators,
    load_model_bundle,
    predict_and_decide,
    reconstruct_history,
    run_baseline_event as _run_baseline_event,
    sha256_file,
)
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast


def build_rule_free_runtime_inp(source: str | Path, target: str | Path) -> Path:
    source = Path(source)
    target = Path(target)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    output: list[str] = []
    in_controls = False
    controls_seen = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip().upper()
            if name == "CONTROLS":
                in_controls = True
                controls_seen = True
                output.append(raw)
                output.append(
                    "; Formal runtime: native control rules disabled; evaluated policy owns Engineering36."
                )
                continue
            if in_controls:
                in_controls = False
            output.append(raw)
            continue
        if in_controls:
            if stripped.startswith(";"):
                output.append(raw)
            continue
        output.append(raw)
    if not controls_seen:
        output.extend(
            [
                "",
                "[CONTROLS]",
                "; Formal runtime: no native control rules in source INP.",
            ]
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    if physical_network_sha256(source) != physical_network_sha256(target):
        raise RuntimeError(
            "removing [CONTROLS] changed the physical-network SHA; refusing Formal runtime clone"
        )
    return target.resolve()


def _runtime_event(event: FormalEventInput, output_dir: Path) -> FormalEventInput:
    digest = hashlib.sha256(
        f"{event.event_id}|{event.rainfall_sha256}|{event.input_sha256}".encode("utf-8")
    ).hexdigest()[:16]
    runtime_inp = build_rule_free_runtime_inp(
        event.inp_path,
        output_dir / "runtime_inp" / f"{digest}__no_native_controls.inp",
    )
    return replace(event, inp_path=runtime_inp)


def _target_readback(link_objs: dict[str, Any], ids: list[str]) -> np.ndarray:
    return np.asarray(
        [
            _as_float(getattr(link_objs[aid], "target_setting", np.nan), np.nan)
            for aid in ids
        ],
        dtype=np.float32,
    )


def _write_and_verify_target(
    link_objs: dict[str, Any], ids: list[str], command: np.ndarray
) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32).reshape(-1)
    if command.size != len(ids) or not np.isfinite(command).all():
        raise RuntimeError("Formal target command is not a finite Engineering36 vector")
    for i, aid in enumerate(ids):
        link_objs[aid].target_setting = float(command[i])
    written = _target_readback(link_objs, ids)
    if not np.isfinite(written).all() or not np.allclose(
        written, command, atol=1.0e-6, rtol=0.0
    ):
        raise RuntimeError(
            "PySWMM target_setting write/readback mismatch; refusing Formal execution"
        )
    return written


def _runtime_dwell_guard(
    command: np.ndarray,
    current: np.ndarray,
    ids: list[str],
    *,
    decision_step: int,
    last_change_step: dict[str, int],
) -> tuple[np.ndarray, bool, list[str]]:
    """Enforce >=2 control-step dwell across rolling-MPC decisions.

    The two binary pumps and the verified variable-speed pump are the devices
    with explicit minimum dwell in the frozen Engineering36 controller lineage.
    If any proposed first-step change violates dwell, the complete action falls
    back to the current readback rather than partially executing a differently
    scored candidate.
    """
    command = np.asarray(command, dtype=np.float32).copy()
    current = np.asarray(current, dtype=np.float32).copy()
    protected = {"ADD301.2", "ADD301.3", "add350.1"}
    violations: list[str] = []
    changed = np.abs(command - current) > 1.0e-6
    for i, aid in enumerate(ids):
        if aid not in protected or not changed[i]:
            continue
        if int(decision_step) - int(last_change_step.get(aid, -1000)) < 2:
            violations.append(aid)
    if violations:
        return current, False, violations
    for i, aid in enumerate(ids):
        if changed[i] and aid in protected:
            last_change_step[aid] = int(decision_step)
    return command, True, []


def _record_row_with_target(
    *,
    frame: dict[str, Any],
    event_id: str,
    strategy: str,
    command: np.ndarray,
    readback: np.ndarray,
    node_objs: dict[str, Any],
    link_objs: dict[str, Any],
    actuator_ids: list[str],
) -> dict[str, Any]:
    row = _record_row(
        frame=frame,
        event_id=event_id,
        strategy=strategy,
        command=command,
        readback=readback,
        node_objs=node_objs,
        link_objs=link_objs,
        actuator_ids=actuator_ids,
    )
    written = _target_readback(link_objs, actuator_ids)
    for i, aid in enumerate(actuator_ids):
        row[f"target:{aid}"] = float(written[i])
    return row


def run_baseline_event(
    event: FormalEventInput,
    *,
    strategy: str,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    if strategy == "Internal":
        result = _run_baseline_event(
            event,
            strategy=strategy,
            project_root=project_root,
            output_dir=out,
        )
        result["native_controls_preserved"] = True
        return result
    runtime_event = _runtime_event(event, out)
    result = _run_baseline_event(
        runtime_event,
        strategy=strategy,
        project_root=project_root,
        output_dir=out,
    )
    result["source_input_sha256"] = event.input_sha256
    result["runtime_rule_free_inp_sha256"] = runtime_event.input_sha256
    result["native_controls_disabled"] = True
    result["physical_network_sha256"] = physical_network_sha256(event.inp_path)
    return result


def run_proposed_event(
    event: FormalEventInput,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    state_source: str = "gat_sparse_reconstruction",
    device: str = "auto",
    max_candidate_sequences: int = 64,
) -> dict[str, Any]:
    """Run rule-free Proposed plant plus causal native-rule Internal shadow."""
    from pyswmm import Links, Nodes, RainGages, Simulation

    root = Path(project_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_event = _runtime_event(event, out_dir)
    actuators = load_actuators(root)
    bundle = load_model_bundle(root, device)
    ids = actuators["actuator_id"].astype(str).tolist()
    if ids != [str(x) for x in bundle.graph.facility_ids]:
        raise RuntimeError("Formal Proposed actuator order differs from trained graph")
    priority_nodes = [str(bundle.graph.node_ids[i]) for i in bundle.priority_indices]
    detail_path = out_dir / "detail.csv"
    decision_path = out_dir / "decisions.jsonl"
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    last_change_step: dict[str, int] = {}
    decision_step = 0
    target_write_verified_count = 0
    runtime_dwell_fallback_count = 0
    started = time.time()

    with Simulation(str(runtime_event.inp_path)) as sim, Simulation(str(event.inp_path)) as internal_sim:
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
            raise RuntimeError("Formal Proposed plant/shadow do not expose Engineering36")
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        current_action = _observed_action_from_links(link_objs, ids, actuators)
        internal_initial = _observed_action_from_links(internal_link_objs, ids, actuators)
        command = internal_initial.copy()
        _write_and_verify_target(link_objs, ids, command)
        shadow_iter = iter(internal_sim)
        for _ in sim:
            try:
                next(shadow_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Dynamic-Internal causal shadow ended before Proposed plant"
                ) from exc
            pre = _frame(sim, node_objs, rain_obj, link_objs, ids, actuators)
            frames.append(pre)
            elapsed = float(pre["elapsed_min"])
            internal_current = _observed_action_from_links(
                internal_link_objs, ids, actuators
            )
            if elapsed < 120.0 - 1.0e-6:
                command = internal_current.copy()
                _write_and_verify_target(link_objs, ids, command)
            elif _is_decision_time(elapsed):
                history, uncertainty, ood_score = reconstruct_history(
                    frames, bundle, state_source=state_source
                )
                historical_actions = np.stack(
                    [np.asarray(x["action"], np.float32) for x in frames[-13:]]
                )
                rainfall_forecast = make_causal_rainfall_forecast(
                    [float(x["rain"]) for x in frames], horizon_steps=HORIZON_STEPS
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
                command, dwell_pass, dwell_violations = _runtime_dwell_guard(
                    command,
                    current_action,
                    ids,
                    decision_step=decision_step,
                    last_change_step=last_change_step,
                )
                if not dwell_pass:
                    runtime_dwell_fallback_count += 1
                    info["used_fallback"] = True
                    info["selected_id"] = "frozen_hold_readback"
                    info["reason"] = "runtime_cross_decision_dwell_guard"
                    info["runtime_dwell_violations"] = dwell_violations
                written = _write_and_verify_target(link_objs, ids, command)
                target_write_verified_count += 1
                info.update(
                    {
                        "event_id": event.event_id,
                        "rainfall_sha256": event.rainfall_sha256,
                        "elapsed_min": elapsed,
                        "state_source": state_source,
                        "gat_uncertainty_mean": float(np.mean(uncertainty)),
                        "reconstructed_history_ready_before_mpc": True,
                        "reconstructed_history_contract": (
                            "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1"
                            if state_source == "gat_sparse_reconstruction"
                            else "true_state_diagnostic"
                        ),
                        "current_frame_repetition_used": False,
                        "authoritative_swmm_history_used_as_online_input": state_source
                        == "true_state",
                        "gat_uncertainty_used": state_source
                        == "gat_sparse_reconstruction",
                        "ood_gate_used": state_source
                        == "gat_sparse_reconstruction",
                        "sensor_layout_sha256": bundle.sensor_layout_sha256,
                        "plant_native_controls_disabled": True,
                        "internal_shadow_native_controls_preserved": True,
                        "internal_shadow_future_state_used_online": False,
                        "precontrol_prefix_contract": "causal_internal_readback_replay",
                        "runtime_cross_decision_dwell_pass": dwell_pass,
                        "target_write_verified": bool(
                            np.allclose(written, command, atol=1.0e-6, rtol=0.0)
                        ),
                    }
                )
                decisions.append(info)
                decision_step += 1
            readback = _observed_action_from_links(link_objs, ids, actuators)
            current_action = readback.copy()
            records.append(
                _record_row_with_target(
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
        "source_input_sha256": event.input_sha256,
        "runtime_rule_free_inp_sha256": runtime_event.input_sha256,
        "physical_network_sha256": physical_network_sha256(event.inp_path),
        "kpis": kpis,
        "decision_count": len(decisions),
        "fallback_rate": (
            float(np.mean([bool(x.get("used_fallback", True)) for x in decisions]))
            if decisions
            else 1.0
        ),
        "canonical_pfvfirst_mpc_v42": True,
        "control_objective_contract": FORMAL_OBJECTIVE_CONTRACT,
        "gat_model_sha256": bundle.gat_model_sha256,
        "surrogate_model_sha256": bundle.surrogate_model_sha256,
        "fallback_contract_sha256": bundle.fallback_contract_sha256,
        "online_future_hydraulic_truth_used": False,
        "realized_future_rainfall_used_online": False,
        "dynamic_internal_online_forecast": "causal_current_native_rule_setting_persistence",
        "precontrol_prefix_contract": "causal_internal_readback_replay",
        "plant_native_controls_disabled": True,
        "internal_shadow_native_controls_preserved": True,
        "internal_shadow_future_state_used_online": False,
        "target_write_verified_count": target_write_verified_count,
        "target_write_all_decisions_verified": bool(
            decisions and target_write_verified_count == len(decisions)
        ),
        "runtime_dwell_fallback_count": runtime_dwell_fallback_count,
        "runtime_cross_decision_dwell_enforced": True,
        "runtime_sec": time.time() - started,
    }
    (out_dir / "run_result.json").write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result
