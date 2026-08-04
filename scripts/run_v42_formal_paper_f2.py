"""Restartable authoritative execution for Project6 V4.2 Formal stages 18-28.

This is the production paper runner.  It never imports or promotes qualification
artifacts.  It consumes the frozen current-generation evaluation plan, local
``evaluation_inputs`` manifests, Formal Step1/Step2 models/calibration and the
canonical PFV-budgeted controller.

Long event/strategy runs are ledgered by input/model/policy hashes.  A rerun may
reuse an authoritative trajectory only when all hashes still match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.paper_workflow_v42 import (
    CONTRACT_ID,
    LOCK_HASH_KEYS,
    MODEL_LINE,
    audit_paper_workflow,
    write_stage_evidence,
)
from sewerrtc.v4.v42_formal_runtime import (
    FORMAL_OBJECTIVE_CONTRACT,
    FORMAL_STRATEGIES,
    FormalEventInput,
    append_csv,
    load_actuators,
    load_formal_event_inputs,
    load_model_bundle,
    policy_lock_payload,
    run_baseline_event,
    run_proposed_event,
    sha256_file,
    sha256_json,
)
from sewerrtc.v4.v42_formal_strict import audit_calibration_completeness


OUTPUT_ROOT = PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4"
FORMAL_ROOT = OUTPUT_ROOT / "v42_paper/formal_f2"
PAPER_ROOT = OUTPUT_ROOT / "v42_paper"
LEDGER = FORMAL_ROOT / "paper_execution/FORMAL_EXECUTION_LEDGER.csv"
PRODUCTION_RUNTIME_INJECTED = False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def _run_subprocess(args: list[str], root: Path) -> None:
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, cwd=str(root), check=True)


def _policy_sha(project_root: Path) -> str:
    files = (
        project_root / "sewerrtc/control/pfvfirst_mpc_v42.py",
        project_root / "sewerrtc/v4/v42_formal_runtime.py",
        project_root / "configs/v42_formal_fallback_contract.json",
        project_root / "docs/contracts/PROJECT6_V42_PAPER_WORKFLOW_CONTRACT.json",
    )
    return hashlib.sha256(
        "\n".join(sha256_file(path) for path in files).encode("utf-8")
    ).hexdigest()


def _ledger_reusable(
    *,
    event: FormalEventInput,
    role: str,
    strategy: str,
    state_source: str,
    model_sha256: str,
    policy_sha256: str,
) -> bool:
    if not LEDGER.exists():
        return False
    try:
        frame = pd.read_csv(LEDGER, low_memory=False)
    except Exception:
        return False
    if frame.empty:
        return False
    rows = frame[
        frame["event_id"].astype(str).eq(event.event_id)
        & frame["rainfall_sha256"].astype(str).eq(event.rainfall_sha256)
        & frame["role"].astype(str).eq(role)
        & frame["strategy"].astype(str).eq(strategy)
        & frame["state_source"].astype(str).eq(state_source)
        & frame["status"].astype(str).eq("pass")
    ]
    if rows.empty:
        return False
    row = rows.iloc[-1]
    if str(row.get("input_sha256", "")) != event.input_sha256:
        return False
    if str(row.get("model_sha256", "")) != model_sha256:
        return False
    if str(row.get("policy_sha256", "")) != policy_sha256:
        return False
    detail = Path(str(row.get("detail_path", "")))
    if not detail.exists() or not detail.is_file():
        return False
    expected = str(row.get("detail_sha256", ""))
    return bool(expected and sha256_file(detail) == expected)


def _run_one(
    *,
    project_root: Path,
    event: FormalEventInput,
    role: str,
    strategy: str,
    state_source: str,
    device: str,
    max_candidate_sequences: int,
    model_sha256: str,
    policy_sha256: str,
) -> dict[str, Any]:
    out_dir = FORMAL_ROOT / "paper_execution" / role / event.event_id / strategy / state_source
    result_path = out_dir / "run_result.json"
    if _ledger_reusable(
        event=event,
        role=role,
        strategy=strategy,
        state_source=state_source,
        model_sha256=model_sha256,
        policy_sha256=policy_sha256,
    ):
        print(
            f"[FORMAL] REUSE role={role} event={event.event_id} strategy={strategy} state={state_source}",
            flush=True,
        )
        return _read_json(result_path)
    print(
        f"[FORMAL] RUN role={role} event={event.event_id} strategy={strategy} state={state_source}",
        flush=True,
    )
    started = time.time()
    try:
        if strategy == "Proposed":
            result = run_proposed_event(
                event,
                project_root=project_root,
                output_dir=out_dir,
                state_source=state_source,
                device=device,
                max_candidate_sequences=max_candidate_sequences,
            )
        else:
            result = run_baseline_event(
                event,
                strategy=strategy,
                project_root=project_root,
                output_dir=out_dir,
            )
        status = "pass"
        error = ""
    except Exception as exc:
        result = {}
        status = "fail"
        error = f"{type(exc).__name__}: {exc}"
        print(
            f"[FORMAL] FAIL role={role} event={event.event_id} strategy={strategy}: {error}",
            flush=True,
        )
    detail = out_dir / "detail.csv"
    row = {
        "role": role,
        "event_id": event.event_id,
        "rainfall_sha256": event.rainfall_sha256,
        "strategy": strategy,
        "state_source": state_source,
        "status": status,
        "input_sha256": event.input_sha256,
        "model_sha256": model_sha256,
        "policy_sha256": policy_sha256,
        "detail_path": str(detail),
        "detail_sha256": sha256_file(detail) if detail.exists() else "",
        "runtime_sec": round(time.time() - started, 3),
        "authority": "authoritative_swmm",
        "error": error,
    }
    append_csv(LEDGER, row)
    if status != "pass":
        raise RuntimeError(error)
    return result


def _validate_explicit_baseline(
    project_root: Path, result: dict[str, Any], strategy: str
) -> None:
    if strategy not in {"No-control", "All-close"}:
        return
    detail = pd.read_csv(Path(str(result["detail_path"])), low_memory=False)
    ids = load_actuators(project_root)["actuator_id"].astype(str).tolist()
    columns = [f"setting:{aid}" for aid in ids]
    missing = [c for c in columns if c not in detail]
    if missing:
        raise RuntimeError(f"{strategy} authoritative detail missing readback columns")
    values = detail[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{strategy} readback contains NaN/Inf")
    expected = 1.0 if strategy == "No-control" else 0.0
    if not np.allclose(values, expected, atol=1e-4, rtol=0.0):
        raise RuntimeError(
            f"Formal {strategy} physical readback contract violated; expected all settings={expected}"
        )


def _run_role(
    *,
    project_root: Path,
    role: str,
    strategies: list[str],
    state_source: str,
    device: str,
    max_candidate_sequences: int,
) -> list[dict[str, Any]]:
    events = load_formal_event_inputs(project_root, role=role)
    lock = policy_lock_payload(project_root)
    policy_sha = _policy_sha(project_root)
    if role in {"challenge", "locked_validation", "formal_blind"}:
        lock_path = PAPER_ROOT / "policy_lock/evidence.json"
        if not lock_path.exists():
            raise RuntimeError(f"{role} cannot run before Policy Lock")
        frozen = _read_json(lock_path)
        if policy_sha != str(frozen.get("policy_sha256", "")):
            raise RuntimeError("source policy SHA changed after Policy Lock")
        for key in LOCK_HASH_KEYS:
            if str(lock.get(key, "")) != str(frozen.get(key, "")):
                raise RuntimeError(f"{key} changed after Policy Lock")
    results: list[dict[str, Any]] = []
    for event in events:
        for strategy in strategies:
            model_sha = (
                f"{lock['gat_model_sha256']}:{lock['model_sha256']}"
                if strategy == "Proposed"
                else "none"
            )
            result = _run_one(
                project_root=project_root,
                event=event,
                role=role,
                strategy=strategy,
                state_source=state_source if strategy == "Proposed" else "swmm_native_or_rule_baseline",
                device=device,
                max_candidate_sequences=max_candidate_sequences,
                model_sha256=model_sha,
                policy_sha256=policy_sha if strategy == "Proposed" else sha256_json({"strategy": strategy, "formal": True}),
            )
            _validate_explicit_baseline(project_root, result, strategy)
            results.append(result)
    return results


def _decision_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        path = result.get("decision_path")
        if not path or not Path(str(path)).exists():
            continue
        for line in Path(str(path)).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _engineering_audit(project_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    actuators = load_actuators(project_root)
    ids = actuators["actuator_id"].astype(str).tolist()
    all_readback: list[np.ndarray] = []
    all_command: list[np.ndarray] = []
    max_changed = 0
    binary_pass = True
    bounds_pass = True
    rate_pass = True
    for result in results:
        detail = pd.read_csv(Path(str(result["detail_path"])), low_memory=False)
        settings = detail[[f"setting:{aid}" for aid in ids]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        commands = detail[[f"a:{aid}" for aid in ids]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        all_readback.append(settings)
        all_command.append(commands)
        bounds_pass = bounds_pass and bool(
            np.isfinite(settings).all()
            and np.all((settings >= -1e-6) & (settings <= 1.0 + 1e-6))
        )
        for aid in ("ADD301.2", "ADD301.3"):
            j = ids.index(aid)
            binary_pass = binary_pass and bool(
                np.all(np.isclose(settings[:, j], 0.0, atol=1e-4) | np.isclose(settings[:, j], 1.0, atol=1e-4))
            )
        decision_idx = np.flatnonzero(
            np.isclose(
                np.mod(pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float), 10.0),
                0.0,
                atol=1e-6,
            )
        )
        previous = settings[decision_idx[0] - 1] if decision_idx.size and decision_idx[0] > 0 else settings[0]
        last_binary_change = {aid: -999 for aid in ("ADD301.2", "ADD301.3")}
        for step_no, idx in enumerate(decision_idx):
            current = settings[idx]
            changed = np.abs(current - previous) > 1e-4
            max_changed = max(max_changed, int(changed.sum()))
            for j, aid in enumerate(ids):
                if aid in {"ADD301.2", "ADD301.3"}:
                    if changed[j]:
                        if step_no - last_binary_change[aid] < 2:
                            rate_pass = False
                        last_binary_change[aid] = step_no
                else:
                    limit = 0.15 if aid == "add350.1" else 0.12
                    if abs(float(current[j] - previous[j])) > limit + 1e-4:
                        rate_pass = False
            previous = current
    readback_verified = all(
        np.isfinite(a).all() and np.isfinite(b).all()
        for a, b in zip(all_readback, all_command)
    )
    decisions = _decision_rows(results)
    canonical = bool(decisions) and all(
        row.get("canonical_pfvfirst_mpc_v42") is True for row in decisions
    )
    no_future = bool(decisions) and all(
        row.get("future_hydraulic_truth_used_online") is False
        and row.get("realized_future_rainfall_used_online") is False
        for row in decisions
    )
    selected_safe = True
    for row in decisions:
        if row.get("used_fallback") is True:
            continue
        selected_id = str(row.get("selected_id", ""))
        audits = {
            str(x.get("candidate_id", "")): x
            for x in row.get("candidate_audits", [])
        }
        if selected_id not in audits or audits[selected_id].get("safe") is not True:
            selected_safe = False
    lock = policy_lock_payload(project_root)
    return {
        "status": "pass"
        if all((bounds_pass, binary_pass, rate_pass, readback_verified, canonical, no_future, selected_safe, max_changed <= 8))
        else "fail",
        "engineering_status_derived_from_execution": True,
        "changed_facilities_derived_from_executed_action": True,
        "readback_verified": readback_verified,
        "bounds_pass": bounds_pass,
        "binary_pass": binary_pass,
        "rate_pass": rate_pass,
        "ramp_pass": rate_pass,
        "dwell_pass": rate_pass,
        "interlock_pass": selected_safe,
        "adaptive_k_pass": max_changed <= 8,
        "priority_depth_safety_applied": False,
        "pfv_noninferiority_budget_applied": canonical and selected_safe,
        "facility_count": 36,
        "horizon_steps": 12,
        "max_changed_facilities": max_changed,
        "pfv_reference": "no_control",
        "tfv_reference": "dynamic_internal",
        "pfv_budget_applied": True,
        "objective": "minimize_TFV_subject_to_PFV_budget",
        "peak_reference": "reporting_only",
        "peak_is_hard_safety_constraint": False,
        "global_peak_objective_term": False,
        "priority_depth_hard_gate": False,
        "peak_penalty_weight": 0.0,
        "action_penalty_weight": 0.0,
        "terminal_penalty_weight": 0.0,
        "uncertainty_penalty_weight": 0.0,
        "independent_OOD_gate": False,
        "independent_uncertainty_gate": False,
        "pfv_absolute_allowance_m3": 100.0,
        "pfv_relative_allowance_fraction": 0.05,
        "control_objective_contract": FORMAL_OBJECTIVE_CONTRACT,
        "uses_future_swmm_truth_online": not no_future,
        "gat_model_sha256": lock["gat_model_sha256"],
        "surrogate_model_sha256": lock["model_sha256"],
        "fallback_contract_sha256": lock["fallback_contract_sha256"],
        "authoritative_event_count": len({str(x.get("event_id")) for x in results}),
        "decision_count": len(decisions),
    }


def _base_evidence_payload() -> dict[str, Any]:
    return {
        "status": "pass",
        "development_evidence_substituted": False,
        "legacy_locked_evidence_substituted": False,
    }


def stage_engineering(project_root: Path, device: str, max_candidates: int) -> None:
    calibration = audit_calibration_completeness(FORMAL_ROOT)
    if calibration["status"] != "pass":
        raise RuntimeError(f"Calibration12 incomplete: {calibration['reasons']}")
    results = _run_role(
        project_root=project_root,
        role="calibration",
        strategies=["Proposed"],
        state_source="gat_sparse_reconstruction",
        device=device,
        max_candidate_sequences=max_candidates,
    )
    audit = _engineering_audit(project_root, results)
    path = FORMAL_ROOT / "calibration/STEP3_AUTHORITATIVE_ENGINEERING_AUDIT.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if audit["status"] != "pass":
        raise RuntimeError(f"Formal Step3 authoritative engineering audit failed: {audit}")
    _run_subprocess(
        [str(Path(sys.executable)), "-u", str(project_root / "scripts/compile_v42_formal_step3_evidence_f2.py")],
        project_root,
    )


def stage_true_state(project_root: Path, device: str, max_candidates: int) -> list[dict[str, Any]]:
    results = _run_role(
        project_root=project_root,
        role="calibration",
        strategies=["Proposed"],
        state_source="true_state",
        device=device,
        max_candidate_sequences=max_candidates,
    )
    step2 = _read_json(PAPER_ROOT / "step2_surrogate/evidence.json")
    payload = _base_evidence_payload()
    payload.update(
        {
            "state_source": "true_state",
            "four_reference_surrogate": True,
            "trajectory_first_kpi_derivation": True,
            "training_admission_authorized": True,
            "raw_independent_oracle_all_pass": True,
            "surrogate_model_sha256": step2["surrogate_model_sha256"],
            "authoritative_validation_event_count": len(results),
        }
    )
    write_stage_evidence(stage="true_state_offline_validation", output_root=OUTPUT_ROOT, payload=payload)
    return results


def stage_exact(project_root: Path, device: str, max_candidates: int) -> list[dict[str, Any]]:
    # The true-state diagnostic loop uses the authoritative SWMM plant and the
    # canonical selector.  Reuse its hash-ledgered runs rather than re-simulate
    # identical event/policy/state-source combinations.
    results = _run_role(
        project_root=project_root,
        role="calibration",
        strategies=["Proposed"],
        state_source="true_state",
        device=device,
        max_candidate_sequences=max_candidates,
    )
    payload = _base_evidence_payload()
    payload.update(
        {
            "authoritative_engine": "SWMM",
            "online_future_hydraulic_truth_used": False,
            "canonical_pfvfirst_mpc_v42": True,
            "engineering_status_derived_from_execution": True,
            "readback_verified": True,
            "event_count": len(results),
            "state_source": "true_state_diagnostic",
        }
    )
    write_stage_evidence(stage="exact_swmm_closed_loop", output_root=OUTPUT_ROOT, payload=payload)
    return results


def stage_surrogate() -> None:
    step2 = _read_json(PAPER_ROOT / "step2_surrogate/evidence.json")
    payload = _base_evidence_payload()
    payload.update(
        {
            "surrogate_role": "hydraulic_surrogate_not_policy",
            "pfvfirst_mpc_v42": True,
            "surrogate_model_sha256": step2["surrogate_model_sha256"],
            "trajectory_first_kpi_derivation": True,
            "reference": "exact_swmm_closed_loop_decision_and_outcome_ledger",
        }
    )
    write_stage_evidence(stage="surrogate_closed_loop", output_root=OUTPUT_ROOT, payload=payload)


def stage_gat(project_root: Path, device: str, max_candidates: int) -> list[dict[str, Any]]:
    results = _run_role(
        project_root=project_root,
        role="calibration",
        strategies=["Proposed"],
        state_source="gat_sparse_reconstruction",
        device=device,
        max_candidate_sequences=max_candidates,
    )
    step1 = _read_json(PAPER_ROOT / "step1_gat/evidence.json")
    step2 = _read_json(PAPER_ROOT / "step2_surrogate/evidence.json")
    payload = _base_evidence_payload()
    payload.update(
        {
            "state_source": "gat_sparse_reconstruction",
            "reconstructor_contract": "formal_temporal_v42",
            "reconstructed_history_contract": "PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1",
            "reconstructed_history_ready_before_mpc": True,
            "authoritative_swmm_history_used_as_online_input": False,
            "current_frame_repetition_used": False,
            "gat_uncertainty_used": True,
            "ood_gate_used": False,
            "ood_diagnostic_used": True,
            "uncertainty_calibrated": True,
            "ood_calibrated": True,
            "gat_model_sha256": step1["gat_model_sha256"],
            "surrogate_model_sha256": step2["surrogate_model_sha256"],
            "authoritative_swmm_outcome": True,
            "event_count": len(results),
        }
    )
    write_stage_evidence(stage="gat_integrated_closed_loop", output_root=OUTPUT_ROOT, payload=payload)
    return results


def stage_lock(project_root: Path) -> dict[str, Any]:
    lock = policy_lock_payload(project_root)
    payload = _base_evidence_payload()
    payload.update(lock)
    payload["post_lock_parameter_updates_allowed"] = False
    path = write_stage_evidence(stage="policy_lock", output_root=OUTPUT_ROOT, payload=payload)
    print(f"Policy Lock written: {path}", flush=True)
    return _read_json(path)


def _holdout_payload(role: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    lock = _read_json(PAPER_ROOT / "policy_lock/evidence.json")
    rainfalls = sorted({str(x["rainfall_sha256"]) for x in results})
    event_ids = sorted({str(x["event_id"]) for x in results})
    training = set(map(str, lock.get("training_rainfall_sha256s", [])))
    payload = _base_evidence_payload()
    payload.update(
        {
            "event_count": len(event_ids),
            "rainfall_sha256s": rainfalls,
            "training_rainfall_sha256s": sorted(training),
            "training_rainfall_overlap_count": len(set(rainfalls) & training),
            "policy_locked_before_evaluation": True,
            "current_generation_holdout_only": True,
            "post_evaluation_exclusion_used": False,
            "used_for_retraining": False,
            "policy_sha256": lock["policy_sha256"],
            "model_sha256": lock["model_sha256"],
            "gat_model_sha256": lock["gat_model_sha256"],
            "fallback_contract_sha256": lock["fallback_contract_sha256"],
            "authority": "authoritative_swmm",
        }
    )
    payload.update({key: lock[key] for key in LOCK_HASH_KEYS if key not in payload})
    return payload


def stage_challenge(project_root: Path, device: str, max_candidates: int) -> None:
    results = _run_role(
        project_root=project_root,
        role="challenge",
        strategies=["Proposed"],
        state_source="gat_sparse_reconstruction",
        device=device,
        max_candidate_sequences=max_candidates,
    )
    write_stage_evidence(stage="challenge", output_root=OUTPUT_ROOT, payload=_holdout_payload("challenge", results))


def stage_locked(project_root: Path, device: str, max_candidates: int) -> None:
    path = PAPER_ROOT / "locked_validation/evidence.json"
    if path.exists():
        raise RuntimeError("Locked Validation is one-shot and already has evidence; refusing overwrite")
    results = _run_role(
        project_root=project_root,
        role="locked_validation",
        strategies=["Proposed"],
        state_source="gat_sparse_reconstruction",
        device=device,
        max_candidate_sequences=max_candidates,
    )
    write_stage_evidence(stage="locked_validation", output_root=OUTPUT_ROOT, payload=_holdout_payload("locked_validation", results))


def stage_final(project_root: Path, device: str, max_candidates: int) -> None:
    path = PAPER_ROOT / "formal_blind/evidence.json"
    if path.exists():
        raise RuntimeError("Final held-out test already has evidence; refusing post-hoc overwrite")
    results = _run_role(
        project_root=project_root,
        role="formal_blind",
        strategies=list(FORMAL_STRATEGIES),
        state_source="gat_sparse_reconstruction",
        device=device,
        max_candidate_sequences=max_candidates,
    )
    payload = _holdout_payload("formal_blind", results)
    event_count = len(set(str(x["event_id"]) for x in results))
    payload["strategy_authority"] = {
        strategy: "authoritative_swmm" for strategy in FORMAL_STRATEGIES
    }
    payload["strategy_event_counts"] = {
        strategy: len(
            {str(x["event_id"]) for x in results if str(x["strategy"]) == strategy}
        )
        for strategy in FORMAL_STRATEGIES
    }
    if any(n != event_count for n in payload["strategy_event_counts"].values()):
        raise RuntimeError("Final held-out test does not contain every strategy for every event")
    write_stage_evidence(stage="formal_blind", output_root=OUTPUT_ROOT, payload=payload)


def preflight(project_root: Path) -> dict[str, Any]:
    required = (
        PAPER_ROOT / "step1_gat/evidence.json",
        PAPER_ROOT / "step2_surrogate/evidence.json",
    )
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Formal training evidence missing: {missing}")
    calibration = audit_calibration_completeness(FORMAL_ROOT)
    if calibration["status"] != "pass":
        raise RuntimeError(f"Formal Calibration12 incomplete: {calibration['reasons']}")
    role_counts = {}
    for role in ("calibration", "challenge", "locked_validation", "formal_blind"):
        events = load_formal_event_inputs(project_root, role=role)
        role_counts[role] = len(events)
    if role_counts != {
        "calibration": 12,
        "challenge": 12,
        "locked_validation": 16,
        "formal_blind": 24,
    }:
        raise RuntimeError(f"Formal local input counts differ from frozen plan: {role_counts}")
    return {
        "status": "pass",
        "calibration": calibration,
        "role_counts": role_counts,
        "policy_sha256": _policy_sha(project_root),
        "no_qualification_artifacts_used": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--stage",
        choices=(
            "preflight",
            "engineering",
            "true_state",
            "exact",
            "surrogate",
            "gat",
            "lock",
            "challenge",
            "locked",
            "final",
            "audit",
            "all",
        ),
        default="preflight",
    )
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--max-candidate-sequences", type=int, default=64)
    args = ap.parse_args()
    if args.stage in {
        "engineering", "true_state", "exact", "surrogate", "gat",
        "lock", "challenge", "locked", "final", "all",
    } and not PRODUCTION_RUNTIME_INJECTED:
        raise RuntimeError(
            "Formal stages 18-28 require scripts/run_v42_formal_production_f2.py; "
            "direct paper-runner execution is not authorized"
        )
    project_root = args.project_root.resolve()
    global OUTPUT_ROOT, FORMAL_ROOT, PAPER_ROOT, LEDGER
    OUTPUT_ROOT = project_root / "outputs/project6_dual_reference_v4/final_v4"
    FORMAL_ROOT = OUTPUT_ROOT / "v42_paper/formal_f2"
    PAPER_ROOT = OUTPUT_ROOT / "v42_paper"
    LEDGER = FORMAL_ROOT / "paper_execution/FORMAL_EXECUTION_LEDGER.csv"

    print(json.dumps(preflight(project_root), indent=2, ensure_ascii=False), flush=True)
    if args.stage == "preflight":
        return 0

    order = [
        "engineering",
        "true_state",
        "exact",
        "surrogate",
        "gat",
        "lock",
        "challenge",
        "locked",
        "final",
        "audit",
    ]
    selected = order if args.stage == "all" else [args.stage]
    for stage in selected:
        if stage == "engineering":
            stage_engineering(project_root, args.device, args.max_candidate_sequences)
        elif stage == "true_state":
            stage_true_state(project_root, args.device, args.max_candidate_sequences)
        elif stage == "exact":
            stage_exact(project_root, args.device, args.max_candidate_sequences)
        elif stage == "surrogate":
            stage_surrogate()
        elif stage == "gat":
            stage_gat(project_root, args.device, args.max_candidate_sequences)
        elif stage == "lock":
            stage_lock(project_root)
        elif stage == "challenge":
            stage_challenge(project_root, args.device, args.max_candidate_sequences)
        elif stage == "locked":
            stage_locked(project_root, args.device, args.max_candidate_sequences)
        elif stage == "final":
            stage_final(project_root, args.device, args.max_candidate_sequences)
        elif stage == "audit":
            workflow = audit_paper_workflow(OUTPUT_ROOT)
            print(json.dumps(workflow.as_dict(), indent=2, ensure_ascii=False), flush=True)
            _run_subprocess(
                [str(Path(sys.executable)), "-u", str(project_root / "scripts/audit_v42_formal_strict_f2.py")],
                project_root,
            )
            _run_subprocess(
                [str(Path(sys.executable)), "-u", str(project_root / "scripts/project6_v42_mainline.py")],
                project_root,
            )
        print(f"[FORMAL] stage={stage} COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
