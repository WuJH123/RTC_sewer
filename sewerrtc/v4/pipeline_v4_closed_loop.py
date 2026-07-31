"""Final-V4 closed-loop stage guards and plan/audit helpers.

The physical controller remains in :mod:`sewerrtc.simulation.pyswmm_runner`.
This module owns the evidence gates around it so a CSV or placeholder artifact
can never authorize a closed-loop, Challenge, or Formal run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .closed_loop import SURROGATE_ABLATIONS
from .runtime import (
    EXIT_BLOCKED,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_SCIENTIFIC_FAIL,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    working_code_sha,
)


CLOSED_LOOP_ABLATIONS = SURROGATE_ABLATIONS
PREDICTIVE_GATE_REL = "models/v4_compact_v1/v4_predictive_generalization_gate.json"
GAT_DECISION_REL = "docs/contracts/gat_primary_selection_decision.json"


def predictive_gate_authorizes_closed_loop(payload: dict) -> bool:
    """Only the fresh V4.1 Locked gate may release a closed loop."""
    return bool(
        payload.get("status") == "pass"
        and payload.get("authorizes_closed_loop") is True
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _missing(stage: str, *paths: Path) -> StageResult:
    return StageResult(
        stage,
        "incomplete",
        EXIT_INCOMPLETE,
        remaining=len(paths),
        evidence={"missing_inputs": [str(path) for path in paths]},
    )


def _predictive_authorization(
    project_root: Path, output_root: Path
) -> tuple[bool, dict]:
    gate_path = output_root / PREDICTIVE_GATE_REL
    if not gate_path.exists():
        return False, {"reason": "v41_predictive_gate_missing", "path": str(gate_path)}
    gate = _read_json(gate_path)
    if not predictive_gate_authorizes_closed_loop(gate):
        return False, {
            "reason": "v41_predictive_gate_not_authorized",
            "status": gate.get("status"),
            "authorizes_closed_loop": gate.get("authorizes_closed_loop"),
        }
    return True, {"predictive_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest()}


def _gat_readiness_handler(
    project_root: Path, output_root: Path, _config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditGATClosedLoopReadiness"
        decision_path = project_root / GAT_DECISION_REL
        if not decision_path.exists():
            return _missing(stage, decision_path)
        decision = _read_json(decision_path)
        selected = str(decision.get("registry_name", ""))
        passed = selected == "sr0p15"
        report = {
            "stage": stage,
            "selected_primary_gat": selected,
            "checks": {
                "primary_is_sr0p15": passed,
                "decision_file_present": True,
            },
            "code_sha256": working_code_sha(project_root),
        }
        atomic_write_json(output_root / "audits" / "gat_closed_loop_readiness.json", report)
        return StageResult(
            stage,
            "pass" if passed else "blocked",
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=int(passed),
            remaining=0 if passed else 1,
            batch_complete=True,
            scope_complete=passed,
            evidence=report,
        )

    return handler


def _closed_loop_plan_handler(
    project_root: Path, output_root: Path, config: dict, *, stage: str, evaluator: str
) -> Callable[[RuntimeOptions], StageResult]:
    """Freeze a development plan only after the V4.1 Locked release.

    It deliberately writes *plan metadata*, not an executable SWMM row.  The
    runtime adapter must bind each row to the audited V4.1 evaluator; this
    prevents the generic legacy runner from silently substituting V3 models.
    """
    def handler(_options: RuntimeOptions) -> StageResult:
        authorized, evidence = _predictive_authorization(project_root, output_root)
        if not authorized:
            return StageResult(stage, "blocked", EXIT_BLOCKED, remaining=1, evidence=evidence)
        ledger_path = output_root / "inventory" / "event_usage_ledger.csv"
        if not ledger_path.exists():
            return _missing(stage, ledger_path)
        ledger = pd.read_csv(ledger_path)
        required = {"event_id", "rainfall_sha256", "assigned_split", "used_challenge", "used_formal"}
        missing = required - set(ledger)
        if missing:
            return StageResult(stage, "blocked", EXIT_BLOCKED, evidence={"reason": "ledger_schema", "missing": sorted(missing)})
        count = int(config.get("closed_loop", {}).get("development_events", 12))
        eligible = ledger[
            ledger["assigned_split"].astype(str).isin(["train", "pilot"])
            & ~ledger["used_challenge"].astype(bool)
            & ~ledger["used_formal"].astype(bool)
        ].drop_duplicates("event_id")
        if len(eligible) < count:
            return StageResult(stage, "blocked", EXIT_BLOCKED, remaining=count - len(eligible), evidence={"reason": "insufficient_development_events", "eligible": int(len(eligible)), "required": count})
        selected = eligible.sort_values("event_id").iloc[:count][["event_id", "rainfall_sha256"]].copy()
        selected["evaluator"] = evaluator
        selected["state_source"] = "true_state" if evaluator == "exact" else "mixed"
        selected["ablation_ids"] = json.dumps(
            ["A", "C"] if evaluator == "exact" else ["B", "D"]
        )
        selected["development_only"] = True
        plan_rel = "exact_closed_loop/plan.csv" if evaluator == "exact" else "surrogate_closed_loop/plan.csv"
        target = output_root / plan_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(target, index=False)
        freeze = {
            "stage": stage,
            "plan": plan_rel,
            "evaluator": evaluator,
            "development_only": True,
            "event_count": int(len(selected)),
            "plan_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            **evidence,
            "code_sha256": working_code_sha(project_root),
        }
        atomic_write_json(target.with_suffix(".freeze.json"), freeze)
        return StageResult(stage, "pass", EXIT_PASS, completed=len(selected), batch_complete=True, scope_complete=True, evidence=freeze)

    return handler


def _fail_closed_runtime_handler(
    project_root: Path, output_root: Path, _config: dict, *, stage: str, plan_rel: str
) -> Callable[[RuntimeOptions], StageResult]:
    """Reject legacy execution until the V4.1 online feature adapter is bound.

    This is intentionally a hard block rather than letting the generic run
    dispatcher call an old V3 predictor under a V4.1 stage name.
    """
    def handler(_options: RuntimeOptions) -> StageResult:
        authorized, evidence = _predictive_authorization(project_root, output_root)
        if not authorized:
            return StageResult(stage, "blocked", EXIT_BLOCKED, remaining=1, evidence=evidence)
        plan = output_root / plan_rel
        if not plan.exists():
            return _missing(stage, plan)
        return StageResult(
            stage,
            "blocked",
            EXIT_BLOCKED,
            remaining=len(pd.read_csv(plan)),
            evidence={
                "reason": "v41_online_feature_and_reference_forecaster_required",
                "prohibited_fallback": "generic_legacy_v3_mpc_runner",
                "action_required": (
                    "bind CompactHeadSpecificModel to complete leakage-free "
                    "telemetry and separately validated online No-control/DI "
                    "reference forecasters before SWMM execution"
                ),
            },
        )

    return handler


def build_v4_closed_loop_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    """Real handlers for the Final-V4 downstream gates.

    Run stages are deliberately fail-closed until a V4.1 online feature
    adapter exists; a legacy V3 controller must never be auto-selected.
    """
    return {
        "AuditGATClosedLoopReadiness": _gat_readiness_handler(project_root, output_root, config),
        "PlanExactClosedLoop": _closed_loop_plan_handler(project_root, output_root, config, stage="PlanExactClosedLoop", evaluator="exact"),
        "PlanSurrogateClosedLoop": _closed_loop_plan_handler(project_root, output_root, config, stage="PlanSurrogateClosedLoop", evaluator="surrogate"),
        "RunExactClosedLoop": _fail_closed_runtime_handler(project_root, output_root, config, stage="RunExactClosedLoop", plan_rel="exact_closed_loop/plan.csv"),
        "RunSurrogateClosedLoop": _fail_closed_runtime_handler(project_root, output_root, config, stage="RunSurrogateClosedLoop", plan_rel="surrogate_closed_loop/plan.csv"),
    }
