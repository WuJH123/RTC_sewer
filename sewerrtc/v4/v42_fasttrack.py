"""Development-only fast-track for proving V4.2 learnability/controlability.

The formal paper contract is intentionally unchanged.  This module creates a
small, deterministic, rainfall-isolated evidence core from the existing R0.1
inventory, audits only that core at full finite-value fidelity, and provides
GO/NO-GO stage gates for Step 1--4.  Passing this workflow never authorizes
Formal Blind or substitutes for the complete Phase R0 evidence audit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_case_alignment_audit import audit_case_alignment
from sewerrtc.v4.v42_existing_pool_audit import (
    _finite_target_check,
    _load_engineering36_ids,
    _load_graph_topology,
    _parse_inp_topology,
    _target_columns,
)
from sewerrtc.v4.v42_reusable_pool import build_reusable_paper_pool


CONTRACT_ID = "PROJECT6_V42_FASTTRACK_POC_V1"
FORMAL_CONTRACT_ID = "PROJECT6_V42_PAPER_WORKFLOW_V1"
STAGES = (
    "core_pool",
    "step1_gat",
    "step2_surrogate",
    "step3_mpc",
    "step4_micro_closed_loop",
)
EVIDENCE_RELATIVE_PATHS = {
    "core_pool": "v42_fasttrack/core_pool/evidence.json",
    "step1_gat": "v42_fasttrack/step1_gat/evidence.json",
    "step2_surrogate": "v42_fasttrack/step2_surrogate/evidence.json",
    "step3_mpc": "v42_fasttrack/step3_mpc/evidence.json",
    "step4_micro_closed_loop": "v42_fasttrack/step4_micro_closed_loop/evidence.json",
}

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "core_pool": {
        "independent_rainfall_groups": 8,
        "aligned_cases": 12,
        "finite_pass_fraction": 0.95,
    },
    "step1_gat": {
        "val_nse_median": 0.65,
        "priority_nse_median": 0.55,
    },
    "step2_surrogate": {
        "pfv_direction_accuracy": 0.70,
        "tfv_direction_accuracy": 0.60,
        "peak_direction_accuracy": 0.60,
        "safe_candidate_recall": 0.70,
        "false_safe_rate_max": 0.15,
    },
    "step3_mpc": {
        "states_evaluated": 12,
        "safe_selection_precision": 0.80,
        "good_candidate_recall": 0.60,
    },
    "step4_micro_closed_loop": {
        "event_count": 3,
        "pfv_noninferior_rate": 2.0 / 3.0,
        "peak_noninferior_rate": 2.0 / 3.0,
        "tfv_improved_rate": 0.50,
    },
}


@dataclass(frozen=True)
class FastTrackCoreResult:
    physical_manifest: Path
    case_manifest: Path
    summary_path: Path
    selected_events: int
    selected_cases: int
    selected_physical_runs: int


@dataclass(frozen=True)
class StageDecision:
    stage: str
    passed: bool
    reasons: tuple[str, ...]
    next_action: str
    evidence_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "next_action": self.next_action,
            "evidence_path": self.evidence_path,
        }


@dataclass(frozen=True)
class FastTrackAudit:
    passed_through: str | None
    next_stage: str | None
    complete: bool
    decisions: tuple[StageDecision, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": CONTRACT_ID,
            "formal_contract_id": FORMAL_CONTRACT_ID,
            "development_only": True,
            "formal_authorization": False,
            "passed_through": self.passed_through,
            "next_stage": self.next_stage,
            "complete": self.complete,
            "decisions": [x.as_dict() for x in self.decisions],
        }


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def _json_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parsed = json.loads(str(value))
    return [str(x) for x in parsed]


def _stable_rank(text: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}::{text}".encode("utf-8")).hexdigest()


def _spread_take(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if len(frame) <= n:
        return frame.copy()
    idx = np.linspace(0, len(frame) - 1, num=n)
    idx = np.unique(np.rint(idx).astype(int))
    return frame.iloc[idx].copy()


def _assign_splits(group_keys: list[str]) -> dict[str, str]:
    keys = list(group_keys)
    n = len(keys)
    if n < 3:
        return {key: "train" for key in keys}
    n_test = max(1, int(round(0.20 * n)))
    n_val = max(1, int(round(0.20 * n)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    n_train = n - n_val - n_test
    result: dict[str, str] = {}
    for i, key in enumerate(keys):
        if i < n_train:
            result[key] = "train"
        elif i < n_train + n_val:
            result[key] = "validation"
        else:
            result[key] = "holdout"
    return result


def select_fasttrack_core(
    *,
    physical_inventory: str | Path,
    case_inventory: str | Path,
    output_dir: str | Path,
    max_events: int = 16,
    cases_per_event: int = 3,
    seed: int = 42,
    include_consumed_development: bool = True,
) -> FastTrackCoreResult:
    """Select a small representative four-reference core without full R0.2.

    Selection uses only R0.1 metadata.  Full finite checks and numeric alignment
    are deliberately deferred to the selected subset, which is the main compute
    saving of the fast-track path.
    """
    physical = _read_table(physical_inventory)
    cases = _read_table(case_inventory)
    if physical.empty or cases.empty:
        raise ValueError("R0.1 inventory is empty")

    required = {
        "case_uid",
        "event_id",
        "rainfall_sha256",
        "branch_physical_ids",
        "four_reference_complete",
        "core_trajectory_targets",
        "classification",
        "source_role",
        "domain_id",
    }
    missing = required - set(cases.columns)
    if missing:
        raise KeyError(f"case inventory missing columns: {sorted(missing)}")

    eligible = cases.copy()
    eligible = eligible[eligible["four_reference_complete"].fillna(False).astype(bool)]
    eligible = eligible[eligible["core_trajectory_targets"].fillna(False).astype(bool)]
    eligible = eligible[eligible["source_role"].astype(str) != "reserved_evaluation"]
    if not include_consumed_development:
        eligible = eligible[eligible["source_role"].astype(str) != "consumed_development"]
    eligible = eligible[eligible["domain_id"].fillna("").astype(str).str.startswith("target_no_dwf")]
    eligible = eligible[
        eligible["classification"].astype(str).isin(
            ["FULL_REUSE", "REUSE_AFTER_EXTRACTION", "PARTIAL_AUX_REUSE"]
        )
    ]
    if eligible.empty:
        raise RuntimeError("no metadata-eligible four-reference cases for fast-track")

    rainfall = eligible["rainfall_sha256"].fillna("").astype(str)
    event = eligible["event_id"].fillna("").astype(str)
    eligible["fasttrack_group"] = np.where(rainfall.str.len() > 0, rainfall, event)
    eligible = eligible[eligible["fasttrack_group"].astype(str).str.len() > 0].copy()
    if eligible.empty:
        raise RuntimeError("eligible cases have no rainfall/event grouping key")

    groups = sorted(
        eligible["fasttrack_group"].unique().tolist(),
        key=lambda x: _stable_rank(str(x), seed),
    )[: max(1, int(max_events))]
    splits = _assign_splits(groups)

    selected_parts: list[pd.DataFrame] = []
    for group in groups:
        part = eligible[eligible["fasttrack_group"] == group].copy()
        if "checkpoint_min" in part.columns:
            part["_checkpoint_sort"] = pd.to_numeric(part["checkpoint_min"], errors="coerce")
            part = part.sort_values(["_checkpoint_sort", "case_uid"], na_position="last")
        else:
            part = part.sort_values(["case_uid"])
        part = _spread_take(part, int(cases_per_event))
        part["fasttrack_split"] = splits[str(group)]
        selected_parts.append(part.drop(columns=["_checkpoint_sort"], errors="ignore"))
    selected_cases = pd.concat(selected_parts, ignore_index=True)

    physical_ids: set[str] = set()
    for value in selected_cases["branch_physical_ids"]:
        physical_ids.update(_json_ids(value))
    selected_physical = physical[
        physical["physical_identity_sha256"].astype(str).isin(physical_ids)
    ].copy()
    if selected_physical.empty:
        raise RuntimeError("selected cases did not resolve to physical runs")

    output_dir = Path(output_dir)
    physical_path = _write_table(selected_physical, output_dir / "physical_run_inventory.parquet")
    case_path = _write_table(selected_cases, output_dir / "target_coverage_by_case.csv")
    summary = {
        "contract_id": CONTRACT_ID,
        "stage": "core_pool_selection",
        "development_only": True,
        "formal_authorization": False,
        "selection_source": "R0.1_metadata_only",
        "max_events": int(max_events),
        "cases_per_event": int(cases_per_event),
        "seed": int(seed),
        "selected_independent_groups": int(selected_cases["fasttrack_group"].nunique()),
        "selected_cases": int(len(selected_cases)),
        "selected_physical_runs": int(len(selected_physical)),
        "split_counts": {
            str(k): int(v)
            for k, v in selected_cases["fasttrack_split"].value_counts().to_dict().items()
        },
        "reserved_evaluation_included": False,
    }
    summary_path = output_dir / "fasttrack_core_selection.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return FastTrackCoreResult(
        physical_manifest=physical_path,
        case_manifest=case_path,
        summary_path=summary_path,
        selected_events=int(selected_cases["fasttrack_group"].nunique()),
        selected_cases=int(len(selected_cases)),
        selected_physical_runs=int(len(selected_physical)),
    )


def targeted_finite_audit(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    output_manifest: str | Path | None = None,
) -> pd.DataFrame:
    """Run the expensive finite scan only on the selected fast-track core."""
    project_root = Path(project_root)
    frame = _read_table(physical_manifest).copy()
    graph = _load_graph_topology(project_root)
    node_ids = list(graph["node_ids"])
    facility_ids = _load_engineering36_ids(project_root)
    nodes, _ = _parse_inp_topology(project_root / "data" / "wuhan_v8_storage_retrofit.inp")
    storage_ids = [str(x) for x in nodes.loc[nodes["node_type"] == "storage", "node_id"].tolist()]
    outfall_ids = [str(x) for x in nodes.loc[nodes["node_type"] == "outfall", "node_id"].tolist()]
    expected = _target_columns(
        node_ids=node_ids,
        storage_ids=storage_ids,
        facility_ids=facility_ids,
        outfall_ids=outfall_ids,
    )

    results: list[bool] = []
    for row in frame.itertuples(index=False):
        path = Path(str(getattr(row, "detail_path")))
        try:
            results.append(bool(path.exists() and _finite_target_check(path, expected)))
        except Exception:
            results.append(False)
    frame["available_finite_checked"] = True
    frame["available_finite_pass"] = results
    target = Path(output_manifest) if output_manifest is not None else Path(physical_manifest)
    _write_table(frame, target)
    return frame


def prepare_fasttrack_core(
    *,
    project_root: str | Path,
    r01_audit_dir: str | Path,
    output_dir: str | Path,
    max_events: int = 16,
    cases_per_event: int = 3,
    seed: int = 42,
    min_events: int = 8,
    min_aligned_cases: int = 12,
) -> dict[str, Any]:
    """Select, finite-audit, align and materialize the small reusable core."""
    r01_audit_dir = Path(r01_audit_dir)
    output_dir = Path(output_dir)
    core = select_fasttrack_core(
        physical_inventory=r01_audit_dir / "physical_run_inventory.parquet",
        case_inventory=r01_audit_dir / "target_coverage_by_case.csv",
        output_dir=output_dir,
        max_events=max_events,
        cases_per_event=cases_per_event,
        seed=seed,
    )
    finite = targeted_finite_audit(
        project_root=project_root,
        physical_manifest=core.physical_manifest,
    )
    alignment_path = output_dir / "case_alignment_audit.csv"
    alignment = audit_case_alignment(
        project_root=project_root,
        physical_inventory=core.physical_manifest,
        case_inventory=core.case_manifest,
        output_path=alignment_path,
    )
    reusable = build_reusable_paper_pool(
        physical_inventory=core.physical_manifest,
        case_inventory=core.case_manifest,
        alignment_inventory=alignment_path,
        output_physical_manifest=output_dir / "reusable_pool_manifest.parquet",
        output_case_manifest=output_dir / "reusable_case_manifest.parquet",
        audit_output=output_dir / "reusable_pool_summary.json",
        include_source_domain=False,
        include_consumed_development=True,
        require_finite_audit=True,
    )

    finite_fraction = float(finite["available_finite_pass"].mean()) if len(finite) else 0.0
    aligned = int(
        (
            alignment["same_state_numeric_pass"].fillna(False).astype(bool)
            & alignment["same_forcing_pass"].fillna(False).astype(bool)
        ).sum()
    )
    passed = bool(
        core.selected_events >= int(min_events)
        and aligned >= int(min_aligned_cases)
        and finite_fraction >= 0.95
    )
    evidence = {
        "contract_id": CONTRACT_ID,
        "formal_contract_id": FORMAL_CONTRACT_ID,
        "stage": "core_pool",
        "status": "pass" if passed else "fail",
        "development_only": True,
        "formal_authorization": False,
        "metrics": {
            "independent_rainfall_groups": int(core.selected_events),
            "selected_cases": int(core.selected_cases),
            "selected_physical_runs": int(core.selected_physical_runs),
            "finite_pass_fraction": finite_fraction,
            "aligned_cases": aligned,
            "reusable_physical_rows": int(reusable.physical_row_count),
            "reusable_case_rows": int(reusable.case_row_count),
        },
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return evidence


def diagnose_learning_curve(
    points: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    pass_threshold: float,
) -> str:
    """Classify whether a failed pilot is data-limited or model/target-limited."""
    valid: list[tuple[float, float, float]] = []
    for point in points:
        try:
            size = float(point["train_groups"])
            train = float(point["train_score"])
            val = float(point[score_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(size) and np.isfinite(train) and np.isfinite(val):
            valid.append((size, train, val))
    if not valid:
        return "insufficient_learning_curve_evidence"
    valid.sort(key=lambda x: x[0])
    first = valid[0]
    last = valid[-1]
    if last[2] >= float(pass_threshold):
        return "learnable_at_current_core"
    improvement = last[2] - first[2]
    gap = last[1] - last[2]
    if improvement >= 0.05 and gap >= 0.08:
        return "data_limited_expand_targeted_evidence"
    if last[1] < float(pass_threshold) and gap <= 0.05 and improvement < 0.03:
        return "model_or_target_limited_do_not_bulk_expand"
    return "mixed_or_uncertain_targeted_diagnosis_needed"


def _thresholds(stage: str, overrides: Mapping[str, Any] | None) -> dict[str, float]:
    base = dict(DEFAULT_THRESHOLDS[stage])
    if overrides:
        base.update({str(k): float(v) for k, v in overrides.items()})
    return base


def evaluate_stage_payload(
    stage: str,
    payload: Mapping[str, Any],
    *,
    threshold_overrides: Mapping[str, Any] | None = None,
) -> StageDecision:
    if stage not in STAGES:
        raise KeyError(stage)
    reasons: list[str] = []
    if payload.get("contract_id") != CONTRACT_ID:
        reasons.append("wrong_fasttrack_contract")
    if payload.get("stage") != stage:
        reasons.append("stage_name_mismatch")
    if payload.get("development_only") is not True:
        reasons.append("fasttrack_must_be_development_only")
    if payload.get("formal_authorization") is not False:
        reasons.append("fasttrack_must_not_authorize_formal")
    if payload.get("status") != "pass":
        reasons.append("stage_status_not_pass")

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
        reasons.append("metrics_missing")
    t = _thresholds(stage, threshold_overrides)

    def require_min(key: str) -> None:
        try:
            value = float(metrics.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value) or value < float(t[key]):
            reasons.append(f"{key}_below_fasttrack_threshold")

    if stage == "core_pool":
        for key in ("independent_rainfall_groups", "aligned_cases", "finite_pass_fraction"):
            require_min(key)
    elif stage == "step1_gat":
        for key in ("val_nse_median", "priority_nse_median"):
            require_min(key)
    elif stage == "step2_surrogate":
        for key in (
            "pfv_direction_accuracy",
            "tfv_direction_accuracy",
            "peak_direction_accuracy",
            "safe_candidate_recall",
        ):
            require_min(key)
        try:
            false_safe = float(metrics.get("false_safe_rate", float("nan")))
        except (TypeError, ValueError):
            false_safe = float("nan")
        if not np.isfinite(false_safe) or false_safe > float(t["false_safe_rate_max"]):
            reasons.append("false_safe_rate_above_fasttrack_threshold")
    elif stage == "step3_mpc":
        for key in ("states_evaluated", "safe_selection_precision", "good_candidate_recall"):
            require_min(key)
    elif stage == "step4_micro_closed_loop":
        for key in ("event_count", "pfv_noninferior_rate", "peak_noninferior_rate", "tfv_improved_rate"):
            require_min(key)

    next_action = "continue"
    if reasons and stage in {"step1_gat", "step2_surrogate"}:
        curve = payload.get("learning_curve", [])
        score_key = "val_score"
        pass_key = "val_nse_median" if stage == "step1_gat" else "pfv_direction_accuracy"
        next_action = diagnose_learning_curve(
            curve if isinstance(curve, Sequence) else [],
            score_key=score_key,
            pass_threshold=float(t[pass_key]),
        )
    elif reasons:
        next_action = "targeted_repair_before_more_compute"
    elif stage == STAGES[-1]:
        next_action = "expand_evidence_for_formal_workflow"

    return StageDecision(
        stage=stage,
        passed=not reasons,
        reasons=tuple(reasons),
        next_action=next_action,
        evidence_path="",
    )


def audit_fasttrack_workflow(
    output_root: str | Path,
    *,
    threshold_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> FastTrackAudit:
    root = Path(output_root)
    decisions: list[StageDecision] = []
    passed_through: str | None = None
    next_stage: str | None = None
    for stage in STAGES:
        path = root / EVIDENCE_RELATIVE_PATHS[stage]
        if not path.exists():
            decision = StageDecision(
                stage=stage,
                passed=False,
                reasons=("evidence_missing",),
                next_action="run_stage",
                evidence_path=str(path),
            )
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                override = None if threshold_overrides is None else threshold_overrides.get(stage)
                raw = evaluate_stage_payload(stage, payload, threshold_overrides=override)
                decision = StageDecision(
                    stage=raw.stage,
                    passed=raw.passed,
                    reasons=raw.reasons,
                    next_action=raw.next_action,
                    evidence_path=str(path),
                )
            except Exception as exc:
                decision = StageDecision(
                    stage=stage,
                    passed=False,
                    reasons=(f"evidence_unreadable:{type(exc).__name__}",),
                    next_action="repair_evidence",
                    evidence_path=str(path),
                )
        decisions.append(decision)
        if not decision.passed:
            next_stage = stage
            break
        passed_through = stage
    complete = passed_through == STAGES[-1]
    return FastTrackAudit(
        passed_through=passed_through,
        next_stage=None if complete else next_stage,
        complete=complete,
        decisions=tuple(decisions),
    )
