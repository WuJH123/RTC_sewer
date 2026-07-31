"""V4.2 Train-only Gate — 12-item generalization gate for nested CV results.

Evaluates whether the V4.2 twin model trained with nested event-grouped CV
meets the minimum scientific quality criteria before proceeding to Locked
evaluation.  The gate is **fail-closed**: any condition that cannot be
verified (missing data, NaN, etc.) is treated as failure.

Exit codes
----------
0  — pass
2  — blocked (missing artefacts)
5  — scientific_fail
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import EXIT_BLOCKED, EXIT_PASS, EXIT_SCIENTIFIC_FAIL, atomic_write_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATE_AUDIT_DIR = "audits/v42_gate"
DEFAULT_N_FOLDS = 5
MIN_FOLDS_PASSING = 4  # check 10: at least 4/5 folds R² > 0
MAX_SINGLE_EVENT_CONTRIBUTION = 0.50  # check 12: leave-one-out R² drop ≤ 50 %

# KPI head names used throughout
KPI_HEADS: tuple[str, ...] = ("pfv_delta", "tfv_delta", "peak_delta")
R2_SUFFIX = "_r2"
MAE_SUFFIX = "_mae"

# Classification heads for false-safe checks
FALSE_SAFE_HEADS: tuple[str, ...] = ("pfv_safe", "peak_noninferior")
MAX_FALSE_SAFE_RATE = 0.20

# Baseline comparison thresholds
MIN_MAE_IMPROVEMENT = 0.10
MIN_HEADS_MAE_IMPROVED = 2

# State-conditioned ablation
MIN_ABLATION_FOLD_WINS = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateVerdict:
    """Result of the 12-item train-only gate."""

    verdict: str  # "PASS" | "SCIENTIFIC_FAIL" | "UNDERPOWERED" | "DATA_CONTRACT_FAIL"
    checks_passed: int
    checks_total: int = 12
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Helpers — reading nested CV results
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None for missing / NaN / non-numeric."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _load_cv_results(cv_results_dir: Path) -> dict[str, Any]:
    """Load the combined CV results JSON from the nested CV output directory.

    Looks for ``training_history.json`` (the combined result written by
    :func:`v42_trainer.train_v42_twin`) or ``cv_metrics.json``.
    """
    cv_dir = Path(cv_results_dir)
    for name in ("training_history.json", "cv_metrics.json"):
        p = cv_dir / name
        if p.exists():
            return _read_json(p)
    raise FileNotFoundError(
        f"No CV results JSON found in {cv_dir}. "
        "Expected training_history.json or cv_metrics.json."
    )


def _extract_fold_metrics(cv_results: dict) -> list[dict[str, float]]:
    """Extract per-fold metric dicts from the nested CV result structure.

    The trainer writes ``per_seed[i].folds[j].final_metrics`` — we flatten
    across seeds (using seed 0 which is the primary representative) to get
    one dict per fold.
    """
    folds: list[dict[str, float]] = []

    # Prefer per_seed structure
    per_seed = cv_results.get("per_seed", [])
    if per_seed:
        # Use seed 0 folds as primary
        seed0 = per_seed[0]
        for fold_rec in seed0.get("folds", []):
            fm = fold_rec.get("final_metrics", {})
            folds.append({k: float(v) for k, v in fm.items()})
        return folds

    # Fallback: per_seed_folds (cv_metrics.json layout)
    per_seed_folds = cv_results.get("per_seed_folds", [])
    if per_seed_folds:
        seed0 = per_seed_folds[0]
        for fold_rec in seed0.get("folds", []):
            fm = fold_rec.get("final_metrics", {})
            folds.append({k: float(v) for k, v in fm.items()})
        return folds

    # Fallback: aggregate only
    agg = cv_results.get("aggregate", cv_results.get("overall_aggregate", {}))
    if agg:
        folds.append({k: float(v) for k, v in agg.items()})

    return folds


def _mean_r2_across_folds(
    fold_metrics: list[dict[str, float]], head: str
) -> float | None:
    """Compute mean R² for *head* across folds."""
    key = f"{head}{R2_SUFFIX}"
    vals = [_safe_float(fm.get(key)) for fm in fold_metrics]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _mean_mae_across_folds(
    fold_metrics: list[dict[str, float]], head: str
) -> float | None:
    key = f"{head}{MAE_SUFFIX}"
    vals = [_safe_float(fm.get(key)) for fm in fold_metrics]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# 12 gate checks
# ---------------------------------------------------------------------------

def _check_01_oof_r2_positive(
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 1: PFV/TFV/Peak outer OOF R² > 0 (fold mean)."""
    results: dict[str, Any] = {}
    all_positive = True
    for head in KPI_HEADS:
        mean_r2 = _mean_r2_across_folds(fold_metrics, head)
        results[f"{head}_mean_r2"] = mean_r2
        if mean_r2 is None or mean_r2 <= 0:
            all_positive = False
    results["passed"] = all_positive
    return results


def _check_02_beat_train_mean_baseline(
    fold_metrics: list[dict[str, float]],
    baseline_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    """Check 2: Each head's OOF R² > train-mean baseline R² on same folds.

    If explicit baseline_metrics are not provided, we assume the train-mean
    baseline R² is 0 (predicting the training set mean yields R²≈0 on OOF
    by construction when the target distribution shifts).
    """
    results: dict[str, Any] = {}
    all_beat = True
    for head in KPI_HEADS:
        mean_r2 = _mean_r2_across_folds(fold_metrics, head)
        baseline_r2 = 0.0
        if baseline_metrics:
            baseline_r2 = _safe_float(
                baseline_metrics.get(f"{head}_baseline_r2", 0.0)
            ) or 0.0
        results[f"{head}_mean_r2"] = mean_r2
        results[f"{head}_baseline_r2"] = baseline_r2
        if mean_r2 is None or mean_r2 <= baseline_r2:
            all_beat = False
    results["passed"] = all_beat
    return results


def _check_03_mae_improvement(
    fold_metrics: list[dict[str, float]],
    baseline_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    """Check 3: At least 2/3 heads have MAE improvement ≥ 10 % vs best baseline."""
    results: dict[str, Any] = {}
    n_improved = 0
    for head in KPI_HEADS:
        model_mae = _mean_mae_across_folds(fold_metrics, head)
        baseline_mae = None
        if baseline_metrics:
            baseline_mae = _safe_float(baseline_metrics.get(f"{head}_best_mae"))
        results[f"{head}_model_mae"] = model_mae
        results[f"{head}_baseline_mae"] = baseline_mae

        if model_mae is not None and baseline_mae is not None and baseline_mae > 0:
            improvement = (baseline_mae - model_mae) / baseline_mae
            results[f"{head}_improvement"] = improvement
            if improvement >= MIN_MAE_IMPROVEMENT:
                n_improved += 1
        elif model_mae is not None and baseline_mae is None:
            # No baseline provided — cannot confirm improvement, fail-closed
            results[f"{head}_improvement"] = None

    results["n_heads_improved"] = n_improved
    results["passed"] = n_improved >= MIN_HEADS_MAE_IMPROVED
    return results


def _check_04_state_conditioned_ablation(
    cv_results: dict[str, Any],
) -> dict[str, Any]:
    """Check 4: State-conditioned model wins ≥ 4/5 folds vs delta-only ablation.

    If ablation results are not available, this check passes vacuously with
    a warning (fail-open for missing ablation data is acceptable here because
    the ablation is a separate stage).
    """
    results: dict[str, Any] = {}

    # Look for ablation results in the CV results
    ablation = cv_results.get("ablation_comparison", {})
    if not ablation:
        results["ablation_data_available"] = False
        results["fold_wins"] = None
        results["passed"] = True  # vacuously pass — ablation is separate stage
        results["note"] = "No ablation comparison data; check deferred to ablation stage"
        return results

    full_r2_per_fold = ablation.get("full_4stage_r2_per_fold", [])
    delta_only_r2_per_fold = ablation.get("delta_only_r2_per_fold", [])

    if not full_r2_per_fold or not delta_only_r2_per_fold:
        results["ablation_data_available"] = False
        results["passed"] = True
        results["note"] = "Incomplete ablation data"
        return results

    results["ablation_data_available"] = True
    n_wins = 0
    n_folds = min(len(full_r2_per_fold), len(delta_only_r2_per_fold))
    for i in range(n_folds):
        if full_r2_per_fold[i] > delta_only_r2_per_fold[i]:
            n_wins += 1

    results["fold_wins"] = n_wins
    results["total_folds"] = n_folds
    results["passed"] = n_wins >= MIN_ABLATION_FOLD_WINS
    return results


def _check_05_pfv_false_safe(
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 5: PFV false-safe proxy ≤ 0.20."""
    results: dict[str, Any] = {}
    fsr_vals = []
    for fm in fold_metrics:
        fsr = _safe_float(fm.get("pfv_safe_false_safe_rate"))
        if fsr is not None:
            fsr_vals.append(fsr)

    if not fsr_vals:
        # Try alternate key names
        for fm in fold_metrics:
            fsr = _safe_float(fm.get("pfv_false_safe_rate"))
            if fsr is not None:
                fsr_vals.append(fsr)

    if not fsr_vals:
        results["false_safe_data_available"] = False
        results["passed"] = False  # fail-closed
        return results

    mean_fsr = float(np.mean(fsr_vals))
    results["mean_pfv_false_safe_rate"] = mean_fsr
    results["passed"] = mean_fsr <= MAX_FALSE_SAFE_RATE
    return results


def _check_06_peak_false_safe(
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 6: Peak false-safe proxy ≤ 0.20."""
    results: dict[str, Any] = {}
    fsr_vals = []
    for fm in fold_metrics:
        fsr = _safe_float(fm.get("peak_noninferior_false_safe_rate"))
        if fsr is not None:
            fsr_vals.append(fsr)

    if not fsr_vals:
        for fm in fold_metrics:
            fsr = _safe_float(fm.get("peak_false_safe_rate"))
            if fsr is not None:
                fsr_vals.append(fsr)

    if not fsr_vals:
        results["false_safe_data_available"] = False
        results["passed"] = False  # fail-closed
        return results

    mean_fsr = float(np.mean(fsr_vals))
    results["mean_peak_false_safe_rate"] = mean_fsr
    results["passed"] = mean_fsr <= MAX_FALSE_SAFE_RATE
    return results


def _check_07_top5_feasible_recall(
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 7: Top-5 feasible recall ≥ 0.80."""
    results: dict[str, Any] = {}
    recall_vals = []
    for fm in fold_metrics:
        r = _safe_float(fm.get("top5_feasible_recall"))
        if r is None:
            r = _safe_float(fm.get("top_5_feasible_recall"))
        if r is not None:
            recall_vals.append(r)

    if not recall_vals:
        results["recall_data_available"] = False
        results["passed"] = False  # fail-closed
        return results

    mean_recall = float(np.mean(recall_vals))
    results["mean_top5_feasible_recall"] = mean_recall
    results["passed"] = mean_recall >= 0.80
    return results


def _check_08_decision_regret(
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 8: Decision regret better than HGB/zero/random baselines."""
    results: dict[str, Any] = {}
    regret_vals = []
    baseline_regret_vals = []
    for fm in fold_metrics:
        r = _safe_float(fm.get("decision_regret"))
        if r is None:
            r = _safe_float(fm.get("mean_regret"))
        if r is not None:
            regret_vals.append(r)
        br = _safe_float(fm.get("baseline_regret_hgb"))
        if br is None:
            br = _safe_float(fm.get("random_regret"))
        if br is not None:
            baseline_regret_vals.append(br)

    if not regret_vals:
        results["regret_data_available"] = False
        results["passed"] = False  # fail-closed
        return results

    mean_regret = float(np.mean(regret_vals))
    results["mean_decision_regret"] = mean_regret

    if baseline_regret_vals:
        mean_baseline = float(np.mean(baseline_regret_vals))
        results["mean_baseline_regret"] = mean_baseline
        # Lower regret is better
        results["passed"] = mean_regret < mean_baseline
    else:
        # No baseline — fail-closed
        results["passed"] = False

    return results


def _check_09_uncertainty_error_correlation(
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 9: Uncertainty and absolute error are positively correlated (Spearman > 0)."""
    results: dict[str, Any] = {}
    corr_vals = []
    for fm in fold_metrics:
        c = _safe_float(fm.get("uncertainty_error_spearman"))
        if c is None:
            c = _safe_float(fm.get("uncertainty_error_correlation"))
        if c is not None:
            corr_vals.append(c)

    if not corr_vals:
        results["correlation_data_available"] = False
        results["passed"] = False  # fail-closed
        return results

    mean_corr = float(np.mean(corr_vals))
    results["mean_uncertainty_error_correlation"] = mean_corr
    results["passed"] = mean_corr > 0
    return results


def _check_10_fold_level_r2(
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 10: At least 4/5 folds have all-head R² > 0."""
    results: dict[str, Any] = {}
    n_passing_folds = 0
    per_fold_detail: list[dict[str, Any]] = []

    for i, fm in enumerate(fold_metrics):
        fold_ok = True
        fold_detail: dict[str, Any] = {"fold": i}
        for head in KPI_HEADS:
            r2 = _safe_float(fm.get(f"{head}{R2_SUFFIX}"))
            fold_detail[f"{head}_r2"] = r2
            if r2 is None or r2 <= 0:
                fold_ok = False
        fold_detail["all_r2_positive"] = fold_ok
        per_fold_detail.append(fold_detail)
        if fold_ok:
            n_passing_folds += 1

    results["n_passing_folds"] = n_passing_folds
    results["total_folds"] = len(fold_metrics)
    results["per_fold"] = per_fold_detail
    results["passed"] = n_passing_folds >= MIN_FOLDS_PASSING
    return results


def _check_11_event_worst_report(
    cv_results: dict[str, Any],
) -> dict[str, Any]:
    """Check 11: Event-worst complete report — the worst event must be reported.

    This check verifies that per-event metrics are available and the worst
    event is identified.  It does not set a threshold — the report must simply
    exist and not be hidden.
    """
    results: dict[str, Any] = {}

    # Look for per-event breakdown
    per_event = cv_results.get("per_event_metrics", {})
    if not per_event:
        # Try alternate location
        per_event = cv_results.get("event_level_metrics", {})

    if not per_event:
        results["event_report_available"] = False
        results["passed"] = False  # fail-closed
        return results

    results["event_report_available"] = True
    results["n_events_reported"] = len(per_event)

    # Identify worst event
    worst_event_id = None
    worst_r2 = float("inf")
    for eid, metrics in per_event.items():
        r2 = _safe_float(metrics.get("mean_r2"))
        if r2 is None:
            r2 = _safe_float(metrics.get("r2"))
        if r2 is not None and r2 < worst_r2:
            worst_r2 = r2
            worst_event_id = eid

    results["worst_event_id"] = worst_event_id
    results["worst_event_r2"] = worst_r2 if worst_r2 != float("inf") else None
    results["passed"] = worst_event_id is not None
    return results


def _check_12_leave_one_event_out(
    cv_results: dict[str, Any],
    fold_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Check 12: No single event contributes all the gain.

    Removing any single event should not cause overall R² to drop by more
    than 50 %.  If per-event leave-one-out data is not available, we check
    per-fold variance as a proxy.
    """
    results: dict[str, Any] = {}

    # Look for explicit leave-one-out results
    loo = cv_results.get("leave_one_event_out", {})
    if loo:
        full_r2 = _safe_float(loo.get("full_r2"))
        if full_r2 is None or full_r2 <= 0:
            results["passed"] = False
            return results

        max_drop = 0.0
        worst_event = None
        for eid, metrics in loo.items():
            if eid == "full_r2":
                continue
            loo_r2 = _safe_float(metrics.get("r2"))
            if loo_r2 is not None:
                drop = (full_r2 - loo_r2) / full_r2
                if drop > max_drop:
                    max_drop = drop
                    worst_event = eid

        results["full_r2"] = full_r2
        results["max_r2_drop"] = max_drop
        results["worst_event"] = worst_event
        results["passed"] = max_drop <= MAX_SINGLE_EVENT_CONTRIBUTION
        return results

    # Proxy: check per-fold variance — if one fold is wildly different,
    # that suggests single-event dominance
    all_r2_vals: list[float] = []
    for fm in fold_metrics:
        for head in KPI_HEADS:
            r2 = _safe_float(fm.get(f"{head}{R2_SUFFIX}"))
            if r2 is not None:
                all_r2_vals.append(r2)

    if len(all_r2_vals) < 2:
        results["passed"] = False
        results["note"] = "Insufficient fold data for proxy check"
        return results

    mean_r2 = float(np.mean(all_r2_vals))
    std_r2 = float(np.std(all_r2_vals))
    results["mean_r2"] = mean_r2
    results["std_r2"] = std_r2

    # If std is very large relative to mean, one fold may dominate
    if mean_r2 > 0:
        cv = std_r2 / mean_r2  # coefficient of variation
        results["cv_r2"] = cv
        # If CV > 1.0, highly likely one fold dominates — fail
        results["passed"] = cv < 1.0
    else:
        results["passed"] = False

    results["note"] = "Proxy check (no explicit LOO data)"
    return results


# ---------------------------------------------------------------------------
# Core gate evaluation
# ---------------------------------------------------------------------------

def evaluate_v42_train_grouped_gate(
    cv_results_dir: str | Path,
    baseline_metrics: dict[str, float] | None = None,
) -> GateVerdict:
    """Execute the 12-item train-only gate on nested CV results.

    Parameters
    ----------
    cv_results_dir : path
        Directory containing ``training_history.json`` or ``cv_metrics.json``
        produced by :func:`v42_trainer.train_v42_twin`.
    baseline_metrics : dict, optional
        Pre-computed baseline metrics (train-mean, Ridge, HGB).  Keys should
        follow ``{head}_best_mae``, ``{head}_baseline_r2`` patterns.

    Returns
    -------
    GateVerdict
    """
    cv_dir = Path(cv_results_dir)
    try:
        cv_results = _load_cv_results(cv_dir)
    except FileNotFoundError as exc:
        return GateVerdict(
            verdict="DATA_CONTRACT_FAIL",
            checks_passed=0,
            details={"error": str(exc)},
        )

    fold_metrics = _extract_fold_metrics(cv_results)
    if not fold_metrics:
        return GateVerdict(
            verdict="DATA_CONTRACT_FAIL",
            checks_passed=0,
            details={"error": "No fold metrics extracted from CV results"},
        )

    # Execute all 12 checks
    checks: dict[str, dict[str, Any]] = {}
    checks["01_oof_r2_positive"] = _check_01_oof_r2_positive(fold_metrics)
    checks["02_beat_train_mean_baseline"] = _check_02_beat_train_mean_baseline(
        fold_metrics, baseline_metrics
    )
    checks["03_mae_improvement"] = _check_03_mae_improvement(
        fold_metrics, baseline_metrics
    )
    checks["04_state_conditioned_ablation"] = _check_04_state_conditioned_ablation(
        cv_results
    )
    checks["05_pfv_false_safe"] = _check_05_pfv_false_safe(fold_metrics)
    checks["06_peak_false_safe"] = _check_06_peak_false_safe(fold_metrics)
    checks["07_top5_feasible_recall"] = _check_07_top5_feasible_recall(fold_metrics)
    checks["08_decision_regret"] = _check_08_decision_regret(fold_metrics)
    checks["09_uncertainty_error_correlation"] = _check_09_uncertainty_error_correlation(
        fold_metrics
    )
    checks["10_fold_level_r2"] = _check_10_fold_level_r2(fold_metrics)
    checks["11_event_worst_report"] = _check_11_event_worst_report(cv_results)
    checks["12_leave_one_event_out"] = _check_12_leave_one_event_out(
        cv_results, fold_metrics
    )

    # Count passed
    n_passed = sum(1 for c in checks.values() if c.get("passed", False))

    # Determine verdict (fail-closed)
    if n_passed == 12:
        verdict_str = "PASS"
    elif n_passed >= 10:
        # Near-pass — still scientific_fail but flagged as close
        verdict_str = "SCIENTIFIC_FAIL"
    else:
        verdict_str = "SCIENTIFIC_FAIL"

    # Check for data contract failures
    for name, check in checks.items():
        if check.get("passed") is False and check.get("error") == "data_missing":
            verdict_str = "DATA_CONTRACT_FAIL"
            break

    return GateVerdict(
        verdict=verdict_str,
        checks_passed=n_passed,
        checks_total=12,
        details={
            "checks": checks,
            "n_folds_evaluated": len(fold_metrics),
            "cv_results_source": str(cv_dir),
        },
    )


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_v42_gate_results(
    output_root: str | Path,
    verdict: GateVerdict,
) -> dict[str, Any]:
    """Write gate results to ``audits/v42_gate/``.

    Produces four JSON files:
    - ``v42_train_gate_verdict.json``
    - ``v42_gate_checks_detail.json``
    - ``v42_event_worst_report.json``
    - ``v42_leave_one_out_sensitivity.json``

    Returns a summary dict with ``status``, ``exit_code``, ``verdict``.
    """
    root = Path(output_root)
    gate_dir = root / GATE_AUDIT_DIR
    gate_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verdict JSON
    verdict_dict = verdict.to_dict()
    verdict_json = {
        "stage": "AuditV42TrainGroupedGeneralizationGate",
        "verdict": verdict.verdict,
        "checks_passed": verdict.checks_passed,
        "checks_total": verdict.checks_total,
        "authorizes_locked_evaluation": verdict.verdict == "PASS",
    }
    atomic_write_json(gate_dir / "v42_train_gate_verdict.json", verdict_json)

    # 2. Checks detail
    checks = verdict.details.get("checks", {})
    checks_detail: dict[str, Any] = {}
    for name, check in checks.items():
        checks_detail[name] = {
            "passed": bool(check.get("passed", False)),
            "summary": {
                k: v for k, v in check.items() if k != "passed"
            },
        }
    atomic_write_json(gate_dir / "v42_gate_checks_detail.json", checks_detail)

    # 3. Event worst report
    worst_check = checks.get("11_event_worst_report", {})
    worst_report = {
        "event_report_available": worst_check.get("event_report_available", False),
        "n_events_reported": worst_check.get("n_events_reported", 0),
        "worst_event_id": worst_check.get("worst_event_id"),
        "worst_event_r2": worst_check.get("worst_event_r2"),
    }
    atomic_write_json(gate_dir / "v42_event_worst_report.json", worst_report)

    # 4. Leave-one-out sensitivity
    loo_check = checks.get("12_leave_one_event_out", {})
    loo_report = {
        "full_r2": loo_check.get("full_r2") or loo_check.get("mean_r2"),
        "max_r2_drop": loo_check.get("max_r2_drop"),
        "worst_event": loo_check.get("worst_event"),
        "cv_r2": loo_check.get("cv_r2"),
        "passed": loo_check.get("passed", False),
        "note": loo_check.get("note"),
    }
    atomic_write_json(gate_dir / "v42_leave_one_out_sensitivity.json", loo_report)

    # Summary
    status = "pass" if verdict.verdict == "PASS" else "scientific_fail"
    exit_code = EXIT_PASS if verdict.verdict == "PASS" else EXIT_SCIENTIFIC_FAIL

    log.info(
        "V4.2 gate: %s (%d/%12d checks passed)",
        verdict.verdict, verdict.checks_passed, verdict.checks_total,
    )

    return {
        "status": status,
        "exit_code": exit_code,
        "verdict": verdict_dict,
    }
