"""V4.1 Compact rescue -- Phase-2 evaluation ops (spec sections 12, 14-16).

Pure-logic building blocks for the *brand-new independent* Calibration / Locked
evaluation of the compact V4.1 model:

* section 12 -- deterministic fresh evaluation-split planning from the 16 unused
  Reserve events (4 calibration / 8 locked / 4 accrual reserve), frozen *before*
  any new label exists and never chosen by predicted feasibility or old-Locked
  performance;
* section 14 -- calibrate the compact model on the *new* Calibration only, with a
  class-support gate that disables (never fakes) an under-supported probability
  head and falls back to a continuous UCB;
* section 15/16 -- one-shot Locked evaluation of the compact model plus the
  Predictive Generalization Gate verdict (pass / scientific_fail / underpowered).

No SWMM here: these functions take an already-built ``TrainingData`` and the
split names to use, so the same logic drives both the synthetic test fixture
and the real fresh evaluation data produced by the section-13 SWMM stages.
The old Calibration / old Locked are never read to pick anything.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
import pandas as pd

from .train_v4_metrics import (
    classification_metrics,
    decision_metrics,
    one_sided_conformal,
    regression_metrics,
    uncertainty_error_correlation,
    worst_event,
)

CONTINUOUS_HEADS = ("pfv", "tfv", "peak")
CLASSIFICATION_HEADS = (
    "pfv_safe",
    "tfv_improved",
    "peak_noninferior",
    "joint_noninferior",
)

# Conservative one-sided direction per continuous head: PFV increase beyond the
# prediction is unsafe (actual worse than predicted -> underprediction); TFV /
# Peak improvement *claims* larger than reality are unsafe (overprediction).
HEAD_DIRECTION = {
    "pfv": "underprediction",
    "tfv": "overprediction",
    "peak": "overprediction",
}

# Minimum per-class calibration support before a probability head is trusted.
MIN_CLASS_SUPPORT = 5
CONFORMAL_COVERAGE = 0.9


# ===========================================================================
# Section 12 -- fresh evaluation split plan (Reserve events only)
# ===========================================================================

def _frozen_event_order(event_ids: list[str]) -> list[str]:
    """Deterministic, outcome-independent order (content hash of the id).

    Never uses predicted feasibility or old-Locked performance to order the
    events; a stable hash of the event id alone freezes the assignment order.
    """
    return sorted(
        (str(e) for e in event_ids),
        key=lambda e: hashlib.sha256(e.encode("utf-8")).hexdigest(),
    )


def plan_fresh_evaluation_split(
    ledger: pd.DataFrame,
    *,
    counts: dict[str, int] | None = None,
    states_per_event: int = 5,
    candidates_per_state: int = 5,
) -> dict[str, Any]:
    """Allocate the 16 unused Reserve events to fresh calibration/locked/accrual.

    Returns a JSON-friendly plan freeze with per-split event lists, rainfall
    SHAs and the frozen sample budgets.  Fail-closed when fewer than the
    required Reserve events are available.
    """
    counts = dict(
        counts or {"v4.1_calibration": 4, "v4.1_locked": 8, "locked_accrual_reserve": 4}
    )
    required = {"event_id", "rainfall_sha256", "assigned_split"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger missing columns: {sorted(missing)}")
    reserve = ledger[ledger["assigned_split"].astype(str) == "reserve"].copy()
    reserve_events = reserve["event_id"].astype(str).tolist()
    total = int(sum(counts.values()))
    if len(reserve_events) < total:
        raise ValueError(
            f"need {total} reserve events for fresh evaluation, "
            f"found {len(reserve_events)}"
        )
    sha_by_event = dict(
        zip(reserve["event_id"].astype(str), reserve["rainfall_sha256"].astype(str))
    )
    ordered = _frozen_event_order(reserve_events)
    splits: dict[str, list[str]] = {}
    cursor = 0
    for split in ("v4.1_calibration", "v4.1_locked", "locked_accrual_reserve"):
        span = int(counts[split])
        splits[split] = ordered[cursor : cursor + span]
        cursor += span
    per_split_samples = {
        split: len(events) * states_per_event * candidates_per_state
        for split, events in splits.items()
    }
    plans = {
        split: [
            {
                "event_id": event,
                "rainfall_sha256": sha_by_event[event],
                "assign_split": split,
                "states_per_event": states_per_event,
                "candidates_per_state": candidates_per_state,
            }
            for event in events
        ]
        for split, events in splits.items()
    }
    order_sha = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
    return {
        "stage": "PlanV4CompactCalibrationLockedV1",
        "source": "reserve_events_only",
        "selection_rule": (
            "deterministic frozen hash order of event_id; never chosen by "
            "predicted feasibility or old-Locked performance"
        ),
        "frozen_before_any_new_label": True,
        "reads_old_locked_for_selection": False,
        "counts": {k: int(v) for k, v in counts.items()},
        "splits": splits,
        "per_split_samples": {k: int(v) for k, v in per_split_samples.items()},
        "plans": plans,
        "frozen_order": ordered,
        "frozen_order_sha256": order_sha,
        "states_per_event": states_per_event,
        "candidates_per_state": candidates_per_state,
    }


def audit_evaluation_plan(plan: dict[str, Any], ledger: pd.DataFrame) -> dict[str, Any]:
    """Verify the fresh plan is frozen, Reserve-only, fresh and disjoint."""
    splits = plan.get("splits", {})
    counts = plan.get("counts", {})
    ledger_indexed = ledger.set_index(ledger["event_id"].astype(str))
    all_events = [e for events in splits.values() for e in events]
    checks: dict[str, bool] = {}
    checks["counts_match"] = all(
        len(splits.get(split, [])) == int(counts.get(split, -1)) for split in counts
    )
    checks["no_event_overlap"] = len(all_events) == len(set(all_events))
    from_reserve = all(
        str(ledger_indexed.loc[e, "assigned_split"]) == "reserve"
        for e in all_events
        if e in ledger_indexed.index
    )
    checks["all_from_reserve"] = bool(from_reserve) and all(
        e in ledger_indexed.index for e in all_events
    )
    # Rainfall SHA fresh: distinct across the plan and unused by prior splits.
    plan_shas = [
        str(ledger_indexed.loc[e, "rainfall_sha256"])
        for e in all_events
        if e in ledger_indexed.index
    ]
    checks["rainfall_sha_distinct"] = len(plan_shas) == len(set(plan_shas))
    used_flags = ("used_train", "used_calibration", "used_locked_validation")
    unused = True
    for e in all_events:
        if e not in ledger_indexed.index:
            continue
        for flag in used_flags:
            if flag in ledger_indexed.columns and bool(ledger_indexed.loc[e, flag]):
                unused = False
    checks["events_never_used_in_prior_splits"] = unused
    checks["frozen_before_any_new_label"] = bool(
        plan.get("frozen_before_any_new_label")
    )
    checks["not_selected_by_old_locked"] = (
        plan.get("reads_old_locked_for_selection") is False
    )
    passed = all(checks.values())
    return {
        "stage": "AuditV4CompactEvaluationPlanV1",
        "status": "pass" if passed else "blocked",
        "checks": checks,
        "n_calibration_events": len(splits.get("v4.1_calibration", [])),
        "n_locked_events": len(splits.get("v4.1_locked", [])),
        "n_accrual_events": len(splits.get("locked_accrual_reserve", [])),
        "frozen_order_sha256": plan.get("frozen_order_sha256"),
    }


# ===========================================================================
# Section 14 -- calibrate the compact model on the NEW Calibration only
# ===========================================================================

def _temperature_fit(p: np.ndarray, y: np.ndarray) -> float:
    """Single scalar temperature by minimising NLL on a coarse grid."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    logit = np.log(p / (1 - p))
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 26):
        q = 1.0 / (1.0 + np.exp(-logit / t))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        nll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def calibrate_compact(
    model,
    data,
    *,
    cfg,
    dead_zones: dict[str, float],
    calibration_split: str = "v4.1_calibration",
) -> dict[str, Any]:
    """Calibrate compact intervals / probabilities on the NEW Calibration.

    Never updates model weights and never reads Locked.  A probability head
    whose per-class calibration support is below ``MIN_CLASS_SUPPORT`` is
    disabled (not faked); those decisions fall back to the continuous UCB.
    """
    ca = data.split_index(calibration_split)
    if ca.size == 0:
        raise ValueError(f"empty calibration split: {calibration_split}")
    pred = model.predict(data, ca)

    intervals: dict[str, Any] = {}
    coverage_rows: list[dict[str, Any]] = []
    for head, yhat in pred["continuous"].items():
        y = data.continuous[head][ca]
        direction = HEAD_DIRECTION.get(head, "underprediction")
        conf = one_sided_conformal(
            y, yhat, direction=direction, coverage=CONFORMAL_COVERAGE
        )
        intervals[head] = conf
        coverage_rows.append(
            {
                "head": head,
                "direction": direction,
                "coverage_target": CONFORMAL_COVERAGE,
                "empirical_coverage": conf["empirical_coverage"],
                "bound": conf["bound"],
                "n": conf["n"],
            }
        )

    temperatures: dict[str, float] = {}
    class_support: dict[str, dict[str, int]] = {}
    disabled_heads: list[str] = []
    for col in CLASSIFICATION_HEADS:
        if col not in pred["classification"] or col not in data.classification:
            continue
        y = data.classification[col][ca].astype(int)
        support = {"positive": int((y == 1).sum()), "negative": int((y == 0).sum())}
        class_support[col] = support
        insufficient = min(support.values()) < MIN_CLASS_SUPPORT
        if insufficient or len(np.unique(y)) < 2:
            disabled_heads.append(col)
            temperatures[col] = 1.0
            continue
        temperatures[col] = _temperature_fit(pred["classification"][col], y)

    unc = np.asarray(pred.get("uncertainty", np.zeros(ca.size)), dtype=float)
    abstain_threshold = float(
        np.quantile(unc, cfg.abstain_uncertainty_quantile)
    ) if unc.size else 0.0
    ood_threshold = float(np.quantile(unc, cfg.ood_quantile)) if unc.size else 0.0

    return {
        "stage": "CalibrateV4CompactV1",
        "split_used": calibration_split,
        "calibration_n": int(ca.size),
        "reads_locked": False,
        "updates_model_weights": False,
        "continuous_interval_calibration": {
            "method": "one_sided_conformal_q90",
            "direction_by_head": HEAD_DIRECTION,
            "intervals": intervals,
        },
        "empirical_coverage": coverage_rows,
        "probability_calibration": {
            "method": "temperature_scaling",
            "temperatures": temperatures,
            "class_support": class_support,
            "min_class_support": MIN_CLASS_SUPPORT,
        },
        "disabled_probability_heads": disabled_heads,
        "fallback_when_head_disabled": "continuous_ucb_and_safe_fallback",
        "abstain_uncertainty_threshold": abstain_threshold,
        "ood_threshold": ood_threshold,
    }


# ===========================================================================
# Section 16 -- one-shot Locked evaluation of the compact model
# ===========================================================================

def locked_baselines(
    data, *, locked_split: str, train_split: str = "train"
) -> dict[str, dict[str, float]]:
    """Zero and Train-mean baseline MAE per continuous head on the Locked split."""
    lk = data.split_index(locked_split)
    tr = data.split_index(train_split)
    out: dict[str, dict[str, float]] = {}
    for head in CONTINUOUS_HEADS:
        if head not in data.continuous:
            continue
        y = data.continuous[head][lk]
        zero_mae = float(np.mean(np.abs(y)))
        train_mean = float(np.mean(data.continuous[head][tr])) if tr.size else 0.0
        mean_mae = float(np.mean(np.abs(y - train_mean)))
        best = min(zero_mae, mean_mae)
        out[head] = {
            "zero_mae": zero_mae,
            "train_mean_mae": mean_mae,
            "train_mean": train_mean,
            "best_simple_mae": best,
        }
    return out


def _apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / max(t, 1e-6)))


def evaluate_compact_locked(
    model,
    data,
    *,
    cfg,
    dead_zones: dict[str, float],
    calibration: dict[str, Any],
    locked_split: str = "v4.1_locked",
    train_split: str = "train",
) -> dict[str, Any]:
    """Read-only one-shot Locked report for the compact model (never tuned)."""
    lk = data.split_index(locked_split)
    if lk.size == 0:
        raise ValueError(f"empty locked split: {locked_split}")
    pred = model.predict(data, lk)
    baselines = locked_baselines(data, locked_split=locked_split, train_split=train_split)
    temps = calibration.get("probability_calibration", {}).get("temperatures", {})
    disabled = set(calibration.get("disabled_probability_heads", []))

    continuous: dict[str, Any] = {}
    for head, yhat in pred["continuous"].items():
        y = data.continuous[head][lk]
        reg = regression_metrics(y, yhat, dead_zone=float(dead_zones.get(head, 0.0)))
        var = float(np.var(y))
        reg["r2"] = (
            float(1.0 - np.mean((np.asarray(yhat) - y) ** 2) / var)
            if var > 0
            else None
        )
        base = baselines.get(head, {})
        best_simple = base.get("best_simple_mae")
        improvement = (
            float((best_simple - reg["mae"]) / best_simple)
            if best_simple and best_simple > 0
            else None
        )
        reg["baseline"] = base
        reg["mae_improvement_vs_best_simple"] = improvement
        reg["beats_mean_baseline"] = (
            best_simple is not None and reg["mae"] < best_simple
        )
        reg["worst_event"] = worst_event(
            {
                str(e): float(np.mean(np.abs(yhat[data.event_id[lk] == e] - y[data.event_id[lk] == e])))
                for e in np.unique(data.event_id[lk])
            }
        )
        reg["uncertainty_error"] = uncertainty_error_correlation(
            np.asarray(pred.get("uncertainty", np.zeros(lk.size)))[: len(y)],
            np.abs(yhat - y),
        )
        continuous[head] = reg

    classification: dict[str, Any] = {}
    for col in CLASSIFICATION_HEADS:
        if col not in pred["classification"] or col not in data.classification:
            continue
        y = data.classification[col][lk].astype(int)
        p = pred["classification"][col]
        if col not in disabled:
            p = _apply_temperature(p, float(temps.get(col, 1.0)))
        cm = classification_metrics(y, p)
        cm["probability_head_disabled"] = col in disabled
        classification[col] = cm

    decision: dict[str, Any] = {}
    if "joint_noninferior" in pred["classification"]:
        score = pred["classification"]["joint_noninferior"]
        feasible = data.classification["joint_noninferior"][lk].astype(bool)
        regret = data.ranking.get(
            "regret_to_exact_best", np.full(len(data.split), np.nan)
        )[lk]
        pfv_safe_true = (
            data.classification["pfv_safe"][lk].astype(bool)
            if "pfv_safe" in data.classification
            else None
        )
        peak_ok_true = (
            data.classification["peak_noninferior"][lk].astype(bool)
            if "peak_noninferior" in data.classification
            else None
        )
        decision = decision_metrics(
            state_key=data.state_key[lk],
            score=score,
            feasible_true=feasible,
            regret=regret,
            pfv_unsafe_true=None if pfv_safe_true is None else ~pfv_safe_true,
            peak_unsafe_true=None if peak_ok_true is None else ~peak_ok_true,
        )
        decision.pop("per_state", None)

    return {
        "stage": "EvaluateV4CompactLockedV1",
        "model_version": "v4.1_compact",
        "split_used": locked_split,
        "n": int(lk.size),
        "used_for_tuning": False,
        "continuous": continuous,
        "classification": classification,
        "decision": decision,
        "baselines": baselines,
    }


# ===========================================================================
# Section 15 -- Predictive Generalization Gate verdict
# ===========================================================================

def evaluate_predictive_gate(
    locked_report: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Score the frozen Predictive Generalization Gate against the Locked report.

    Verdict is ``pass`` only when every continuous / classification / ranking /
    calibration condition holds; ``underpowered`` when positive class support or
    feasible states are insufficient (triggers pre-registered accrual, never a
    model/threshold change); otherwise ``scientific_fail``.
    """
    cont_c = contract.get("continuous", {})
    cls_c = contract.get("classification", {})
    rank_c = contract.get("ranking_and_decision", {})
    continuous = locked_report.get("continuous", {})
    classification = locked_report.get("classification", {})
    decision = locked_report.get("decision", {})

    checks: dict[str, Any] = {}
    reasons: list[str] = []

    # --- continuous heads ---
    r2_all_positive = True
    beat_mean_all = True
    n_improved = 0
    sign_ok_all = True
    for head in CONTINUOUS_HEADS:
        if head not in continuous:
            continue
        m = continuous[head]
        r2 = m.get("r2")
        # regression_metrics stores r2 under a key; guard both names.
        if r2 is None:
            r2 = m.get("r_squared")
        if r2 is None or r2 <= 0:
            r2_all_positive = False
        if not m.get("beats_mean_baseline"):
            beat_mean_all = False
        imp = m.get("mae_improvement_vs_best_simple")
        if imp is not None and imp >= float(
            cont_c.get("min_relative_mae_improvement", 0.10)
        ):
            n_improved += 1
        sign = m.get("sign_accuracy_outside_dead_zone")
        if sign is not None and sign < 0.5:
            sign_ok_all = False
    checks["all_continuous_r2_positive"] = r2_all_positive
    checks["all_continuous_beat_mean_baseline"] = beat_mean_all
    checks["at_least_two_heads_improve_10pct"] = n_improved >= int(
        cont_c.get("min_heads_improving", 2)
    )
    checks["sign_accuracy_beats_coin_flip"] = sign_ok_all

    # --- safety classification ---
    mcc_ok = True
    ap_ok = True
    ba_ok = True
    false_safe_ok = True
    underpowered = False
    for col in ("pfv_safe", "peak_noninferior"):
        if col not in classification:
            continue
        cm = classification[col]
        if cm.get("probability_head_disabled") or cm.get("mcc") is None:
            underpowered = True
            continue
        if cm.get("mcc", -1) <= 0:
            mcc_ok = False
        support = cm.get("class_support", {})
        prevalence = (
            support.get("positive", 0) / max(cm.get("n", 1), 1)
            if support
            else 0.0
        )
        if cm.get("average_precision") is not None and cm["average_precision"] <= prevalence:
            ap_ok = False
        if cm.get("balanced_accuracy") is not None and cm["balanced_accuracy"] < float(
            cls_c.get("min_balanced_accuracy", 0.60)
        ):
            ba_ok = False
        fsr = cm.get("false_safe_rate")
        if fsr is not None and fsr > float(cls_c.get("max_false_safe_rate", 0.20)):
            false_safe_ok = False
    checks["classification_mcc_positive"] = mcc_ok
    checks["average_precision_beats_prevalence"] = ap_ok
    checks["balanced_accuracy_ok"] = ba_ok
    checks["false_safe_within_limit"] = false_safe_ok

    # --- ranking / decision ---
    topk = decision.get("top_k_feasible_recall", {})
    top5 = topk.get("5")
    checks["top5_feasible_recall_ok"] = (
        top5 is not None and top5 >= float(rank_c.get("min_top5_feasible_recall", 0.80))
    )
    states_with_feasible = decision.get("states_with_feasible", 0)
    checks["min_feasible_states"] = states_with_feasible >= int(
        rank_c.get("min_feasible_states", 5)
    )
    if top5 is None or states_with_feasible < int(rank_c.get("min_feasible_states", 5)):
        underpowered = True

    passed = all(bool(v) for v in checks.values())
    if passed:
        verdict = "pass"
    elif underpowered and not _hard_fail(checks):
        verdict = "underpowered"
        reasons.append("insufficient positive class support or feasible states")
    else:
        verdict = "scientific_fail"

    return {
        "stage": "AuditV4PredictiveGeneralizationGateV1",
        "gate": contract.get("contract_id", "PROJECT6_V4_PREDICTIVE_GENERALIZATION_GATE_V1"),
        "status": verdict,
        "authorizes_closed_loop": verdict == "pass",
        "checks": checks,
        "reasons": reasons,
        "underpowered_policy": (
            "trigger pre-registered accrual; never change model, thresholds or "
            "delete original events"
        ),
    }


def _hard_fail(checks: dict[str, Any]) -> bool:
    """A hard scientific failure regardless of statistical power."""
    hard_keys = (
        "all_continuous_r2_positive",
        "all_continuous_beat_mean_baseline",
        "false_safe_within_limit",
        "classification_mcc_positive",
    )
    return any(not bool(checks.get(k, True)) for k in hard_keys)
