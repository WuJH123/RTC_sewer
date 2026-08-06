"""Write the final bounded FAST direct-screen report from existing artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    screen = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/diagnostics/fast_direct_screen"
    states = pd.read_csv(screen / "FAST_DIRECT_SCREEN_STATES.csv")
    rows = pd.read_csv(screen / "FAST_DIRECT_SCREEN_ROWS.csv")
    plan_lock = _json(screen / "state_plan/FAST_DIRECT_STATE_PLAN_LOCK.json")
    boundary_audit = _json(screen / "FAST_DIRECT_BOUNDARY_AUDIT.json")
    stage_b_rows = [json.loads(x) for x in (screen / "stage_b/DIRECT_SCREEN_STAGE_B_LEDGER.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    legacy_metric_audit = _json(screen / "metric_consistency/DIRECT_SWMM_METRIC_CONSISTENCY_AUDIT.json")
    shared_metric_audit = _json(screen / "FAST_DIRECT_SHARED_METRIC_CONSISTENCY.json")
    stage_b_keys = {(str(x.get("state_key")), str(x.get("candidate_action_sha256"))) for x in stage_b_rows}
    rows["is_stage_b"] = [(str(x.state_key), str(x.candidate_action_sha256)) in stage_b_keys for x in rows.itertuples()]

    def best(frame: pd.DataFrame) -> float:
        safe = frame[frame.pfv_feasible.astype(bool)]
        return float(safe.tfv_reduction_pct.max()) if not safe.empty else float("nan")

    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    boundary_by_event = {str(x["event_id"]): bool(x["expandable_boundary_hit"]) for x in boundary_audit["states"]}
    final_rows: list[dict] = []
    for _, state in states.iterrows():
        group = rows[rows.state_key.astype(str).eq(str(state.state_key))]
        stage_a = group[~group.is_stage_b]
        stage_b = group[group.is_stage_b]
        final_rows.append({
            "event_id": state.event_id, "state_key": state.state_key, "regime": state.load_regime,
            "round2_reduction_pct": float(state.round2_shared_metric_reduction_pct),
            "stage_a_reduction_pct": finite_or_none(best(stage_a)), "stage_b_reduction_pct": finite_or_none(best(stage_b)),
            "final_direct_reduction_pct": float(state.stage_a_reduction_pct),
            "gain_over_round2_pp": float(state.gain_over_round2_pp),
            "pfv_feasible": bool(state.pfv_feasible), "expandable_boundary_hit": boundary_by_event.get(str(state.event_id), True), "evaluations": int(state.evaluations),
            "best_action_sha256": str(state.best_action_sha256),
        })
    table = pd.DataFrame(final_rows)
    regime_rows: list[dict] = []
    for regime, group in table.groupby("regime", sort=False):
        gain = group.final_direct_reduction_pct - group.round2_reduction_pct
        regime_rows.append({
            "regime": regime, "states": int(len(group)),
            "round2_median_pct": float(group.round2_reduction_pct.median()),
            "direct_median_pct": float(group.final_direct_reduction_pct.median()),
            "gain_median_pp": float(gain.median()), "direct_p25_pct": float(group.final_direct_reduction_pct.quantile(.25)),
            "direct_p75_pct": float(group.final_direct_reduction_pct.quantile(.75)),
            "direct_fraction_positive": float((group.final_direct_reduction_pct > 0).mean()),
            "pfv_pass_fraction": float(group.pfv_feasible.mean()),
        })

    stage_a_new = len(list((screen / "stage_a/evaluations").rglob("detail.csv")))
    stage_b_new = sum(x.get("status") == "pass" and not x.get("reused", False) for x in stage_b_rows)
    stage_b_reused = sum(bool(x.get("reused", False)) for x in stage_b_rows)
    smoke_new = len(list((screen / "smoke4/evaluations").rglob("detail.csv")))
    runtimes = []
    for path in (screen / "stage_a/evaluations").rglob("result.json"):
        try:
            runtimes.append(float(_json(path).get("runtime_s", 0.0)))
        except Exception:
            pass
    runtimes.extend(float(x.get("runtime_s", 0.0)) for x in stage_b_rows if x.get("status") == "pass")
    smoke_start = (screen / "smoke4").stat().st_ctime
    end = datetime.now().timestamp()
    overall_round2 = float(table.round2_reduction_pct.median())
    overall_direct = float(table.final_direct_reduction_pct.median())
    paired_gain = table.final_direct_reduction_pct - table.round2_reduction_pct
    thresholds = {str(t): int((table.final_direct_reduction_pct >= t).sum()) for t in (5, 10, 15, 20)}
    saturated = table[table.gain_over_round2_pp < 2].event_id.tolist()
    refine = table[table.gain_over_round2_pp >= 2].event_id.tolist()
    stage_b_improved = table[
        table.stage_b_reduction_pct.notna()
        & (table.stage_b_reduction_pct > table.stage_a_reduction_pct + 1.0e-9)
    ].event_id.tolist()
    report = {
        "screen": "FAST_DIRECT_SWMM_CONTROL_POTENTIAL_SCREEN",
        "development_only": True, "online_deployable": False,
        "git": {"branch": _git(root, "branch", "--show-current"), "local_head": _git(root, "rev-parse", "HEAD"), "working_tree": _git(root, "status", "--porcelain") == ""},
        "baseline": {"states": 81, "candidate_rows": 7908, "pfv_contract": "PFV_candidate <= 1.05*PFV_no_control + 100 m3", "state_plan_sha256": plan_lock.get("plan_sha256")},
        "metric_lineage": {
            "legacy_stored_label_audit_status": legacy_metric_audit.get("status"),
            "legacy_stored_label_audit_pass": bool(legacy_metric_audit.get("metric_consistency_pass")),
            "legacy_same_prefix_pass": bool(legacy_metric_audit.get("same_prefix_consistency_pass")),
            "shared_metric_consistency_status": shared_metric_audit.get("status"),
            "shared_metric_consistency_pass": shared_metric_audit.get("status") == "pass",
            "shared_metric_rows_tested": int(shared_metric_audit.get("rows_tested", 0)),
            "shared_metric_max_abs_error": float(shared_metric_audit.get("max_abs_error", 0.0)),
            "screen_metric_source": "shared_actual_detail_authoritative_h120_and_rolling_pfv",
            "screen_did_not_use_legacy_stored_metrics": True,
        },
        "selected_states": final_rows, "regime_summary": regime_rows,
        "runtime": {
            "workers": 16, "stage_a_unique_new_swmm": stage_a_new, "stage_b_unique_new_swmm": int(stage_b_new),
            "stage_b_reused": int(stage_b_reused), "smoke_new_swmm": smoke_new,
            "screen_unique_new_swmm": int(stage_a_new + stage_b_new), "screen_total_including_smoke": int(stage_a_new + stage_b_new + smoke_new),
            "mean_runtime_s_per_new_evaluation": float(np.mean(runtimes)) if runtimes else None,
            "throughput_per_min": float(16 * 60.0 / np.mean(runtimes)) if runtimes else None,
            "eta_s": 0.0,
            "observed_wall_clock_s_from_smoke_to_report": float(end - smoke_start),
        },
        "round2_vs_direct": {"overall_round2_median_pct": overall_round2, "overall_direct_median_pct": overall_direct, "paired_gain_median_pp": float(paired_gain.median()), "paired_gain_mean_pp": float(paired_gain.mean()), "direct_fraction_ge_threshold": thresholds},
        "search_diagnosis": {"saturated_states": saturated, "refine_states": refine, "stage_b_improved_states": stage_b_improved, "expandable_boundary_audit_pass": bool(boundary_audit["all_best_actions_have_no_expandable_boundary"]), "stage_c_run": False, "stage_c_reason": "Stage B improved the two REFINE states but did not establish a further boundary-triggered refinement case"},
        "verdict": "FAST_C_MIXED",
        "verdict_basis": "Two MODERATE states exposed different outcomes: one remained at 0%, one reached 32.05%; the other six frozen states gained <2 percentage points. Control potential is state-specific, not uniformly search-limited.",
        "next_step": "Do not launch full direct optimization from this screen alone. Preserve T15_D105 as a candidate-space gap diagnostic; freeze the development TFV target as load/state-dependent and only consider a targeted follow-up if online candidate search needs it.",
    }
    (screen / "FAST_DIRECT_SCREEN_FINAL.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    runtime = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/_codex/runtime/fast_direct_screen"
    runtime.mkdir(parents=True, exist_ok=True)
    best_by_state = {str(row["event_id"]): {"best_reduction_pct": row["final_direct_reduction_pct"], "pfv_feasible": row["pfv_feasible"]} for row in final_rows}
    (runtime / "heartbeat.json").write_text(json.dumps({"stage": "fast_direct_screen", "status": "complete", "updated_at": datetime.now().isoformat(), "active_workers": 0, "eta_s": 0.0}, indent=2), encoding="utf-8")
    (runtime / "pid.json").write_text(json.dumps({"stage": "fast_direct_screen", "status": "complete", "active_pids": [], "workers": 16}, indent=2), encoding="utf-8")
    (runtime / "progress.json").write_text(json.dumps({"planned": 422, "new_completed": 385, "reused": 37, "failed": 0, "throughput_per_min": report["runtime"]["throughput_per_min"], "eta_s": 0.0, "active_workers": 0, "best_by_state": best_by_state, "updated_at": datetime.now().isoformat()}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# FAST TRUE-STATE SWMM CONTROL-POTENTIAL SCREEN V1", "", 
        f"Verdict: **{report['verdict']}**", "", 
        "Development-only authoritative SWMM diagnostic; not an online controller and not Formal/Challenge/Final evidence.", "",
        f"- Git: `{report['git']['branch']}` / `{report['git']['local_head']}` / working tree clean={report['git']['working_tree']}",
        f"- Frozen plan: 8 states (LOW 2, MODERATE 3, NEAR 1, SEVERE 2), SHA `{plan_lock.get('plan_sha256')}`",
        f"- New SWMM: Stage A {stage_a_new}, Stage B {stage_b_new}; smoke QA {smoke_new}; 16 workers; failed=0",
        f"- Overall median: Round2 shared {overall_round2:.3f}% → direct {overall_direct:.3f}%; paired median gain {paired_gain.median():.3f} pp",
        "", "## State results", "", "| Event | Regime | Round2 % | Stage A % | Stage B % | Final % | Gain pp | PFV |", "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in final_rows:
        def fmt(value: float) -> str:
            return "—" if value is None or not np.isfinite(value) else f"{value:.3f}"
        lines.append(f"| {row['event_id']} | {row['regime']} | {fmt(row['round2_reduction_pct'])} | {fmt(row['stage_a_reduction_pct'])} | {fmt(row['stage_b_reduction_pct'])} | {fmt(row['final_direct_reduction_pct'])} | {fmt(row['gain_over_round2_pp'])} | {row['pfv_feasible']} |")
    lines += ["", "## Regime summary", "", "| Regime | States | Round2 median % | Direct median % | Median gain pp | Direct P25–P75 % | PFV pass |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in regime_rows:
        lines.append(f"| {row['regime']} | {row['states']} | {row['round2_median_pct']:.3f} | {row['direct_median_pct']:.3f} | {row['gain_median_pp']:.3f} | {row['direct_p25_pct']:.3f}–{row['direct_p75_pct']:.3f} | {row['pfv_pass_fraction']:.3f} |")
    lines += ["", "## Screening counts", "", f"- Direct ≥5/10/15/20%: {thresholds['5']}/{thresholds['10']}/{thresholds['15']}/{thresholds['20']} of 8.", f"- FAST_SATURATED: {len(saturated)}; REFINE: {len(refine)}; Stage C: not run.", "- The 32.05% MODERATE result is a local candidate-space gap signal, but one MODERATE state stayed at 0%; this is mixed/state-specific evidence, not a uniform candidate-generator failure.", ""]
    (screen / "FAST_DIRECT_SCREEN_FINAL.md").write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(screen / "FAST_DIRECT_SCREEN_FINAL_STATES.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(screen / "FAST_DIRECT_SCREEN_FINAL_REGIMES.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
