from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.control.hierarchical_core26_residual10 import build_strict_preflight
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config, resolve_gat_model_path
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.io.safe_paths import path_budget_check, short_run_tag
from sewerrtc.evaluation.policy_sets import paper_baseline_policy_ids, paper_policy_ids
from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.simulation.pyswmm_runner import run_swmm_mpc_closed_loop, run_swmm_trajectory


def _configured_dir(cfg: dict, key: str, default: str) -> Path:
    raw = (cfg.get("outputs", {}) or {}).get(key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = cfg_path(cfg, "project_root") / path
    return path


def _detail_kpis(detail_path: str | Path, priority: list[str], control_step_sec: int, policy_id: str, event_id: str, duration_min: int) -> dict:
    detail_path = Path(detail_path)
    detail = pd.read_csv(detail_path)
    row = compute_kpis(detail, priority, int(control_step_sec))
    row.update(
        {
            "event_id": event_id,
            "policy_id": policy_id,
            "duration_min": int(duration_min),
            "detail_file": str(detail_path),
            "rows": int(len(detail)),
        }
    )
    return row


def _run_baseline_job(job: dict) -> dict:
    actuators = pd.read_csv(job["actuator_csv"])
    return run_swmm_trajectory(
        job["run_inp"],
        job["policy"],
        actuators,
        job["priority"],
        job["detail"],
        job["event_id"],
        int(job["duration_min"]),
        int(job["control_step_sec"]),
        int(job["seed"]),
        int(job["max_steps"]),
        simulation_duration_min=int(job["simulation_duration_min"]),
        recession_min=int(job["recession_min"]),
        pump_control_mode=str(job.get("pump_control_mode", "continuous")),
        variable_speed_pump_ids=job.get("variable_speed_pump_ids"),
    )


def _run_proposed_job(job: dict) -> dict:
    actuators = pd.read_csv(job["actuator_csv"])
    return run_swmm_mpc_closed_loop(
        job["proposed_inp"],
        actuators,
        job["priority"],
        job["detail"],
        job["history"],
        job["event_id"],
        int(job["duration_min"]),
        job["gat_path"],
        job["surrogate_path"],
        job["sensors"],
        job["node_order"],
        int(job["control_step_sec"]),
        job["device"],
        int(job["max_steps"]),
        pump_control_mode=str(job.get("pump_control_mode", "continuous")),
        variable_speed_pump_ids=job.get("variable_speed_pump_ids"),
        nominal_detail_csv=job.get("nominal_detail_csv"),
        no_control_detail_csv=job.get("no_control_detail_csv"),
        passive_detail_csv=job.get("passive_detail_csv"),
        internal_shadow_inp_path=job.get("internal_shadow_inp_path"),
        residual_value_path=job.get("residual_value_path"),
        residual_pfv_prob_min=float(job["residual_pfv_prob_min"]),
        residual_safe_prob_min=float(job["residual_safe_prob_min"]),
        residual_nonzero_prob_min=float(job["residual_nonzero_prob_min"]),
        residual_peak_prob_min=float(job["residual_peak_prob_min"]),
        max_candidate_delta=float(job["max_candidate_delta"]),
        topk_log_count=int(job["topk_log_count"]),
        max_candidate_count=int(job["max_candidate_count"]),
        candidate_hold_steps=job.get("candidate_hold_steps"),
        allowed_candidate_templates=job.get("allowed_candidate_templates"),
        blocked_candidate_templates=job.get("blocked_candidate_templates"),
        allowed_candidate_scopes_by_template=job.get("allowed_candidate_scopes_by_template"),
        priority_khop=int(job["priority_khop"]),
        empirical_guard_path=job.get("empirical_guard_path"),
        empirical_guard_unknown_allow=bool(job["empirical_guard_unknown_allow"]),
        boost_safe_prob_extra=float(job["boost_safe_prob_extra"]),
        boost_peak_prob_extra=float(job["boost_peak_prob_extra"]),
        protective_safe_prob_relief=float(job["protective_safe_prob_relief"]),
        release_peak_hold_max=int(job["release_peak_hold_max"]),
        low_risk_pfv_threshold=float(job["low_risk_pfv_threshold"]),
        high_risk_pfv_threshold=float(job["high_risk_pfv_threshold"]),
        release_recession_pfv_min=float(job["release_recession_pfv_min"]),
        release_recession_priority_depth_min=float(job["release_recession_priority_depth_min"]),
        strict_guard_return_period_max=int(job["strict_guard_return_period_max"]),
        strict_guard_patterns=str(job["strict_guard_patterns"]),
        strict_guard_prob_extra=float(job["strict_guard_prob_extra"]),
        horizon_smooth_weight=float(job["horizon_smooth_weight"]),
        horizon_violation_penalty=float(job["horizon_violation_penalty"]),
        proposed_controller=str(job["proposed_controller"]),
        horizon_steps=int(job["horizon_steps"]),
        priority_to_actuators_csv=job.get("priority_to_actuators_csv"),
        horizon_surrogate_model_path=job.get("horizon_surrogate_model_path"),
        rainfall_csv=job.get("rainfall_csv"),
        generic_default_policy_id=str(job["generic_default_policy_id"]),
        min_pfv_improvement_abs=float(job["min_pfv_improvement_abs"]),
        min_pfv_improvement_frac=float(job["min_pfv_improvement_frac"]),
        max_candidate_sequences=int(job["max_candidate_sequences"]),
        candidate_group_limit=int(job["candidate_group_limit"]),
        tfv_tolerance_abs=float(job["tfv_tolerance_abs"]),
        tfv_tolerance_frac=float(job["tfv_tolerance_frac"]),
        peak_tolerance_abs=float(job["peak_tolerance_abs"]),
        peak_tolerance_frac=float(job["peak_tolerance_frac"]),
        pfv_tolerance_abs=float(job["pfv_tolerance_abs"]),
        pfv_tolerance_frac=float(job["pfv_tolerance_frac"]),
        tfv_required_reduction_abs=float(job["tfv_required_reduction_abs"]),
        tfv_required_reduction_frac=float(job["tfv_required_reduction_frac"]),
        tfv_required_reduction_dry_multiplier=float(job["tfv_required_reduction_dry_multiplier"]),
        tfv_hard_constraint=bool(job["tfv_hard_constraint"]),
        dry_rain_threshold=float(job["dry_rain_threshold"]),
        peak_weight=float(job["peak_weight"]),
        pfv_weight=float(job["pfv_weight"]),
        adaptive_delta_enabled=bool(job["adaptive_delta_enabled"]),
        low_risk_max_candidate_delta=float(job["low_risk_max_candidate_delta"]),
        high_risk_max_candidate_delta=float(job["high_risk_max_candidate_delta"]),
        pfv_high_risk_horizon_threshold=float(job["pfv_high_risk_horizon_threshold"]),
        pfv_low_risk_horizon_threshold=float(job["pfv_low_risk_horizon_threshold"]),
        max_first_step_delta=float(job["max_first_step_delta"]),
        per_actuator_max_delta=dict(job.get("per_actuator_max_delta", {}) or {}),
        min_hold_steps_by_actuator=dict(job.get("min_hold_steps_by_actuator", {}) or {}),
        objective_mode=str(job["objective_mode"]),
        allowed_actuator_ids=job.get("allowed_actuator_ids"),
        blocked_actuator_ids=job.get("blocked_actuator_ids"),
        allowed_action_directions=job.get("allowed_action_directions"),
        phase_reliability_csv=job.get("phase_reliability_csv"),
        phase_reliability_allow_tfv_noninferior=bool(job.get("phase_reliability_allow_tfv_noninferior", False)),
        phase_reliability_require_pfv_improvement=bool(job.get("phase_reliability_require_pfv_improvement", False)),
        phase_reliability_pfv_tolerance_abs=float(job.get("phase_reliability_pfv_tolerance_abs", 100.0)),
        phase_reliability_pfv_tolerance_frac=float(job.get("phase_reliability_pfv_tolerance_frac", 0.005)),
        phase_reliability_tfv_tolerance_abs=float(job.get("phase_reliability_tfv_tolerance_abs", 0.0)),
        phase_reliability_tfv_tolerance_frac=float(job.get("phase_reliability_tfv_tolerance_frac", 0.0)),
        phase_reliability_peak_tolerance_abs=float(job.get("phase_reliability_peak_tolerance_abs", 0.75)),
        phase_reliability_peak_tolerance_frac=float(job.get("phase_reliability_peak_tolerance_frac", 0.005)),
        phase_reliability_pulse_steps=int(job.get("phase_reliability_pulse_steps", 2)),
        phase_reliability_max_overrides=int(job.get("phase_reliability_max_overrides", 0)),
        phase_reliability_candidate_group_limit=int(job.get("phase_reliability_candidate_group_limit", 4)),
        empirical_single_action_gate=bool(job.get("empirical_single_action_gate", False)),
        empirical_hold_steps=int(job.get("empirical_hold_steps", 2)),
        phase_reliability_fallback_to_surrogate=bool(job.get("phase_reliability_fallback_to_surrogate", False)),
        phase_reliability_evidence_time_tolerance_min=float(job.get("phase_reliability_evidence_time_tolerance_min", 2.5)),
        action_effect_model_path=job.get("action_effect_model_path"),
        raw_joint_model_path=job.get("raw_joint_model_path"),
        temporal_joint_config=job.get("temporal_joint_config"),
        v4_quantile=float(job.get("v4_quantile", 0.95)),
        v4_pfv_abs_margin_m3=float(job.get("v4_pfv_abs_margin_m3", 0.0)),
        v4_pfv_rel_margin=float(job.get("v4_pfv_rel_margin", 0.0)),
        v4_max_k=int(job.get("v4_max_k", 8)),
        v4_readback_tolerance=float(job.get("v4_readback_tolerance", 1.0e-4)),
        v4_action_deadband=float(job.get("v4_action_deadband", 0.02)),
        v4_adaptive_k_values=job.get("v4_adaptive_k_values", [0, 2, 4, 6, 8]),
        v4_changed_facility_penalty=float(job.get("v4_changed_facility_penalty", 1.0)),
        v4_variation_penalty=float(job.get("v4_variation_penalty", 1.0)),
        v4_reversal_penalty=float(job.get("v4_reversal_penalty", 5.0)),
        v4_minimum_material_benefit=float(job.get("v4_minimum_material_benefit", 0.0)),
        v4_minimum_benefit_cost_ratio=float(job.get("v4_minimum_benefit_cost_ratio", 1.5)),
    )


def _as_id_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    else:
        raw = value
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _needs_no_control_reference(reference_policy: str) -> bool:
    """Return whether a mode explicitly requests an offline SWMM reference.

    ``online_predicted_default`` is deliberately excluded: it invokes the
    same horizon surrogate for the candidate and passive default sequence and
    must not read the future true no-control trajectory during control.
    """
    policy = str(reference_policy or "").strip().lower()
    return policy in {"no_control", "internal_and_no_control", "precomputed_no_control_twin_horizon"}


def _path_from_config(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _reliable_allowed_actuator_ids(
    cfg: dict,
    controller_cfg: dict,
    *,
    rain_pattern: str = "",
) -> list[str]:
    explicit = _as_id_list(controller_cfg.get("allowed_actuator_ids"))
    repair_cfg = controller_cfg.get("repair_reliable_action_filter", {}) or {}
    if not bool(repair_cfg.get("enabled", False)):
        return explicit
    path_value = repair_cfg.get("pattern_phase_summary_csv", "") if rain_pattern else ""
    if not path_value:
        path_value = repair_cfg.get("summary_csv", "")
    if not path_value:
        return explicit
    path = _path_from_config(cfg_path(cfg, "project_root"), path_value)
    if not path.exists():
        print(f"[closed_loop] repair reliable action filter skipped: missing {path}")
        return explicit
    table = pd.read_csv(path)
    if table.empty or "actuator_id" not in table:
        print(f"[closed_loop] repair reliable action filter skipped: unusable {path}")
        return explicit
    work = table.copy()
    using_pattern_source = False
    if rain_pattern and "pattern" in work:
        pattern_work = work[work["pattern"].astype(str).eq(str(rain_pattern))].copy()
        if not pattern_work.empty:
            work = pattern_work
            using_pattern_source = True
        else:
            fallback_path_value = repair_cfg.get("summary_csv", "")
            if fallback_path_value:
                fallback_path = _path_from_config(cfg_path(cfg, "project_root"), fallback_path_value)
                if fallback_path.exists():
                    work = pd.read_csv(fallback_path)
                    path = fallback_path
    for col in ["repair_safe_frac", "pfv_noninferior_frac", "tfv_improved_frac", "peak_safe_frac", "events", "rows"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    min_pfv = float(repair_cfg.get("min_pfv_safe_prob", repair_cfg.get("min_pfv_noninferior_frac", 0.80)))
    min_tfv = float(repair_cfg.get("min_tfv_repair_prob", repair_cfg.get("min_tfv_improved_frac", 0.50)))
    min_peak = float(repair_cfg.get("min_peak_safe_prob", 0.80))
    min_events = int(
        repair_cfg.get("min_pattern_events", repair_cfg.get("min_events", 5))
        if using_pattern_source
        else repair_cfg.get("min_events", 5)
    )
    min_rows = int(
        repair_cfg.get("min_pattern_rows", repair_cfg.get("min_rows", 1))
        if using_pattern_source
        else repair_cfg.get("min_rows", 1)
    )
    mask = (
        (work.get("pfv_noninferior_frac", pd.Series(0.0, index=work.index)).fillna(0.0) >= min_pfv)
        & (work.get("tfv_improved_frac", pd.Series(0.0, index=work.index)).fillna(0.0) >= min_tfv)
        & (work.get("peak_safe_frac", pd.Series(0.0, index=work.index)).fillna(0.0) >= min_peak)
        & (work.get("events", pd.Series(0, index=work.index)).fillna(0).astype(int) >= min_events)
        & (work.get("rows", pd.Series(0, index=work.index)).fillna(0).astype(int) >= min_rows)
    )
    reliable = [str(x).strip() for x in work.loc[mask, "actuator_id"].tolist() if str(x).strip()]
    if not reliable and using_pattern_source:
        fallback_path_value = repair_cfg.get("summary_csv", "")
        if fallback_path_value:
            fallback_path = _path_from_config(cfg_path(cfg, "project_root"), fallback_path_value)
            if fallback_path.exists():
                print(
                    f"[closed_loop] repair reliable action filter pattern={rain_pattern} "
                    f"has no passing actuator; fallback to global summary"
                )
                fallback_cfg = dict(repair_cfg)
                fallback_cfg["pattern_phase_summary_csv"] = ""
                fallback_controller_cfg = dict(controller_cfg)
                fallback_controller_cfg["repair_reliable_action_filter"] = fallback_cfg
                return _reliable_allowed_actuator_ids(cfg, fallback_controller_cfg, rain_pattern="")
    manual = _as_id_list(repair_cfg.get("manual_include_actuator_ids"))
    blocked = set(_as_id_list(controller_cfg.get("blocked_actuator_ids")) + _as_id_list(repair_cfg.get("manual_exclude_actuator_ids")))
    if explicit:
        allowed = [aid for aid in explicit if aid in set(reliable) or aid in set(manual)]
    else:
        allowed = reliable + manual
    out: list[str] = []
    for aid in allowed:
        if aid and aid not in blocked and aid not in out and not aid.startswith("__"):
            out.append(aid)
    if not out:
        # An enabled empirical filter that finds no reliable actuator means
        # deny-all, not "no whitelist". The latter silently reopened all 109
        # facilities in the old implementation.
        out = ["__NO_RELIABLE_ACTUATOR__"]
    print(
        f"[closed_loop] repair reliable action filter allowed_actuators={len(out)} "
        f"pattern={rain_pattern or '*'} source={path}"
    )
    return out


def _reliable_allowed_action_directions(
    cfg: dict,
    controller_cfg: dict,
    allowed_actuator_ids: list[str],
    *,
    rain_pattern: str = "",
) -> dict[str, list[str]]:
    """Preserve the empirically safe sign of each actuator intervention."""
    if not allowed_actuator_ids:
        return {}
    repair_cfg = controller_cfg.get("repair_reliable_action_filter", {}) or {}
    path_value = repair_cfg.get("pattern_phase_summary_csv", "") if rain_pattern else ""
    if not path_value:
        path_value = repair_cfg.get("summary_csv", "")
    if not path_value:
        return {}
    path = _path_from_config(cfg_path(cfg, "project_root"), path_value)
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if rain_pattern and "pattern" in table:
        matched = table[table["pattern"].astype(str).eq(str(rain_pattern))]
        if not matched.empty:
            table = matched
    if "action_direction" not in table or "actuator_id" not in table:
        return {}
    allowed = set(allowed_actuator_ids)
    permissions: dict[str, list[str]] = {}
    for row in table.itertuples(index=False):
        aid = str(getattr(row, "actuator_id", "")).strip()
        direction = str(getattr(row, "action_direction", "")).strip().lower()
        if aid in allowed and direction in {"increase", "decrease"}:
            permissions.setdefault(aid, [])
            if direction not in permissions[aid]:
                permissions[aid].append(direction)
    return permissions


def _filter_pfv_positive_debug_events(
    cfg: dict,
    rain_table: pd.DataFrame,
    policies: list[str],
    min_pfv: float,
    out_root: Path,
) -> pd.DataFrame:
    summary_path = cfg_path(cfg, "outputs.data_bank_train") / "summary.csv"
    frames: list[pd.DataFrame] = []
    if summary_path.exists():
        frames.append(pd.read_csv(summary_path))
    else:
        print(f"[closed_loop] PFV-positive debug filter: training summary missing {summary_path}")
    if "internal_rules" in set(map(str, policies)):
        internal_candidates = [
            cfg_path(cfg, "outputs.closed_loop") / "internal_residual_counterfactuals" / "internal_baseline_results.csv",
            cfg_path(cfg, "outputs.closed_loop") / "debug" / "native_shield" / "baseline_results.csv",
            cfg_path(cfg, "outputs.closed_loop") / "debug" / "native_shield_safe010" / "baseline_results.csv",
        ]
        for p in internal_candidates:
            if p.exists() and p.stat().st_size > 0:
                try:
                    d = pd.read_csv(p)
                    if "policy_id" not in d:
                        d["policy_id"] = "internal_rules"
                    d = d[d["policy_id"].astype(str).eq("internal_rules")]
                    if not d.empty:
                        frames.append(d)
                        print(f"[closed_loop] PFV-positive debug filter loaded internal-risk source: {p}")
                        break
                except Exception as exc:
                    print(f"[closed_loop] PFV-positive debug filter ignored {p}: {exc}")
    if not frames:
        print("[closed_loop] PFV-positive debug filter skipped: no usable summary/internal-risk source.")
        return rain_table
    summary = pd.concat(frames, ignore_index=True, sort=False)
    if "event_id" not in summary or "PFV" not in summary or "policy_id" not in summary:
        print(f"[closed_loop] PFV-positive debug filter skipped: summary columns insufficient")
        return rain_table
    s = summary.copy()
    s["PFV"] = pd.to_numeric(s["PFV"], errors="coerce").fillna(0.0)
    if policies:
        s = s[s["policy_id"].astype(str).isin(policies)]
    s = s[s["PFV"] > float(min_pfv)]
    if s.empty:
        print(
            f"[closed_loop] PFV-positive debug filter found no event with PFV>{min_pfv} "
            f"under policies={policies}; keeping original event order."
        )
        return rain_table
    ranked = (
        s.groupby("event_id", as_index=False)
        .agg(debug_reference_PFV=("PFV", "max"), debug_reference_policy=("policy_id", lambda x: ",".join(sorted(set(map(str, x))))))
        .sort_values("debug_reference_PFV", ascending=False)
    )
    selected = rain_table.merge(ranked, on="event_id", how="inner").sort_values("debug_reference_PFV", ascending=False)
    if selected.empty:
        print("[closed_loop] PFV-positive debug filter matched no rainfall events; keeping original event order.")
        return rain_table
    selected.to_csv(out_root / "pfv_positive_debug_events.csv", index=False)
    print(
        f"[closed_loop] PFV-positive debug filter selected {len(selected)} events "
        f"from {len(rain_table)} using policies={policies}; "
        f"top={selected.iloc[0]['event_id']} PFV={float(selected.iloc[0]['debug_reference_PFV']):.3f}"
    )
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    ap.add_argument(
        "--run-tag",
        default="",
        help=(
            "Optional subdirectory under outputs.closed_loop/<mode>. Use this "
            "to keep native-shield and generic-clean experiments separate."
        ),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--event-ids", default="", help="Comma-separated rainfall event ids to run before max-events truncation.")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker count for CPU-only SWMM runs.")
    ap.add_argument("--proposed-workers", type=int, default=1, help="Event-level worker count for Proposed closed-loop runs.")
    ap.add_argument(
        "--baseline-policies",
        default="",
        help="Comma-separated baseline policy ids. If omitted, evaluation.paper_policy_set baselines are used.",
    )
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-proposed", action="store_true")
    ap.add_argument("--disable-pfv-positive-debug-filter", action="store_true")
    ap.add_argument("--pfv-positive-policies", default="")
    ap.add_argument("--pfv-positive-min", type=float, default=1.0)
    ap.add_argument(
        "--proposed-controller",
        choices=[
            "native_shield",
            "generic_gat_mpc",
            "proposed_pfvfirst_dualfallback_v3",
            "proposed_dual_reference_v4",
            "temporal_joint_36",
            "hierarchical_core26_residual10",
        ],
        default="",
        help="Proposed controller family. proposed_pfvfirst_dualfallback_v3 uses the frozen Project6 V3 action-effect ensemble.",
    )
    ap.add_argument("--proposed-base", choices=["clean", "native"], default="native")
    ap.add_argument("--raw-joint-model", default="", help="Verified same-state 36-action raw joint checkpoint.")
    ap.add_argument("--action-effect-model", default="", help="Frozen Project6 V3 action-effect ensemble .npz.")
    ap.add_argument(
        "--residual-value-path",
        default="",
        help="Optional residual action-value checkpoint for Proposed-NativeShield. "
        "If omitted in native mode, outputs.models/residual_action_value.pt is used when present.",
    )
    ap.add_argument("--residual-pfv-prob-min", type=float, default=None)
    ap.add_argument("--residual-safe-prob-min", type=float, default=None)
    ap.add_argument("--residual-nonzero-prob-min", type=float, default=None)
    ap.add_argument("--residual-peak-prob-min", type=float, default=None)
    ap.add_argument("--max-candidate-delta", type=float, default=None)
    ap.add_argument("--topk-log-count", type=int, default=8)
    ap.add_argument("--max-candidate-count", type=int, default=96)
    ap.add_argument("--priority-khop", type=int, default=3)
    ap.add_argument(
        "--empirical-guard-path",
        default="",
        help="Optional action-template empirical guard table. If omitted in native mode, "
        "outputs.diagnostics/action_template_outcomes/action_template_empirical_guard_table.csv is used when present.",
    )
    ap.add_argument("--disable-empirical-guard", action="store_true")
    ap.add_argument("--empirical-guard-block-unknown", action="store_true", default=None)
    ap.add_argument("--boost-safe-prob-extra", type=float, default=0.12)
    ap.add_argument("--boost-peak-prob-extra", type=float, default=0.10)
    ap.add_argument("--protective-safe-prob-relief", type=float, default=0.05)
    ap.add_argument("--release-peak-hold-max", type=int, default=1)
    ap.add_argument(
        "--low-risk-pfv-threshold",
        type=float,
        default=0.0,
        help="PFV threshold below which NativeShield should not intervene. "
        "If <=0, configs/wuhan.yaml risk_stratification.low_pfv_threshold is used.",
    )
    ap.add_argument(
        "--high-risk-pfv-threshold",
        type=float,
        default=0.0,
        help="PFV threshold defining high-risk events. If <=0, configs/wuhan.yaml risk_stratification.high_pfv_threshold is used.",
    )
    ap.add_argument("--release-recession-pfv-min", type=float, default=500.0)
    ap.add_argument("--release-recession-priority-depth-min", type=float, default=1.0)
    ap.add_argument("--strict-guard-return-period-max", type=int, default=15)
    ap.add_argument("--strict-guard-patterns", default="chicago_late,block,double_peak")
    ap.add_argument("--strict-guard-prob-extra", type=float, default=0.10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    controller_cfg = cfg.get("controller", {}) or {}
    if not args.proposed_controller:
        args.proposed_controller = str(controller_cfg.get("mode", "native_shield") or "native_shield")
    is_v3_controller = args.proposed_controller == "proposed_pfvfirst_dualfallback_v3"
    is_v4_controller = args.proposed_controller == "proposed_dual_reference_v4"
    is_action_effect_controller = is_v3_controller or is_v4_controller
    is_generic_like_controller = args.proposed_controller in {"generic_gat_mpc", "proposed_pfvfirst_dualfallback_v3", "proposed_dual_reference_v4"}
    # V4 observes native targets causally in its own live SWMM branch. It
    # must not pre-run and replay an Internal future trajectory online.
    use_native_nominal = args.proposed_controller in {"native_shield", "proposed_pfvfirst_dualfallback_v3"} and args.proposed_base == "native"
    # V4 must retain the original SWMM rules in its live branch so that an
    # Internal fallback is physically executable. This does not imply an
    # offline Internal future trajectory is generated or passed online.
    proposed_uses_native_rules = bool(use_native_nominal or (is_v4_controller and args.proposed_base == "native"))
    reference_policy = str(controller_cfg.get("reference_policy_for_constraints", "internal_and_no_control") or "internal_and_no_control")
    needs_internal_reference = (
        is_generic_like_controller
        and "internal" in reference_policy
        and not args.skip_proposed
    )
    needs_no_control_reference = (
        is_generic_like_controller
        and _needs_no_control_reference(reference_policy)
        and not args.skip_proposed
        and not is_v4_controller
    )
    # V4 baseline branches are still generated for post-event paired
    # evaluation, but no future baseline detail is passed into the controller.
    needs_passive_reference = False
    intervention_cfg = cfg.get("intervention_policy", {}) or {}
    shield_cfg = intervention_cfg.get("project5_safety_shield", {}) or {}
    if not args.baseline_policies.strip():
        args.baseline_policies = ",".join(paper_baseline_policy_ids(cfg))
    if args.residual_pfv_prob_min is None:
        args.residual_pfv_prob_min = float(intervention_cfg.get("residual_pfv_prob_min", 0.60))
    if args.residual_safe_prob_min is None:
        args.residual_safe_prob_min = float(intervention_cfg.get("residual_safe_prob_min", 0.70))
    if args.residual_nonzero_prob_min is None:
        args.residual_nonzero_prob_min = float(intervention_cfg.get("residual_nonzero_prob_min", 0.45))
    if args.residual_peak_prob_min is None:
        args.residual_peak_prob_min = float(intervention_cfg.get("residual_peak_prob_min", 0.60))
    if args.empirical_guard_block_unknown is None:
        args.empirical_guard_block_unknown = bool(intervention_cfg.get("empirical_guard_block_unknown", True))
    if float(args.low_risk_pfv_threshold) <= 0.0:
        args.low_risk_pfv_threshold = float(
            (cfg.get("risk_stratification", {}) or {}).get("low_pfv_threshold", 1000.0)
        )
    if float(args.high_risk_pfv_threshold) <= 0.0:
        args.high_risk_pfv_threshold = float(
            (cfg.get("risk_stratification", {}) or {}).get("high_pfv_threshold", 20000.0)
        )
    effective_max_candidate_delta = float(
        args.max_candidate_delta
        if args.max_candidate_delta is not None
        else controller_cfg.get("max_candidate_delta", 0.08)
    )
    repair_allowed_actuators_by_event: dict[str, list[str]] = {}
    rain_table = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    if args.event_ids.strip():
        selected_event_ids = [x.strip() for x in args.event_ids.split(",") if x.strip()]
        if selected_event_ids:
            before_events = len(rain_table)
            rain_table = rain_table[rain_table["event_id"].astype(str).isin(selected_event_ids)].copy()
            if rain_table.empty:
                raise ValueError(f"No rainfall events matched --event-ids={selected_event_ids}")
            order = {event_id: i for i, event_id in enumerate(selected_event_ids)}
            rain_table["_event_order"] = rain_table["event_id"].astype(str).map(order).fillna(999).astype(int)
            rain_table = rain_table.sort_values("_event_order").drop(columns=["_event_order"])
            print(f"[closed_loop] selected {len(rain_table)} events from {before_events} using --event-ids")
    out_root = cfg_path(cfg, "outputs.closed_loop") / args.mode
    if args.run_tag:
        original_run_tag = str(args.run_tag)
        args.run_tag = short_run_tag(original_run_tag, max_length=24)
        if args.run_tag != original_run_tag:
            print(f"[closed_loop] shortened run tag for Windows path safety: {original_run_tag} -> {args.run_tag}")
        out_root = out_root / args.run_tag
    path_audit = path_budget_check(out_root, budget=235)
    if not bool(path_audit["within_budget"]):
        raise OSError(f"closed-loop output root exceeds Windows path budget: {path_audit}")
    out_root = ensure_dir(out_root)
    if args.mode == "debug" and not args.disable_pfv_positive_debug_filter:
        default_policy_text = "internal_rules" if use_native_nominal else "auto_rbc,no_control,all_open"
        policy_text = args.pfv_positive_policies.strip() or default_policy_text
        pfv_policies = [p.strip() for p in policy_text.split(",") if p.strip()]
        rain_table = _filter_pfv_positive_debug_events(
            cfg,
            rain_table,
            pfv_policies,
            float(args.pfv_positive_min),
            out_root,
        )
    if args.max_events:
        rain_table = rain_table.head(args.max_events)
        rain_table.to_csv(out_root / "selected_events.csv", index=False)
    inp_dir = ensure_dir(out_root / "event_inp")
    base_dir = ensure_dir(out_root / "baselines")
    prop_dir = ensure_dir(out_root / "proposed")
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    actuator_scope = str(controller_cfg.get("actuator_scope", "existing_rtc"))
    actuators = select_actuators_for_scope(actuators, actuator_scope)
    is_temporal_joint_controller = args.proposed_controller in {"temporal_joint_36", "hierarchical_core26_residual10"}
    if is_temporal_joint_controller:
        from sewerrtc.control.actuator_scope import enrich_temporal_joint_actuator_semantics

        retrofit_manifest_value = ((cfg.get("network", {}) or {}).get("retrofit_asset_manifest", ""))
        if not retrofit_manifest_value:
            raise ValueError("temporal_joint_36 requires network.retrofit_asset_manifest")
        retrofit_manifest_path = cfg_path(cfg, "network.retrofit_asset_manifest")
        actuators = enrich_temporal_joint_actuator_semantics(
            actuators,
            pd.read_csv(retrofit_manifest_path),
        )
    if actuators.empty:
        raise ValueError(f"No actuators available for controller.actuator_scope={actuator_scope}")
    scoped_actuator_path = out_root / "control_actuator_table.csv"
    actuators.to_csv(scoped_actuator_path, index=False)
    node_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    node_order = node_table["node_id"].astype(str).tolist()
    sensors = pd.read_csv(cfg_path(cfg, "outputs.design") / "sensor_nodes.csv")["node_id"].astype(str).tolist()
    priority = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    model_dir = cfg_path(cfg, "outputs.models")
    gat_path = resolve_gat_model_path(cfg)
    surrogate_best_path = model_dir / "graph_surrogate_best.pt"
    surrogate_fallback_path = model_dir / "graph_surrogate.pt"
    surrogate_path = surrogate_best_path if surrogate_best_path.exists() else surrogate_fallback_path
    horizon_temporal_path = model_dir / "horizon_temporal_gnn.pt"
    horizon_ridge_path = model_dir / "horizon_ridge_surrogate.npz"
    horizon_surrogate_path = horizon_temporal_path if horizon_temporal_path.exists() else horizon_ridge_path
    temporal_joint_cfg = dict(controller_cfg.get("temporal_joint", {}) or {})
    action_effect_model_raw = str(args.action_effect_model or controller_cfg.get("action_effect_model_path", "") or "")
    action_effect_model_path = Path(action_effect_model_raw) if action_effect_model_raw else Path("")
    if action_effect_model_raw and not action_effect_model_path.is_absolute():
        action_effect_model_path = cfg_path(cfg, "project_root") / action_effect_model_path
    if is_action_effect_controller and (not action_effect_model_raw or not action_effect_model_path.exists() or not action_effect_model_path.is_file()):
        version = "V4 dual-reference" if is_v4_controller else "V3"
        raise FileNotFoundError(
            f"Missing Project6 {version} action-effect model at {action_effect_model_path}. "
            "Build the aligned dataset and train the matching ensemble before closed-loop evaluation."
        )
    raw_joint_model_path = Path(args.raw_joint_model) if args.raw_joint_model else Path(
        temporal_joint_cfg.get("model_path", model_dir / "raw_joint_36_same_state.pt")
    )
    if is_temporal_joint_controller and not raw_joint_model_path.exists():
        raise FileNotFoundError(
            f"Missing verified 36-action raw joint model at {raw_joint_model_path}. "
            "Build same-state paired data and train the effect model before closed-loop smoke."
        )
    hierarchical_preflight: dict | None = None
    if args.proposed_controller == "hierarchical_core26_residual10":
        hierarchical_preflight = build_strict_preflight(
            cfg=cfg,
            actuators=actuators,
            project_root=cfg_path(cfg, "project_root"),
        )
        (out_root / "hierarchical_preflight_manifest.json").write_text(
            json.dumps(hierarchical_preflight, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not bool(hierarchical_preflight.get("passed", False)):
            raise RuntimeError(
                "hierarchical_core26_residual10 preflight failed: "
                + ",".join(hierarchical_preflight.get("failed_checks", []))
            )
    residual_value_path = Path(args.residual_value_path) if args.residual_value_path else None
    if residual_value_path is None and use_native_nominal:
        candidate_residual = model_dir / "residual_action_value.pt"
        if candidate_residual.exists():
            residual_value_path = candidate_residual
    empirical_guard_path = Path(args.empirical_guard_path) if args.empirical_guard_path else None
    if empirical_guard_path is None and use_native_nominal and not args.disable_empirical_guard:
        candidate_guards = []
        if args.run_tag:
            candidate_guards.append(
                cfg_path(cfg, "outputs.diagnostics")
                / args.mode
                / args.run_tag
                / "action_template_outcomes"
                / "action_template_empirical_guard_table.csv"
            )
        candidate_guards.append(
            cfg_path(cfg, "outputs.diagnostics") / "action_template_outcomes" / "action_template_empirical_guard_table.csv"
        )
        for candidate_guard in candidate_guards:
            if candidate_guard.exists():
                empirical_guard_path = candidate_guard
                break
    if args.disable_empirical_guard:
        empirical_guard_path = None
    if not gat_path.exists():
        raise FileNotFoundError(f"Missing GAT model at {gat_path}. Run scripts/05_train_gat.py first.")
    if args.proposed_controller == "native_shield" and not surrogate_path.exists():
        raise FileNotFoundError(
            f"Missing surrogate model. Expected best checkpoint at {surrogate_best_path} "
            f"or fallback at {surrogate_fallback_path}. Run scripts/06_train_surrogate.py first."
        )
    baseline_rows, proposed_rows = [], []
    baseline_policies = [p.strip() for p in args.baseline_policies.split(",") if p.strip()]
    baseline_jobs = []
    proposed_jobs = []
    handled_baselines: set[tuple[str, str]] = set()
    precomputed_internal: set[str] = set()
    precomputed_no_control: set[str] = set()
    precomputed_passive: set[str] = set()

    # Proposed-NativeShield needs the internal-rule detail trajectory as the
    # nominal action source. These trajectories are pure SWMM CPU jobs and are
    # independent across rainfall events, so precompute them in parallel before
    # entering the proposed controller loop. This removes the old per-event
    # serial bottleneck while preserving the exact same closed-loop logic.
    if (use_native_nominal or needs_internal_reference) and not args.skip_proposed:
        internal_jobs = []
        for _, ev in rain_table.iterrows():
            event_id = str(ev["event_id"])
            event_inp_clean = inp_dir / f"{event_id}__no_controls.inp"
            event_inp_native = inp_dir / f"{event_id}__with_controls.inp"
            if not event_inp_clean.exists():
                mutate_inp_for_event(
                    cfg_path(cfg, "network.inp"),
                    ev["rainfall_csv"],
                    event_inp_clean,
                    int(ev["simulation_duration_min"]),
                    strip_controls=True,
                )
            if not event_inp_native.exists():
                mutate_inp_for_event(
                    cfg_path(cfg, "network.inp"),
                    ev["rainfall_csv"],
                    event_inp_native,
                    int(ev["simulation_duration_min"]),
                    strip_controls=False,
                )
            internal_detail = base_dir / "internal_rules" / f"{event_id}__internal_rules_detail.csv"
            ensure_dir(internal_detail.parent)
            if internal_detail.exists():
                precomputed_internal.add(event_id)
                continue
            internal_jobs.append(
                {
                    "run_inp": str(event_inp_native),
                    "policy": "internal_rules",
                    "actuator_csv": str(scoped_actuator_path),
                    "priority": priority,
                    "detail": str(internal_detail),
                    "event_id": event_id,
                    "duration_min": int(ev["duration_min"]),
                    "simulation_duration_min": int(ev["simulation_duration_min"]),
                    "recession_min": int(cfg["experiment"]["recession_min"]),
                    "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
                    "seed": int(cfg["experiment"]["random_seed"]),
                    "max_steps": int(args.max_steps),
                    "pump_control_mode": str(controller_cfg.get("pump_control_mode", "continuous")),
                    "variable_speed_pump_ids": list(controller_cfg.get("variable_speed_pump_ids", [])),
                }
            )
        if internal_jobs:
            workers = max(1, int(args.workers))
            job_role = "internal nominal" if use_native_nominal else "internal safety-reference"
            print(f"[closed_loop] precomputing {len(internal_jobs)} {job_role} SWMM jobs with workers={workers}")
            if workers == 1:
                for i, job in enumerate(internal_jobs, 1):
                    print(f"[closed_loop] {job_role} {i}/{len(internal_jobs)} {job['event_id']}")
                    _run_baseline_job(job)
                    precomputed_internal.add(str(job["event_id"]))
            else:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_run_baseline_job, job) for job in internal_jobs]
                    for i, fut in enumerate(as_completed(futs), 1):
                        row = fut.result()
                        event_id = str(row.get("event_id"))
                        precomputed_internal.add(event_id)
                        print(f"[closed_loop] done {job_role} {i}/{len(internal_jobs)} {event_id}")

    if needs_no_control_reference and not args.skip_proposed:
        no_control_jobs = []
        for _, ev in rain_table.iterrows():
            event_id = str(ev["event_id"])
            event_inp_clean = inp_dir / f"{event_id}__no_controls.inp"
            if not event_inp_clean.exists():
                mutate_inp_for_event(
                    cfg_path(cfg, "network.inp"),
                    ev["rainfall_csv"],
                    event_inp_clean,
                    int(ev["simulation_duration_min"]),
                    strip_controls=True,
                )
            no_control_detail = base_dir / "no_control" / f"{event_id}__no_control_detail.csv"
            ensure_dir(no_control_detail.parent)
            if no_control_detail.exists():
                precomputed_no_control.add(event_id)
                continue
            no_control_jobs.append(
                {
                    "run_inp": str(event_inp_clean),
                    "policy": "no_control",
                    "actuator_csv": str(scoped_actuator_path),
                    "priority": priority,
                    "detail": str(no_control_detail),
                    "event_id": event_id,
                    "duration_min": int(ev["duration_min"]),
                    "simulation_duration_min": int(ev["simulation_duration_min"]),
                    "recession_min": int(cfg["experiment"]["recession_min"]),
                    "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
                    "seed": int(cfg["experiment"]["random_seed"]),
                    "max_steps": int(args.max_steps),
                    "pump_control_mode": str(controller_cfg.get("pump_control_mode", "continuous")),
                    "variable_speed_pump_ids": list(controller_cfg.get("variable_speed_pump_ids", [])),
                }
            )
        if no_control_jobs:
            workers = max(1, int(args.workers))
            print(f"[closed_loop] precomputing {len(no_control_jobs)} no_control safety-reference SWMM jobs with workers={workers}")
            if workers == 1:
                for i, job in enumerate(no_control_jobs, 1):
                    print(f"[closed_loop] no_control safety-reference {i}/{len(no_control_jobs)} {job['event_id']}")
                    _run_baseline_job(job)
                    precomputed_no_control.add(str(job["event_id"]))
            else:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_run_baseline_job, job) for job in no_control_jobs]
                    for i, fut in enumerate(as_completed(futs), 1):
                        row = fut.result()
                        event_id = str(row.get("event_id"))
                        precomputed_no_control.add(event_id)
                        print(f"[closed_loop] done no_control safety-reference {i}/{len(no_control_jobs)} {event_id}")

    if needs_passive_reference:
        passive_jobs = []
        for _, ev in rain_table.iterrows():
            event_id = str(ev["event_id"])
            event_inp_clean = inp_dir / f"{event_id}__no_controls.inp"
            if not event_inp_clean.exists():
                mutate_inp_for_event(
                    cfg_path(cfg, "network.inp"), ev["rainfall_csv"], event_inp_clean,
                    int(ev["simulation_duration_min"]), strip_controls=True,
                )
            passive_detail = base_dir / "executable_passive" / f"{event_id}__executable_passive_detail.csv"
            ensure_dir(passive_detail.parent)
            if passive_detail.exists():
                precomputed_passive.add(event_id)
                continue
            passive_jobs.append({
                "run_inp": str(event_inp_clean), "policy": "executable_passive",
                "actuator_csv": str(scoped_actuator_path), "priority": priority,
                "detail": str(passive_detail), "event_id": event_id,
                "duration_min": int(ev["duration_min"]),
                "simulation_duration_min": int(ev["simulation_duration_min"]),
                "recession_min": int(cfg["experiment"]["recession_min"]),
                "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
                "seed": int(cfg["experiment"]["random_seed"]),
                "max_steps": int(args.max_steps),
                "pump_control_mode": str(controller_cfg.get("pump_control_mode", "continuous")),
                "variable_speed_pump_ids": list(controller_cfg.get("variable_speed_pump_ids", [])),
            })
        if passive_jobs:
            workers = max(1, int(args.workers))
            print(f"[closed_loop] precomputing {len(passive_jobs)} passive PFV-reference jobs with workers={workers}")
            if workers == 1:
                completed = [_run_baseline_job(job) for job in passive_jobs]
            else:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    completed = [future.result() for future in as_completed([ex.submit(_run_baseline_job, job) for job in passive_jobs])]
            precomputed_passive.update(str(row.get("event_id")) for row in completed)

    for _, ev in rain_table.iterrows():
        event_id = str(ev["event_id"])
        event_inp_clean = inp_dir / f"{event_id}__no_controls.inp"
        event_inp_native = inp_dir / f"{event_id}__with_controls.inp"
        if not event_inp_clean.exists():
            mutate_inp_for_event(
                cfg_path(cfg, "network.inp"),
                ev["rainfall_csv"],
                event_inp_clean,
                int(ev["simulation_duration_min"]),
                strip_controls=True,
            )
        if not event_inp_native.exists():
            mutate_inp_for_event(
                cfg_path(cfg, "network.inp"),
                ev["rainfall_csv"],
                event_inp_native,
                int(ev["simulation_duration_min"]),
                strip_controls=False,
            )
        internal_detail_for_nominal = base_dir / "internal_rules" / f"{event_id}__internal_rules_detail.csv"
        no_control_detail_for_reference = base_dir / "no_control" / f"{event_id}__no_control_detail.csv"
        passive_detail_for_reference = base_dir / "executable_passive" / f"{event_id}__executable_passive_detail.csv"
        if (use_native_nominal or needs_internal_reference) and not args.skip_proposed:
            ensure_dir(internal_detail_for_nominal.parent)
            if internal_detail_for_nominal.exists() and (args.skip_existing or event_id in precomputed_internal):
                internal_row = _detail_kpis(
                    internal_detail_for_nominal,
                    priority,
                    int(cfg["experiment"]["control_step_sec"]),
                    "internal_rules",
                    event_id,
                    int(ev["duration_min"]),
                )
                print(f"[closed_loop] reuse internal reference {event_id}")
            else:
                print(f"[closed_loop] running internal reference {event_id}")
                internal_row = run_swmm_trajectory(
                    event_inp_native,
                    "internal_rules",
                    actuators,
                    priority,
                    internal_detail_for_nominal,
                    event_id,
                    int(ev["duration_min"]),
                    int(cfg["experiment"]["control_step_sec"]),
                    int(cfg["experiment"]["random_seed"]),
                    args.max_steps,
                    simulation_duration_min=int(ev["simulation_duration_min"]),
                    recession_min=int(cfg["experiment"]["recession_min"]),
                )
                print(f"[closed_loop] done internal reference {event_id}")
            if "internal_rules" in baseline_policies:
                baseline_rows.append(internal_row)
                handled_baselines.add((event_id, "internal_rules"))
        if needs_no_control_reference and not args.skip_proposed:
            ensure_dir(no_control_detail_for_reference.parent)
            if no_control_detail_for_reference.exists() and (args.skip_existing or event_id in precomputed_no_control):
                no_control_row = _detail_kpis(
                    no_control_detail_for_reference,
                    priority,
                    int(cfg["experiment"]["control_step_sec"]),
                    "no_control",
                    event_id,
                    int(ev["duration_min"]),
                )
                print(f"[closed_loop] reuse no_control reference {event_id}")
            else:
                print(f"[closed_loop] running no_control reference {event_id}")
                no_control_row = run_swmm_trajectory(
                    event_inp_clean,
                    "no_control",
                    actuators,
                    priority,
                    no_control_detail_for_reference,
                    event_id,
                    int(ev["duration_min"]),
                    int(cfg["experiment"]["control_step_sec"]),
                    int(cfg["experiment"]["random_seed"]),
                    args.max_steps,
                    simulation_duration_min=int(ev["simulation_duration_min"]),
                    recession_min=int(cfg["experiment"]["recession_min"]),
                )
                print(f"[closed_loop] done no_control reference {event_id}")
            if "no_control" in baseline_policies:
                baseline_rows.append(no_control_row)
                handled_baselines.add((event_id, "no_control"))
        if not args.skip_baselines:
            for policy in baseline_policies:
                if (event_id, policy) in handled_baselines:
                    continue
                policy_dir = ensure_dir(base_dir / policy)
                b_detail = policy_dir / f"{event_id}__{policy}_detail.csv"
                b_recovery = base_dir / "recovery" / event_id / f"{event_id}__{policy}__recovery.json"
                if args.skip_existing and b_detail.exists() and b_recovery.exists():
                    baseline_rows.append(
                        _detail_kpis(
                            b_detail,
                            priority,
                            int(cfg["experiment"]["control_step_sec"]),
                            policy,
                            event_id,
                            int(ev["duration_min"]),
                        )
                    )
                    print(f"[closed_loop] reuse baseline {event_id} {policy}")
                    continue
                run_inp = event_inp_native if policy == "internal_rules" else event_inp_clean
                baseline_jobs.append(
                    {
                        "run_inp": str(run_inp),
                        "policy": policy,
                        "actuator_csv": str(scoped_actuator_path),
                        "priority": priority,
                        "detail": str(b_detail),
                        "event_id": event_id,
                        "duration_min": int(ev["duration_min"]),
                        "simulation_duration_min": int(ev["simulation_duration_min"]),
                        "recession_min": int(cfg["experiment"]["recession_min"]),
                        "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
                        "seed": int(cfg["experiment"]["random_seed"]),
                        "max_steps": int(args.max_steps),
                    }
                )
        p_detail = prop_dir / f"{event_id}__proposed_detail.csv"
        p_hist = prop_dir / f"{event_id}__controller_history.csv"
        if args.skip_proposed:
            continue
        proposed_policy_id = (
            "retrofit_hierarchical36" if args.proposed_controller == "hierarchical_core26_residual10"
            else "proposed_temporal_joint_36" if args.proposed_controller == "temporal_joint_36"
            else "proposed_gat_mpc" if args.proposed_controller == "generic_gat_mpc"
            else "proposed_dual_reference_v4" if args.proposed_controller == "proposed_dual_reference_v4"
            else "proposed_pfvfirst_dualfallback_v3"
        )
        if args.skip_existing and p_detail.exists() and p_hist.exists():
            proposed_rows.append(
                _detail_kpis(
                    p_detail,
                    priority,
                    int(cfg["experiment"]["control_step_sec"]),
                    proposed_policy_id,
                    event_id,
                    int(ev["duration_min"]),
                )
            )
            print(f"[closed_loop] reuse proposed {event_id}")
        else:
            print(f"[closed_loop] queue proposed {event_id}")
            proposed_inp = event_inp_clean if is_v4_controller else (event_inp_native if proposed_uses_native_rules else event_inp_clean)
            nominal_detail = internal_detail_for_nominal if (use_native_nominal or needs_internal_reference) else None
            priority_to_actuators_csv = _configured_dir(cfg, "network", "outputs/network") / "priority_to_actuator_candidates.csv"
            event_allowed_actuator_ids = _reliable_allowed_actuator_ids(
                cfg,
                controller_cfg,
                rain_pattern=str(ev.get("pattern", "")),
            )
            event_allowed_action_directions = _reliable_allowed_action_directions(
                cfg,
                controller_cfg,
                event_allowed_actuator_ids,
                rain_pattern=str(ev.get("pattern", "")),
            )
            repair_allowed_actuators_by_event[event_id] = event_allowed_actuator_ids
            proposed_jobs.append(
                {
                    "proposed_inp": str(proposed_inp),
                    "internal_shadow_inp_path": str(event_inp_native) if is_v4_controller else "",
                    "actuator_csv": str(scoped_actuator_path),
                    "priority": priority,
                    "detail": str(p_detail),
                    "history": str(p_hist),
                    "event_id": event_id,
                    "duration_min": int(ev["duration_min"]),
                    "gat_path": str(gat_path),
                    "surrogate_path": str(surrogate_path),
                    "sensors": sensors,
                    "node_order": node_order,
                    "control_step_sec": int(cfg["experiment"]["control_step_sec"]),
                    "device": str(args.device),
                    "max_steps": int(args.max_steps),
                    "nominal_detail_csv": str(nominal_detail) if nominal_detail else None,
                    "no_control_detail_csv": str(no_control_detail_for_reference) if needs_no_control_reference else None,
                    "passive_detail_csv": str(passive_detail_for_reference) if needs_passive_reference else None,
                    "residual_value_path": str(residual_value_path) if residual_value_path else None,
                    "residual_pfv_prob_min": float(args.residual_pfv_prob_min),
                    "residual_safe_prob_min": float(args.residual_safe_prob_min),
                    "residual_nonzero_prob_min": float(args.residual_nonzero_prob_min),
                    "residual_peak_prob_min": float(args.residual_peak_prob_min),
                    "max_candidate_delta": float(effective_max_candidate_delta),
                    "topk_log_count": int(args.topk_log_count),
                    "max_candidate_count": int(args.max_candidate_count),
                    "candidate_hold_steps": shield_cfg.get("candidate_hold_steps", (1, 2, 3)),
                    "allowed_candidate_templates": shield_cfg.get("allowed_templates"),
                    "blocked_candidate_templates": shield_cfg.get("blocked_templates"),
                    "allowed_candidate_scopes_by_template": shield_cfg.get("allowed_scopes_by_template"),
                    "priority_khop": int(args.priority_khop),
                    "empirical_guard_path": str(empirical_guard_path) if empirical_guard_path else None,
                    "empirical_guard_unknown_allow": not bool(args.empirical_guard_block_unknown),
                    "boost_safe_prob_extra": float(args.boost_safe_prob_extra),
                    "boost_peak_prob_extra": float(args.boost_peak_prob_extra),
                    "protective_safe_prob_relief": float(args.protective_safe_prob_relief),
                    "release_peak_hold_max": int(args.release_peak_hold_max),
                    "low_risk_pfv_threshold": float(args.low_risk_pfv_threshold),
                    "high_risk_pfv_threshold": float(args.high_risk_pfv_threshold),
                    "release_recession_pfv_min": float(args.release_recession_pfv_min),
                    "release_recession_priority_depth_min": float(args.release_recession_priority_depth_min),
                    "strict_guard_return_period_max": int(args.strict_guard_return_period_max),
                    "strict_guard_patterns": str(args.strict_guard_patterns),
                    "strict_guard_prob_extra": float(args.strict_guard_prob_extra),
                    "horizon_smooth_weight": float(controller_cfg.get("horizon_smooth_weight", shield_cfg.get("horizon_smooth_weight", cfg.get("experiment", {}).get("smooth_weight", 0.05)))),
                    "horizon_violation_penalty": float(controller_cfg.get("horizon_violation_penalty", shield_cfg.get("horizon_violation_penalty", 1.0e6))),
                    "proposed_controller": str(args.proposed_controller),
                    "horizon_steps": int(controller_cfg.get("horizon_steps", (cfg.get("horizon_surrogate", {}) or {}).get("horizon_steps", 6))),
                    "priority_to_actuators_csv": str(priority_to_actuators_csv) if priority_to_actuators_csv.exists() else None,
                    "horizon_surrogate_model_path": str(horizon_surrogate_path),
                    "raw_joint_model_path": str(raw_joint_model_path) if is_temporal_joint_controller else None,
                    "temporal_joint_config": temporal_joint_cfg,
                    "action_effect_model_path": str(action_effect_model_path) if is_action_effect_controller else None,
                    "rainfall_csv": str(ev["rainfall_csv"]),
                    "generic_default_policy_id": str(controller_cfg.get("default_action_policy", "hold_previous_or_all_open_safe")),
                    "min_pfv_improvement_abs": float(controller_cfg.get("min_pfv_improvement_abs", cfg.get("experiment", {}).get("pfv_guard_min_improve_m3", 1.0))),
                    "min_pfv_improvement_frac": float(controller_cfg.get("min_pfv_improvement_frac", 0.0)),
                    "max_candidate_sequences": int(controller_cfg.get("max_candidate_sequences", 512)),
                    "candidate_group_limit": int(controller_cfg.get("candidate_group_limit", 12)),
                    "tfv_tolerance_abs": float(controller_cfg.get("tfv_tolerance_abs", 0.0)),
                    "tfv_tolerance_frac": float(controller_cfg.get("tfv_tolerance_frac", cfg.get("experiment", {}).get("tfv_guard_pct", 0.0))),
                    "peak_tolerance_abs": float(controller_cfg.get("peak_tolerance_abs", 0.0)),
                    "peak_tolerance_frac": float(controller_cfg.get("peak_tolerance_frac", cfg.get("experiment", {}).get("peak_guard_pct", 0.0))),
                    "pfv_tolerance_abs": float(controller_cfg.get("pfv_tolerance_abs", 0.0)),
                    "pfv_tolerance_frac": float(controller_cfg.get("pfv_tolerance_frac", 0.0)),
                    "tfv_required_reduction_abs": float(controller_cfg.get("tfv_required_reduction_abs", 0.0)),
                    "tfv_required_reduction_frac": float(controller_cfg.get("tfv_required_reduction_frac", 0.0)),
                    "tfv_required_reduction_dry_multiplier": float(controller_cfg.get("tfv_required_reduction_dry_multiplier", 1.0)),
                    "tfv_hard_constraint": bool(controller_cfg.get("tfv_hard_constraint", True)),
                    "dry_rain_threshold": float(controller_cfg.get("dry_rain_threshold", 0.10)),
                    "peak_weight": float(controller_cfg.get("peak_weight", 1.0)),
                    "pfv_weight": float(controller_cfg.get("pfv_weight", 1.0)),
                    "adaptive_delta_enabled": bool(controller_cfg.get("adaptive_delta_enabled", False)),
                    "low_risk_max_candidate_delta": float(controller_cfg.get("low_risk_max_candidate_delta", controller_cfg.get("max_candidate_delta", 0.08))),
                    "high_risk_max_candidate_delta": float(controller_cfg.get("high_risk_max_candidate_delta", controller_cfg.get("max_candidate_delta", 0.03))),
                    "pfv_high_risk_horizon_threshold": float(controller_cfg.get("pfv_high_risk_horizon_threshold", 1000.0)),
                    "pfv_low_risk_horizon_threshold": float(controller_cfg.get("pfv_low_risk_horizon_threshold", 100.0)),
                    "max_first_step_delta": float(controller_cfg.get("max_first_step_delta", 1.0)),
                    "per_actuator_max_delta": dict(controller_cfg.get("per_actuator_max_delta", {}) or {}),
                    "min_hold_steps_by_actuator": dict(controller_cfg.get("min_hold_steps_by_actuator", {}) or {}),
                    "objective_mode": str(controller_cfg.get("objective_mode", "pfv_first")),
                    "allowed_actuator_ids": event_allowed_actuator_ids,
                    "blocked_actuator_ids": controller_cfg.get("blocked_actuator_ids"),
                    "allowed_action_directions": event_allowed_action_directions,
                     "phase_reliability_csv": str((controller_cfg.get("phase_reliability", {}) or {}).get("exact_local_csv", "")),
                     "phase_reliability_allow_tfv_noninferior": bool((controller_cfg.get("phase_reliability", {}) or {}).get("allow_tfv_noninferior", False)),
                     "phase_reliability_require_pfv_improvement": bool((controller_cfg.get("phase_reliability", {}) or {}).get("require_pfv_improvement", False)),
                     "phase_reliability_pfv_tolerance_abs": float((controller_cfg.get("phase_reliability", {}) or {}).get("pfv_tolerance_abs", controller_cfg.get("pfv_tolerance_abs", 100.0))),
                     "phase_reliability_pfv_tolerance_frac": float((controller_cfg.get("phase_reliability", {}) or {}).get("pfv_tolerance_frac", controller_cfg.get("pfv_tolerance_frac", 0.005))),
                     "phase_reliability_tfv_tolerance_abs": float((controller_cfg.get("phase_reliability", {}) or {}).get("tfv_tolerance_abs", controller_cfg.get("tfv_tolerance_abs", 0.0))),
                     "phase_reliability_tfv_tolerance_frac": float((controller_cfg.get("phase_reliability", {}) or {}).get("tfv_tolerance_frac", controller_cfg.get("tfv_tolerance_frac", 0.0))),
                     "phase_reliability_peak_tolerance_abs": float((controller_cfg.get("phase_reliability", {}) or {}).get("peak_tolerance_abs", controller_cfg.get("peak_tolerance_abs", 0.75))),
                     "phase_reliability_peak_tolerance_frac": float((controller_cfg.get("phase_reliability", {}) or {}).get("peak_tolerance_frac", controller_cfg.get("peak_tolerance_frac", 0.005))),
                     "phase_reliability_pulse_steps": int((controller_cfg.get("phase_reliability", {}) or {}).get("pulse_steps", 2)),
                     "phase_reliability_max_overrides": int((controller_cfg.get("phase_reliability", {}) or {}).get("max_overrides_per_phase", 0)),
                     "phase_reliability_candidate_group_limit": int((controller_cfg.get("phase_reliability", {}) or {}).get("candidate_group_limit", 4)),
                     "empirical_single_action_gate": bool((controller_cfg.get("phase_reliability", {}) or {}).get("empirical_single_action_gate", False)),
                    "empirical_hold_steps": int((controller_cfg.get("phase_reliability", {}) or {}).get("pulse_steps", 2)),
                    "phase_reliability_fallback_to_surrogate": bool((controller_cfg.get("phase_reliability", {}) or {}).get("fallback_to_surrogate_without_local_evidence", False)),
                    "phase_reliability_evidence_time_tolerance_min": float((controller_cfg.get("phase_reliability", {}) or {}).get("evidence_time_tolerance_min", 2.5)),
                    "pump_control_mode": str(controller_cfg.get("pump_control_mode", "continuous")),
                    "variable_speed_pump_ids": list(controller_cfg.get("variable_speed_pump_ids", [])),
                    "v4_quantile": float(((controller_cfg.get("dual_reference", {}) or {}).get("pfv_event_quantile", 0.95))),
                    "v4_pfv_abs_margin_m3": float(((controller_cfg.get("dual_reference", {}) or {}).get("pfv_abs_margin_m3", 0.0))),
                    "v4_pfv_rel_margin": float(((controller_cfg.get("dual_reference", {}) or {}).get("pfv_rel_margin", 0.0))),
                    "v4_max_k": int(controller_cfg.get("max_simultaneous_residual_overrides", controller_cfg.get("candidate_group_limit", 8))),
                    "v4_readback_tolerance": float(((controller_cfg.get("readback_hard_constraint", {}) or {}).get("tolerance", 1.0e-4))),
                    "v4_action_deadband": float(((controller_cfg.get("action_cost", {}) or {}).get("setting_deadband", 0.02))),
                    "v4_adaptive_k_values": list(((controller_cfg.get("adaptive_k", {}) or {}).get("allowed_values", [0, 2, 4, 6, 8]))),
                    "v4_changed_facility_penalty": float(((controller_cfg.get("action_cost", {}) or {}).get("changed_facility_penalty", 1.0))),
                    "v4_variation_penalty": float(((controller_cfg.get("action_cost", {}) or {}).get("variation_penalty", 1.0))),
                    "v4_reversal_penalty": float(((controller_cfg.get("action_cost", {}) or {}).get("reversal_penalty", 5.0))),
                    "v4_minimum_material_benefit": float(((controller_cfg.get("action_cost", {}) or {}).get("minimum_material_benefit", 0.0))),
                    "v4_minimum_benefit_cost_ratio": float(((controller_cfg.get("action_cost", {}) or {}).get("minimum_benefit_cost_ratio", 1.5))),
                }
            )
    if proposed_jobs:
        proposed_workers = max(1, int(args.proposed_workers))
        if proposed_workers > 1 and str(args.device).lower() == "cuda":
            print("[closed_loop] warning: --proposed-workers > 1 with --device cuda may duplicate model memory; use cpu if CUDA memory is tight.")
        print(f"[closed_loop] running {len(proposed_jobs)} proposed SWMM-MPC jobs with workers={proposed_workers}")
        if proposed_workers == 1:
            for i, job in enumerate(proposed_jobs, 1):
                print(f"[closed_loop] proposed {i}/{len(proposed_jobs)} {job['event_id']}")
                proposed_rows.append(_run_proposed_job(job))
                print(f"[closed_loop] done proposed {job['event_id']}")
        else:
            with ProcessPoolExecutor(max_workers=proposed_workers) as ex:
                futs = [ex.submit(_run_proposed_job, job) for job in proposed_jobs]
                for i, fut in enumerate(as_completed(futs), 1):
                    row = fut.result()
                    proposed_rows.append(row)
                    print(f"[closed_loop] done proposed {i}/{len(proposed_jobs)} {row.get('event_id')}")

    if baseline_jobs:
        workers = max(1, int(args.workers))
        print(f"[closed_loop] running {len(baseline_jobs)} baseline SWMM jobs with workers={workers}")
        if workers == 1:
            for i, job in enumerate(baseline_jobs, 1):
                print(f"[closed_loop] baseline {i}/{len(baseline_jobs)} {job['event_id']} {job['policy']}")
                baseline_rows.append(_run_baseline_job(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_baseline_job, job) for job in baseline_jobs]
                for i, fut in enumerate(as_completed(futs), 1):
                    row = fut.result()
                    baseline_rows.append(row)
                    print(f"[closed_loop] done baseline {i}/{len(baseline_jobs)} {row.get('event_id')} {row.get('policy_id')}")
    if baseline_rows:
        pd.DataFrame(baseline_rows).to_csv(out_root / "baseline_results.csv", index=False)
    if proposed_rows:
        pd.DataFrame(proposed_rows).to_csv(out_root / "proposed_results.csv", index=False)
    report = {
        "mode": args.mode,
        "run_tag": args.run_tag,
        "events": len(rain_table),
        "event_ids": rain_table["event_id"].astype(str).tolist(),
        "out": str(out_root),
        "paper_policy_set": paper_policy_ids(cfg),
        "baseline_policies": baseline_policies,
        "gat_path": str(gat_path),
        "surrogate_path": str(surrogate_path),
        "residual_value_path": str(residual_value_path) if residual_value_path else "",
        "residual_pfv_prob_min": float(args.residual_pfv_prob_min),
        "residual_safe_prob_min": float(args.residual_safe_prob_min),
        "residual_nonzero_prob_min": float(args.residual_nonzero_prob_min),
        "residual_peak_prob_min": float(args.residual_peak_prob_min),
        "max_candidate_delta": float(effective_max_candidate_delta),
        "proposed_workers": int(args.proposed_workers),
        "repair_reliable_allowed_actuator_count_by_event": {
            event_id: int(len(ids)) for event_id, ids in repair_allowed_actuators_by_event.items()
        },
        "repair_reliable_allowed_actuator_ids_by_event": repair_allowed_actuators_by_event,
        "topk_log_count": int(args.topk_log_count),
        "max_candidate_count": int(args.max_candidate_count),
        "priority_khop": int(args.priority_khop),
        "empirical_guard_path": str(empirical_guard_path) if empirical_guard_path else "",
        "empirical_guard_unknown_allow": not bool(args.empirical_guard_block_unknown),
        "boost_safe_prob_extra": float(args.boost_safe_prob_extra),
        "boost_peak_prob_extra": float(args.boost_peak_prob_extra),
        "protective_safe_prob_relief": float(args.protective_safe_prob_relief),
        "release_peak_hold_max": int(args.release_peak_hold_max),
        "low_risk_pfv_threshold": float(args.low_risk_pfv_threshold),
        "high_risk_pfv_threshold": float(args.high_risk_pfv_threshold),
        "release_recession_pfv_min": float(args.release_recession_pfv_min),
        "release_recession_priority_depth_min": float(args.release_recession_priority_depth_min),
        "strict_guard_return_period_max": int(args.strict_guard_return_period_max),
        "strict_guard_patterns": str(args.strict_guard_patterns),
        "strict_guard_prob_extra": float(args.strict_guard_prob_extra),
        "proposed_controller": str(args.proposed_controller),
        "action_effect_model_path": str(action_effect_model_path) if is_action_effect_controller else "",
        "uses_internal_rules_as_nominal": bool(use_native_nominal),
        "constraint_reference_policy": str(reference_policy),
        "uses_internal_rules_as_constraint_reference": bool(needs_internal_reference),
        "uses_no_control_as_constraint_reference": bool(needs_no_control_reference),
        "uses_offline_true_no_control_reference": bool(needs_no_control_reference and not is_v4_controller),
        "v4_online_reference_contract": "causal_model_only" if is_v4_controller else "not_applicable",
        "proposed_live_branch_uses_native_rules": bool(proposed_uses_native_rules and not is_v4_controller),
        "v4_internal_fallback_source": "synchronized_causal_native_rule_shadow_current_setting" if is_v4_controller else "not_applicable",
        "uses_online_predicted_no_control_reference": bool(
            is_temporal_joint_controller
            and str(reference_policy).strip().lower() == "online_predicted_default"
        ),
        "objective_mode": str(controller_cfg.get("objective_mode", "pfv_first")),
        "default_action_policy": str(controller_cfg.get("default_action_policy", "hold_previous_or_all_open_safe")),
        "actuator_scope": actuator_scope,
        "action_space_actuator_count": int(len(actuators)),
        "control_enabled_actuator_count": int(
            actuators.get("control_enabled", pd.Series(True, index=actuators.index)).fillna(True).astype(bool).sum()
        ),
        "control_actuator_table": str(scoped_actuator_path),
        "pfv_tolerance_abs": float(
            (temporal_joint_cfg.get("safety", {}) or {}).get("pfv_abs_margin_m3", 0.0)
            if is_temporal_joint_controller
            else controller_cfg.get("pfv_tolerance_abs", 0.0)
        ),
        "pfv_tolerance_frac": float(
            (temporal_joint_cfg.get("safety", {}) or {}).get("pfv_rel_margin", 0.0)
            if is_temporal_joint_controller
            else controller_cfg.get("pfv_tolerance_frac", 0.0)
        ),
        "tfv_required_reduction_abs": float(
            (temporal_joint_cfg.get("safety", {}) or {}).get("min_tfv_lcb_reduction", 0.0)
            if is_temporal_joint_controller
            else controller_cfg.get("tfv_required_reduction_abs", 0.0)
        ),
        "tfv_required_reduction_frac": float(controller_cfg.get("tfv_required_reduction_frac", 0.0)),
        "peak_tolerance_abs": float(
            (temporal_joint_cfg.get("safety", {}) or {}).get("peak_margin", 0.0)
            if is_temporal_joint_controller
            else controller_cfg.get("peak_tolerance_abs", 0.0)
        ),
        "peak_tolerance_frac": float(
            0.0
            if is_temporal_joint_controller
            else controller_cfg.get("peak_tolerance_frac", 0.0)
        ),
        "horizon_surrogate_path": str(horizon_surrogate_path),
        "proposed_base": args.proposed_base,
        "hierarchical_preflight": hierarchical_preflight,
        "passed": True,
    }
    if args.proposed_controller == "native_shield":
        report["legacy_native_shield"] = {
            "nominal_policy": str(shield_cfg.get("nominal_policy", "internal_rules")),
            "require_on_policy_guard": bool(shield_cfg.get("require_project5_on_policy_guard", False)),
            "legacy_residual_use": str(shield_cfg.get("project4_residual_use", "")),
            "allowed_templates": shield_cfg.get("allowed_templates", []),
            "blocked_templates": shield_cfg.get("blocked_templates", []),
            "allowed_scopes_by_template": shield_cfg.get("allowed_scopes_by_template", {}),
            "candidate_hold_steps": shield_cfg.get("candidate_hold_steps", []),
        }
    (out_root / "closed_loop_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
