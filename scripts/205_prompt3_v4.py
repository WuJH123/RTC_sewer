from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from sewerrtc.io.project_paths import load_config
from sewerrtc.io.safe_paths import short_run_tag, single_writer_lease
from sewerrtc.prompt3 import action_effect_v4 as v4
from sewerrtc.prompt3 import action_effect_v4_aug1 as v4a
from sewerrtc.simulation.runtime_contracts import sha256_file, utc_now, write_json


def _cfg(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _root(config: str | Path) -> Path:
    cfg = _cfg(config)
    project = Path((cfg.get("project", {}) or {}).get("root", Path(config).resolve().parents[1]))
    raw = Path((cfg.get("project", {}) or {}).get("output_root", "outputs/project6_dual_reference_v4"))
    return raw if raw.is_absolute() else project / raw


def _project(config: str | Path) -> Path:
    return Path((_cfg(config).get("project", {}) or {}).get("root", Path(config).resolve().parents[1]))


def _resolve_gat(project: Path) -> Path:
    locks = [
        project / "outputs/project6_pfvfirst_dualfallback_10min_v3/gat/gat_primary_selection_lock.json",
        project / "outputs/project6_pfvfirst_dualfallback_10min_v3_3/gat/gat_primary_selection_lock.json",
    ]
    for lock in locks:
        if lock.exists():
            try:
                path = Path(json.loads(lock.read_text(encoding="utf-8")).get("checkpoint_path", ""))
                if not path.is_absolute():
                    path = project / path
                if path.exists():
                    return path
            except Exception:
                pass
    candidates = list(project.glob("outputs/**/gat*sr0p15*.pt")) + list(project.glob("outputs/**/gat*.pt"))
    return candidates[0] if candidates else project / "outputs/MISSING_GAT_SR0P15.pt"


def _resolve_effect_model(root: Path) -> Path:
    """Prefer the retrained Aug1 action-effect model when present, else base.

    The Aug1 model repairs the full-event PFV direction heads, so once it has
    been trained the closed-loop smoke must validate it. When the Aug1 model is
    absent (base-only flow), fall back to the frozen base V4 model.
    """
    aug1 = root / "action_effect_models_v4_aug1" / "action_effect_dual_reference_v4_aug1.npz"
    if aug1.exists():
        return aug1
    return root / "action_effect_models_v4/action_effect_dual_reference_v4.npz"


def build_runner_config(config: str | Path) -> tuple[int, dict[str, Path]]:
    cfg = _cfg(config)
    project = _project(config)
    root = _root(config)
    base_raw = Path((cfg.get("formal_evaluation", {}) or {}).get("authoritative_swmm_runner_config", "configs/wuhan_project6_v8_storage.yaml"))
    base_path = base_raw if base_raw.is_absolute() else project / base_raw
    if not base_path.exists():
        report = write_json(root / "audit/v4_runner_config_report.json", {"status": "contract_mismatch", "reason": f"missing_base_runner_config:{base_path}"})
        return 6, {"report": report}
    base = load_config(base_path)
    model = _resolve_effect_model(root)
    gat = _resolve_gat(project)
    bridge = dict(base)
    bridge["project_root"] = str(project)
    bridge["network"] = dict(bridge.get("network", {}) or {})
    bridge["network"]["inp"] = (cfg.get("project", {}) or {}).get("inp", "data/wuhan_v8_storage_retrofit.inp")
    bridge["controller"] = dict(bridge.get("controller", {}) or {})
    dual = dict(((cfg.get("v4", {}) or {}).get("dual_reference", {}) or {}))
    rb = dict(((cfg.get("v4", {}) or {}).get("readback_hard_constraint", {}) or {}))
    action_cost = dict(((cfg.get("v4", {}) or {}).get("action_cost", {}) or {}))
    adaptive_k_cfg = dict(((cfg.get("v4", {}) or {}).get("adaptive_k", {}) or {}))
    bridge["controller"].update({
        "mode": "proposed_dual_reference_v4",
        "action_effect_model_path": str(model),
        "gat_model_path": str(gat),
        "reference_policy_for_constraints": "causal_model_only",
        "prohibit_offline_future_hydraulic_reference": True,
        "default_action_policy": "hold_previous_or_all_open_safe",
        "horizon_steps": 12,
        "objective_mode": "pfv_preserving_system_repair",
        "min_pfv_improvement_abs": 0.0,
        "min_pfv_improvement_frac": 0.0,
        "pfv_tolerance_abs": float(dual.get("pfv_abs_margin_m3", 0.0)),
        "pfv_tolerance_frac": float(dual.get("pfv_rel_margin", 0.0)),
        "tfv_tolerance_abs": float(dual.get("tfv_abs_margin_m3", 0.0)),
        "tfv_tolerance_frac": float(dual.get("tfv_rel_margin", 0.0)),
        "peak_tolerance_abs": float(dual.get("peak_abs_margin", 0.0)),
        "peak_tolerance_frac": float(dual.get("peak_rel_margin", 0.0)),
        "tfv_hard_constraint": True,
        "candidate_group_limit": int((((cfg.get("v4", {}) or {}).get("adaptive_k", {}) or {}).get("max_k", 8))),
        "max_simultaneous_residual_overrides": int((((cfg.get("v4", {}) or {}).get("adaptive_k", {}) or {}).get("max_k", 8))),
        "dual_reference": dual,
        "readback_hard_constraint": rb,
        "action_cost": action_cost,
        "adaptive_k": adaptive_k_cfg,
        "pump_control_mode": "binary_unless_verified",
        "variable_speed_pump_ids": ["add350.1"],
    })
    bridge["experiment"] = dict(bridge.get("experiment", {}) or {})
    bridge["experiment"]["control_step_sec"] = 600
    bridge["outputs"] = dict(bridge.get("outputs", {}) or {})
    bridge["outputs"]["closed_loop"] = str(root / "cl")
    bridge["evaluation"] = dict(bridge.get("evaluation", {}) or {})
    bridge["evaluation"]["paper_policy_set"] = [
        "proposed_dual_reference_v4", "internal_rules", "no_control", "executable_passive"
    ]
    bridge["v4_bridge"] = {
        "source_config": str(config), "source_config_sha256": sha256_file(Path(config)),
        "base_runner_config": str(base_path), "base_runner_config_sha256": sha256_file(base_path),
        "model": str(model), "model_sha256": sha256_file(model),
        "gat": str(gat), "gat_sha256": sha256_file(gat),
        "created_at": utc_now(),
    }
    out = root / "runtime/v4_runner.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(bridge, sort_keys=False, allow_unicode=True), encoding="utf-8")
    missing = [name for name, path in {"model": model, "gat": gat, "network": project / bridge["network"]["inp"]}.items() if not path.exists()]
    status = "pass" if not missing else "blocked"
    report = write_json(root / "audit/v4_runner_config_report.json", {
        "status": status, "missing_assets": missing, "runner_config": str(out),
        "runner_config_sha256": sha256_file(out), "bridge": bridge["v4_bridge"],
    })
    return (0 if status == "pass" else 3), {"runner_config": out, "report": report}


def _latest_mtime(root: Path) -> float:
    latest = root.stat().st_mtime if root.exists() else 0.0
    if root.exists():
        for path in root.rglob("*"):
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                pass
    return latest


def run_smoke(config: str | Path, max_events: int, workers: int, resume: bool) -> tuple[int, dict[str, Path]]:
    code, outputs = build_runner_config(config)
    if code != 0:
        return code, outputs
    cfg = _cfg(config)
    project = _project(config)
    root = _root(config)
    runtime = dict(cfg.get("runtime_limits", {}) or {})
    tag = short_run_tag(f"p6v4_dev_{max_events}", max_length=24)
    out_dir = root / "cl/formal" / tag
    report_path = root / "audit/v4_smoke_run.json"
    model = _resolve_effect_model(root)
    cmd = [
        sys.executable, str(project / "scripts/08_run_closed_loop.py"),
        "--config", str(outputs["runner_config"]), "--mode", "formal",
        "--run-tag", tag, "--max-events", str(max(1, int(max_events))),
        "--baseline-policies", "internal_rules,no_control,executable_passive",
        "--proposed-controller", "proposed_dual_reference_v4", "--proposed-base", "native",
        "--action-effect-model", str(model), "--workers", str(max(1, int(workers))),
        "--proposed-workers", str(max(1, min(int((cfg.get("formal_evaluation", {}) or {}).get("proposed_workers", 2)), int(workers)))),
        "--device", "cpu", "--disable-pfv-positive-debug-filter",
    ]
    if resume:
        cmd.append("--skip-existing")
    retries = max(0, int(runtime.get("retry_count", 2)))
    heartbeat_sec = max(10, int(runtime.get("heartbeat_interval_sec", 60)))
    stall_sec = max(heartbeat_sec * 2, int(runtime.get("no_heartbeat_stall_sec", 1800)))
    hard_sec = max(stall_sec, int(runtime.get("per_event_timeout_sec", 43200))) * max(1, int(max_events))
    stdout_path = root / "logs/v4_smoke_stdout.txt"
    stderr_path = root / "logs/v4_smoke_stderr.txt"
    heartbeat = root / "runtime/v4_smoke_heartbeat.json"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    with single_writer_lease(out_dir, owner="v4_smoke"):
        for attempt in range(retries + 1):
            started = time.time()
            last_progress = started
            last_mtime = _latest_mtime(out_dir)
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
                proc = subprocess.Popen(cmd, cwd=str(project), stdout=stdout, stderr=stderr, text=True)
                while proc.poll() is None:
                    time.sleep(heartbeat_sec)
                    current_mtime = _latest_mtime(out_dir)
                    if current_mtime > last_mtime + 1.0e-6:
                        last_progress, last_mtime = time.time(), current_mtime
                    write_json(heartbeat, {
                        "status": "running", "pid": proc.pid, "attempt": attempt,
                        "elapsed_sec": time.time() - started,
                        "seconds_since_output_progress": time.time() - last_progress,
                        "command": cmd, "updated_at": utc_now(),
                    })
                    if time.time() - last_progress > stall_sec or time.time() - started > hard_sec:
                        write_json(heartbeat, {"status": "hung_or_timeout", "pid": proc.pid, "attempt": attempt, "command": cmd, "updated_at": utc_now()})
                        proc.kill()
                        break
                returncode = proc.wait()
            if returncode == 0:
                report = write_json(report_path, {"status": "pass", "attempt": attempt, "command": cmd, "out_dir": str(out_dir), "stdout": str(stdout_path), "stderr": str(stderr_path), "created_at": utc_now()})
                return 0, {"report": report, "out_dir": out_dir}
            if attempt < retries:
                time.sleep(max(1, int(runtime.get("retry_backoff_sec", 60))))
        report = write_json(report_path, {"status": "failed_runtime", "returncode": returncode, "command": cmd, "out_dir": str(out_dir), "stdout": str(stdout_path), "stderr": str(stderr_path), "created_at": utc_now()})
        return 4, {"report": report, "out_dir": out_dir}


def evaluate_smoke(config: str | Path) -> tuple[int, dict[str, Path]]:
    cfg = _cfg(config)
    root = _root(config)
    run = root / "audit/v4_smoke_run.json"
    payload = json.loads(run.read_text(encoding="utf-8")) if run.exists() else {}
    out_dir = Path(payload.get("out_dir", ""))
    proposed_path, baseline_path = out_dir / "proposed_results.csv", out_dir / "baseline_results.csv"
    structural_failures: list[str] = []
    scientific_failures: list[str] = []
    if payload.get("status") != "pass": structural_failures.append("smoke_run_not_pass")
    if not proposed_path.exists(): structural_failures.append("proposed_results_missing")
    if not baseline_path.exists(): structural_failures.append("baseline_results_missing")
    summary: dict[str, Any] = {}
    if not structural_failures:
        proposed = pd.read_csv(proposed_path)
        baseline = pd.read_csv(baseline_path)
        summary["proposed_event_count"] = int(proposed["event_id"].nunique())
        summary["baseline_policy_counts"] = baseline.groupby("policy_id")["event_id"].nunique().to_dict()
        if summary["proposed_event_count"] < 3:
            structural_failures.append("fewer_than_3_proposed_events")
        if any(int(summary["baseline_policy_counts"].get(policy, 0)) < 3 for policy in ("internal_rules", "no_control", "executable_passive")):
            structural_failures.append("incomplete_three_policy_pairing")
        metric_map = {"PFV": "PFV", "TFV": "TFV", "peak": "peak_TFV_rate"}
        paired: dict[str, pd.DataFrame] = {}
        for policy in ("internal_rules", "no_control", "executable_passive"):
            ref = baseline[baseline["policy_id"] == policy]
            joined = proposed.merge(ref, on="event_id", suffixes=("_proposed", "_baseline"))
            paired[policy] = joined
            for metric_name, column in metric_map.items():
                if len(joined) and f"{column}_proposed" in joined and f"{column}_baseline" in joined:
                    summary[f"mean_delta_{metric_name}_vs_{policy}"] = float((joined[f"{column}_proposed"] - joined[f"{column}_baseline"]).mean())
        dual = dict(((cfg.get("v4", {}) or {}).get("dual_reference", {}) or {}))
        pfv_abs = float(dual.get("pfv_abs_margin_m3", 0.0))
        pfv_rel = float(dual.get("pfv_rel_margin", 0.0))
        for policy in ("no_control", "executable_passive"):
            joined = paired.get(policy, pd.DataFrame())
            if len(joined):
                cap = joined["PFV_baseline"] + np.maximum(pfv_abs, pfv_rel * joined["PFV_baseline"].clip(lower=0.0))
                if bool((joined["PFV_proposed"] > cap + 1.0e-9).any()):
                    scientific_failures.append(f"PFV_event_noninferiority_failed_vs_{policy}")
        internal = paired.get("internal_rules", pd.DataFrame())
        if len(internal):
            tfv_cap = internal["TFV_baseline"] + np.maximum(float(dual.get("tfv_abs_margin_m3", 0.0)), float(dual.get("tfv_rel_margin", 0.0)) * internal["TFV_baseline"].clip(lower=0.0))
            peak_cap = internal["peak_TFV_rate_baseline"] + np.maximum(float(dual.get("peak_abs_margin", 0.0)), float(dual.get("peak_rel_margin", 0.0)) * internal["peak_TFV_rate_baseline"].clip(lower=0.0))
            if bool((internal["TFV_proposed"] > tfv_cap + 1.0e-9).any()):
                scientific_failures.append("TFV_event_noninferiority_failed_vs_internal")
            if bool((internal["peak_TFV_rate_proposed"] > peak_cap + 1.0e-9).any()):
                scientific_failures.append("peak_event_noninferiority_failed_vs_internal")
        history_files = sorted((out_dir / "proposed").glob("*__controller_history.csv"))
        summary["history_file_count"] = len(history_files)
        if len(history_files) < 3:
            structural_failures.append("controller_history_incomplete")
        histories = []
        for path in history_files:
            try:
                frame = pd.read_csv(path)
                frame["history_file"] = str(path)
                histories.append(frame)
            except Exception as exc:
                structural_failures.append(f"history_unreadable:{path.name}:{type(exc).__name__}")
        if histories:
            hist = pd.concat(histories, ignore_index=True, sort=False)
            future_used = hist.get("online_future_hydraulics_used", pd.Series(False, index=hist.index)).astype(str).str.lower().isin(["1", "true", "yes"])
            summary["online_future_hydraulics_used_count"] = int(future_used.sum())
            if bool(future_used.any()): structural_failures.append("truth_future_leakage_detected")
            if "write_readback_match" in hist:
                mismatch = ~hist["write_readback_match"].fillna(False).astype(str).str.lower().isin(["1", "true", "yes"])
                summary["write_readback_mismatch_count"] = int(mismatch.sum())
                if bool(mismatch.any()): structural_failures.append("write_readback_mismatch")
            changed = pd.to_numeric(hist.get("selected_changed_facilities", 0), errors="coerce").fillna(0)
            k_limit = pd.to_numeric(hist.get("selected_adaptive_k_limit", 0), errors="coerce").fillna(0)
            k_violation = changed > k_limit
            summary["adaptive_k_violation_count"] = int(k_violation.sum())
            if bool(k_violation.any()): structural_failures.append("adaptive_k_post_projection_violation")
            fallback = hist.get("fallback_to_default", pd.Series(True, index=hist.index)).astype(str).str.lower().isin(["1", "true", "yes"])
            selected = hist.get("selected_sequence_label", pd.Series("", index=hist.index)).astype(str)
            candidate_executed = (~fallback) & selected.ne("hold_native")
            summary["candidate_executed_count"] = int(candidate_executed.sum())
            summary["passive_fallback_count"] = int((hist.get("selected_fallback", "").astype(str) == "passive_anchor").sum()) if "selected_fallback" in hist else 0
            summary["internal_fallback_count"] = int((hist.get("selected_fallback", "").astype(str) == "internal_rules").sum()) if "selected_fallback" in hist else 0
            if not bool(candidate_executed.any()): scientific_failures.append("controller_degenerated_to_all_fallback")
    structural_failures = sorted(set(structural_failures))
    scientific_failures = sorted(set(scientific_failures))
    status = "blocked" if structural_failures else ("failed_gate" if scientific_failures else "pass")
    gate = write_json(root / "audit/v4_smoke_gate.json", {
        "status": status, "structural_failures": structural_failures,
        "scientific_failures": scientific_failures, "summary": summary,
        "authoritative_source_required": "SWMM", "created_at": utc_now(),
    })
    return (3 if structural_failures else (5 if scientific_failures else 0)), {"gate": gate}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["BuildV4Dataset", "TrainV4", "EvaluateV4ModelGate", "BuildRunnerConfigV4", "AuditV4Readiness", "RunClosedLoopSmokeV4", "EvaluateClosedLoopSmokeV4", "DiagnoseV4FullEventPFVGate", "PlanV4DualReferenceFullEventCases", "GenerateV4DualReferenceFullEventCases", "BuildV4AugmentedDataset", "TrainV4Aug1", "EvaluateV4Aug1ModelGate"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--max-events", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "BuildV4Dataset": code, outputs = v4.build_v4_dataset(args.config, smoke=args.smoke, max_samples=args.max_samples)
    elif args.stage == "TrainV4": code, outputs = v4.train_v4_ensemble(args.config, smoke=args.smoke, ensemble_size=args.ensemble_size)
    elif args.stage == "EvaluateV4ModelGate": code, outputs = v4.evaluate_v4_model_gate(args.config, smoke=args.smoke)
    elif args.stage == "BuildRunnerConfigV4": code, outputs = build_runner_config(args.config)
    elif args.stage == "AuditV4Readiness": code, outputs = v4.audit_v4_readiness(args.config)
    elif args.stage == "DiagnoseV4FullEventPFVGate": code, outputs = v4.diagnose_v4_full_event_pfv_gate(args.config, smoke=args.smoke)
    elif args.stage == "PlanV4DualReferenceFullEventCases": code, outputs = v4a.plan_v4_dual_reference_full_event_cases(args.config, smoke=args.smoke, max_cases=args.max_samples)
    elif args.stage == "GenerateV4DualReferenceFullEventCases": code, outputs = v4a.generate_v4_dual_reference_full_event_cases(args.config, smoke=args.smoke, max_cases=args.max_samples, workers=args.workers, resume=args.resume)
    elif args.stage == "BuildV4AugmentedDataset": code, outputs = v4a.build_v4_augmented_dataset(args.config, smoke=args.smoke)
    elif args.stage == "TrainV4Aug1": code, outputs = v4a.train_v4_aug1(args.config, smoke=args.smoke, ensemble_size=args.ensemble_size)
    elif args.stage == "EvaluateV4Aug1ModelGate": code, outputs = v4a.evaluate_v4_aug1_model_gate(args.config, smoke=args.smoke)
    elif args.stage == "RunClosedLoopSmokeV4": code, outputs = run_smoke(args.config, args.max_events, args.workers, args.resume)
    else: code, outputs = evaluate_smoke(args.config)
    print(json.dumps({"status_code": code, "outputs": {k: str(v) for k, v in outputs.items()}}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
