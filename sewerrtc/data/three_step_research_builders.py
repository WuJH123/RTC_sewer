from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sewerrtc.data.peak_label_semantics import RISK_LABEL_CHANNELS


def _selected_manifest(manifest_path: str | Path, flag_column: str) -> pd.DataFrame:
    table = pd.read_csv(manifest_path)
    if flag_column in table:
        table = table[table[flag_column].fillna(False).astype(bool)].copy()
    if "detail_file" not in table:
        raise ValueError(f"manifest is missing detail_file: {manifest_path}")
    table["detail_file"] = table["detail_file"].astype(str)
    return table


def _read_detail(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _existing_columns(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        line = handle.readline().strip()
    return [part.strip().strip('"') for part in line.split(",")] if line else []


def _numeric_matrix(frame: pd.DataFrame, columns: list[str], *, fill: float = 0.0) -> np.ndarray:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").fillna(fill).to_numpy(np.float32)


def _event_policy(row: pd.Series, path: Path) -> tuple[str, str]:
    event = str(row.get("event_id", "") or "")
    policy = str(row.get("policy_id", "") or "")
    if event and policy:
        return event, policy
    stem = path.stem.removesuffix("_detail")
    event, sep, policy = stem.rpartition("__")
    return (event, policy) if sep else (stem, "unknown")


def _limit_rows(samples: list[int], max_samples: int) -> list[int]:
    return samples[: max(0, int(max_samples))] if int(max_samples or 0) > 0 else samples


def build_mixed_gat_cache(
    *,
    manifest_path: str | Path,
    out_npz: str | Path,
    base_node_cols: Iterable[str],
    time_stride: int = 1,
    max_files: int = 0,
    max_samples: int = 0,
) -> dict:
    manifest = _selected_manifest(manifest_path, "gat_use")
    if int(max_files or 0) > 0:
        manifest = manifest.head(int(max_files)).copy()
    node_cols = [str(c) for c in base_node_cols]
    state_rows: list[np.ndarray] = []
    rain_rows: list[np.ndarray] = []
    sources: list[str] = []
    event_ids: list[str] = []
    policy_ids: list[str] = []
    failures = []
    stride = max(1, int(time_stride))
    for _, row in manifest.iterrows():
        path = Path(str(row["detail_file"]))
        try:
            cols = _existing_columns(path)
            usecols = [c for c in cols if c in set(node_cols) or c in {"rainfall_mm_h"}]
            detail = _read_detail(path, usecols=usecols)
            states = _numeric_matrix(detail, node_cols)
            rainfall = (
                pd.to_numeric(detail["rainfall_mm_h"], errors="coerce").fillna(0.0).to_numpy(np.float32)[:, None]
                if "rainfall_mm_h" in detail
                else np.zeros((len(detail), 1), dtype=np.float32)
            )
            event_id, policy_id = _event_policy(row, path)
            for i in range(0, len(detail), stride):
                state_rows.append(states[i])
                rain_rows.append(rainfall[i])
                sources.append(f"{path.name}:{i}")
                event_ids.append(event_id)
                policy_ids.append(policy_id)
                if max_samples and len(state_rows) >= int(max_samples):
                    break
        except Exception as exc:
            failures.append({"detail_file": str(path), "error": repr(exc)})
        if max_samples and len(state_rows) >= int(max_samples):
            break
    if not state_rows:
        raise RuntimeError("No GAT samples were built from the manifest.")
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        state=np.asarray(state_rows, dtype=np.float32),
        rain=np.asarray(rain_rows, dtype=np.float32),
        node_cols=np.asarray(node_cols, dtype=object),
        sources=np.asarray(sources, dtype=object),
        event_ids=np.asarray(event_ids, dtype=object),
        policy_ids=np.asarray(policy_ids, dtype=object),
    )
    report = {
        "out_npz": str(out_npz),
        "manifest": str(manifest_path),
        "files_requested": int(len(manifest)),
        "samples": int(len(state_rows)),
        "events": int(len(set(event_ids))),
        "policies": sorted(set(policy_ids)),
        "nodes": int(len(node_cols)),
        "time_stride": int(stride),
        "max_files": int(max_files or 0),
        "max_samples": int(max_samples or 0),
        "failures": failures[:50],
        "failure_count": int(len(failures)),
    }
    out_npz.with_suffix(".meta.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_temporal_action_pretrain_dataset(
    *,
    manifest_path: str | Path,
    out_npz: str | Path,
    base_node_cols: Iterable[str],
    canonical_action_ids: Iterable[str],
    priority_nodes: Iterable[str] | None = None,
    local_node_cols: Iterable[str] | None = None,
    horizon_steps: int,
    target_mode: str = "risk_local",
    time_stride: int = 1,
    max_files: int = 0,
    max_samples: int = 0,
    chunk_size_samples: int = 0,
) -> dict:
    manifest = _selected_manifest(manifest_path, "action_learning_use")
    if int(max_files or 0) > 0:
        manifest = manifest.head(int(max_files)).copy()
    node_cols = [str(c) for c in base_node_cols]
    node_ids = [c.split(":", 1)[1] if c.startswith("h:") else c for c in node_cols]
    priority = {str(n) for n in (priority_nodes or [])}
    local_cols = [str(c) for c in (local_node_cols or [])]
    local_cols = [c if c.startswith("h:") else f"h:{c}" for c in local_cols]
    local_cols = [c for c in local_cols if c in node_cols]
    if not local_cols:
        local_cols = [c for c, node in zip(node_cols, node_ids) if node in priority]
    if not local_cols:
        local_cols = node_cols[: min(64, len(node_cols))]
    action_ids = [str(a) for a in canonical_action_ids]
    action_cols = [f"a:{aid}" for aid in action_ids]
    H = max(1, int(horizon_steps))
    stride = max(1, int(time_stride))
    mode = str(target_mode or "risk_local").strip().lower()
    if mode not in {"risk_local", "full_state"}:
        raise ValueError("target_mode must be risk_local or full_state")
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = max(0, int(chunk_size_samples or 0))
    sharded = chunk_size > 0
    shard_dir = out_npz.parent / f"{out_npz.stem}_chunks"
    if sharded:
        shard_dir.mkdir(parents=True, exist_ok=True)
        for stale in shard_dir.glob("chunk_*.npz"):
            stale.unlink()

    output = {key: [] for key in ["state", "candidate_action_seq", "rain_seq", "risk_rate_seq", "local_state_seq"]}
    if mode == "full_state":
        output["target_state_seq"] = []
    event_ids: list[str] = []
    policy_ids: list[str] = []
    source_files: list[str] = []
    row_indices: list[int] = []
    label_roles: list[str] = []
    all_event_ids: set[str] = set()
    all_policy_ids: set[str] = set()
    shard_files: list[str] = []
    total_samples = 0
    failures = []

    def flush_shard() -> None:
        nonlocal output, event_ids, policy_ids, source_files, row_indices, label_roles, total_samples
        if not event_ids:
            return
        payload = {
            "state": np.asarray(output["state"], dtype=np.float32),
            "candidate_action_seq": np.asarray(output["candidate_action_seq"], dtype=np.float32),
            "rain_seq": np.asarray(output["rain_seq"], dtype=np.float32),
            "risk_rate_seq": np.asarray(output["risk_rate_seq"], dtype=np.float32),
            "local_state_seq": np.asarray(output["local_state_seq"], dtype=np.float32),
            "event_ids": np.asarray(event_ids, dtype=object),
            "policy_ids": np.asarray(policy_ids, dtype=object),
            "source_files": np.asarray(source_files, dtype=object),
            "row_indices": np.asarray(row_indices, dtype=np.int64),
            "label_roles": np.asarray(label_roles, dtype=object),
        }
        if mode == "full_state":
            payload["target_state_seq"] = np.asarray(output["target_state_seq"], dtype=np.float32)
        shard_path = shard_dir / f"chunk_{len(shard_files):05d}.npz"
        np.savez_compressed(shard_path, **payload)
        shard_files.append(str(shard_path.resolve()))
        total_samples += len(event_ids)
        output = {key: [] for key in ["state", "candidate_action_seq", "rain_seq", "risk_rate_seq", "local_state_seq"]}
        if mode == "full_state":
            output["target_state_seq"] = []
        event_ids = []
        policy_ids = []
        source_files = []
        row_indices = []
        label_roles = []

    for _, row in manifest.iterrows():
        path = Path(str(row["detail_file"]))
        try:
            cols = _existing_columns(path)
            flood_cols = [f"flood:{node}" for node in node_ids]
            needed = set(node_cols + action_cols + flood_cols + ["rainfall_mm_h"])
            detail = _read_detail(path, usecols=[c for c in cols if c in needed])
            if len(detail) <= H:
                continue
            states = _numeric_matrix(detail, node_cols)
            local_states = _numeric_matrix(detail, local_cols) if local_cols else np.zeros((len(detail), 0), dtype=np.float32)
            actions = _numeric_matrix(detail, action_cols, fill=1.0)
            flood = _numeric_matrix(detail, flood_cols)
            priority_indices = [i for i, node in enumerate(node_ids) if node in priority]
            if priority_indices:
                pfv_rate = flood[:, priority_indices].sum(axis=1)
            else:
                pfv_rate = np.zeros(len(detail), dtype=np.float32)
            tfv_rate = flood.sum(axis=1)
            rain = (
                pd.to_numeric(detail["rainfall_mm_h"], errors="coerce").fillna(0.0).to_numpy(np.float32)[:, None]
                if "rainfall_mm_h" in detail
                else np.zeros((len(detail), 1), dtype=np.float32)
            )
            event_id, policy_id = _event_policy(row, path)
            for i in range(0, len(detail) - H, stride):
                output["state"].append(states[i])
                output["candidate_action_seq"].append(actions[i : i + H])
                output["rain_seq"].append(rain[i : i + H])
                pfv_window = pfv_rate[i + 1 : i + H + 1]
                tfv_window = tfv_rate[i + 1 : i + H + 1]
                output["risk_rate_seq"].append(
                    np.stack(
                        [pfv_window, tfv_window, np.maximum.accumulate(tfv_window)],
                        axis=1,
                    ).astype(np.float32)
                )
                output["local_state_seq"].append(local_states[i + 1 : i + H + 1])
                if mode == "full_state":
                    output["target_state_seq"].append(states[i + 1 : i + H + 1])
                event_ids.append(event_id)
                policy_ids.append(policy_id)
                all_event_ids.add(event_id)
                all_policy_ids.add(policy_id)
                source_files.append(path.name)
                row_indices.append(i)
                label_roles.append(str(row.get("effect_label_role", "observational_dynamics_pretraining")))
                if sharded and len(event_ids) >= chunk_size:
                    flush_shard()
                if max_samples and total_samples + len(event_ids) >= int(max_samples):
                    break
        except Exception as exc:
            failures.append({"detail_file": str(path), "error": repr(exc)})
        if max_samples and total_samples + len(event_ids) >= int(max_samples):
            break
    if sharded:
        flush_shard()
    sample_count = total_samples if sharded else len(event_ids)
    if sample_count == 0:
        raise RuntimeError("No temporal action pretraining samples were built from the manifest.")
    if sharded:
        np.savez_compressed(
            out_npz,
            shard_files=np.asarray(shard_files, dtype=object),
            sample_count=np.asarray([sample_count], dtype=np.int64),
            node_cols=np.asarray(node_cols, dtype=object),
            local_node_cols=np.asarray(local_cols, dtype=object),
            action_ids=np.asarray(action_ids, dtype=object),
        )
    else:
        payload = {
            "state": np.asarray(output["state"], dtype=np.float32),
            "candidate_action_seq": np.asarray(output["candidate_action_seq"], dtype=np.float32),
            "rain_seq": np.asarray(output["rain_seq"], dtype=np.float32),
            "risk_rate_seq": np.asarray(output["risk_rate_seq"], dtype=np.float32),
            "local_state_seq": np.asarray(output["local_state_seq"], dtype=np.float32),
            "node_cols": np.asarray(node_cols, dtype=object),
            "local_node_cols": np.asarray(local_cols, dtype=object),
            "action_ids": np.asarray(action_ids, dtype=object),
            "event_ids": np.asarray(event_ids, dtype=object),
            "policy_ids": np.asarray(policy_ids, dtype=object),
            "source_files": np.asarray(source_files, dtype=object),
            "row_indices": np.asarray(row_indices, dtype=np.int64),
            "label_roles": np.asarray(label_roles, dtype=object),
        }
        if mode == "full_state":
            payload["target_state_seq"] = np.asarray(output["target_state_seq"], dtype=np.float32)
        np.savez_compressed(out_npz, **payload)
    report = {
        "out_npz": str(out_npz),
        "manifest": str(manifest_path),
        "files_requested": int(len(manifest)),
        "samples": int(sample_count),
        "events": int(len(all_event_ids or set(event_ids))),
        "policies": sorted(all_policy_ids or set(policy_ids)),
        "nodes": int(len(node_cols)),
        "actions": int(len(action_ids)),
        "action_tensor_shape": f"[{H},{len(action_ids)}]",
        "target_mode": mode,
        "local_nodes": int(len(local_cols)),
        "horizon_steps": int(H),
        "time_stride": int(stride),
        "storage_mode": "sharded_npz" if sharded else "single_npz",
        "chunk_size_samples": int(chunk_size),
        "shard_count": int(len(shard_files)),
        "shard_dir": str(shard_dir) if sharded else "",
        "label_semantics": "observational temporal dynamics pretraining; not same-state causal effect unless label_roles say so",
        "risk_label_channels": list(RISK_LABEL_CHANNELS),
        "peak_label_definition": "running maximum of TFV_rate within each prediction horizon",
        "failures": failures[:50],
        "failure_count": int(len(failures)),
    }
    out_npz.with_suffix(".meta.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_mpc_readiness(*, config: dict, model_report_path: str | Path) -> dict:
    controller = config.get("controller", {}) or {}
    temporal = controller.get("temporal_joint", {}) or {}
    safety = temporal.get("safety", {}) or {}
    hierarchical = temporal.get("hierarchical", {}) or {}
    hierarchical_enabled = bool(hierarchical.get("enabled", False))
    reasons = []
    residual_reasons = []
    controller_mode = str(controller.get("mode", "")).lower()
    supported_modes = {"temporal_joint_36", "hierarchical_core26_residual10"}
    if controller_mode not in supported_modes:
        reasons.append("controller_mode_not_temporal_joint_36")
    if bool((controller.get("phase_reliability", {}) or {}).get("require_pfv_improvement", False)):
        reasons.append("pfv_improvement_required_instead_of_noninferiority")
    if str(controller.get("reference_policy_for_constraints", "")).lower() != "online_predicted_default":
        reasons.append("reference_policy_not_online_predicted_default")
    if float(safety.get("pfv_abs_margin_m3", 0.0) or 0.0) <= 0.0:
        reasons.append("missing_pfv_noninferiority_margin")
    if hierarchical_enabled:
        legacy_model = Path(str(hierarchical.get("legacy_model_path", "") or ""))
        if not legacy_model.is_absolute():
            legacy_model = Path(config.get("project_root", ".")) / legacy_model
        if not legacy_model.exists():
            reasons.append("missing_legacy_v8_model")
        if not temporal.get("legacy_groups"):
            reasons.append("missing_legacy_v8_groups")
        if not hierarchical.get("residual_actuator_ids"):
            residual_reasons.append("missing_residual_actuator_ids")
    report_path = Path(model_report_path)
    residual_model_gate_passed = False
    residual_smoke_eligibility_passed = False
    if not report_path.exists():
        reasons.append("missing_model_report")
        report = {}
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        residual_model_gate_passed = bool(report.get("validation_gate_passed", False))
        if not residual_model_gate_passed:
            (residual_reasons if hierarchical_enabled else reasons).append("model_validation_gate_false")
        smoke_gate = report.get("rolling_horizon_smoke_eligibility")
        if not isinstance(smoke_gate, dict):
            (residual_reasons if hierarchical_enabled else reasons).append("missing_effect_direction_gate")
        else:
            residual_smoke_eligibility_passed = bool(smoke_gate.get("passed", False))
            if not residual_smoke_eligibility_passed:
                (residual_reasons if hierarchical_enabled else reasons).append("effect_direction_gate_false")
        configured_model = str(temporal.get("model_path", "") or "").replace("\\", "/").lower()
        reported_model = str(report.get("model", "") or "").replace("\\", "/").lower()
        if configured_model and reported_model and not reported_model.endswith(configured_model):
            (residual_reasons if hierarchical_enabled else reasons).append("configured_model_does_not_match_validated_report")
    tier2_allowed = bool(hierarchical_enabled and not residual_reasons)
    return {
        "passed": not reasons,
        "closed_loop_allowed": not reasons,
        "blocking_reasons": reasons,
        "residual_blocking_reasons": residual_reasons,
        "residual_model_gate_passed": residual_model_gate_passed,
        "residual_smoke_eligibility_passed": residual_smoke_eligibility_passed,
        "tier2_residual_allowed": tier2_allowed,
        "deployment_mode": "tier2_residual" if tier2_allowed else ("tier1_only" if hierarchical_enabled and not reasons else "blocked"),
        "model_report": str(report_path),
        "validation_gate_failures": report.get("validation_gate_failures", []),
        "rolling_horizon_smoke_eligibility": report.get("rolling_horizon_smoke_eligibility"),
        "objective_contract": "PFV non-inferiority vs online no-control, peak non-worsening, maximize TFV reduction among safe candidates",
    }


def _comparison_by_policy(gate: dict, policy: str) -> dict:
    for row in gate.get("baseline_comparisons", []) or []:
        if str(row.get("baseline_policy", "")) == str(policy):
            return dict(row)
    return {}


def _interpret_gate(gate: dict) -> str:
    no_control = _comparison_by_policy(gate, "no_control")
    if not no_control:
        return "missing_no_control_comparison"
    pfv = float(no_control.get("PFV_mean_reduction_pct", 0.0) or 0.0)
    tfv = float(no_control.get("TFV_mean_reduction_pct", 0.0) or 0.0)
    peak = float(no_control.get("peak_mean_reduction_pct", 0.0) or 0.0)
    worse = float(no_control.get("PFV_worse_frac_noninferiority", 0.0) or 0.0)
    if pfv < -10.0 or tfv < -10.0 or peak < -10.0 or worse > 0.5:
        return "systemic_failure_vs_no_control"
    if tfv > 3.0 and peak >= 0.0 and not bool(gate.get("passed", False)):
        return "system_repair_success_but_strict_pfv_noninferiority_failed"
    if bool(gate.get("passed", False)):
        return "strict_gate_passed"
    return "gate_failed_mixed_or_marginal"


def summarize_gate_comparison(gate26: dict, gate36: dict) -> dict:
    out = {}
    for label, gate in [("v8_26", gate26), ("temporal_36", gate36)]:
        no_control = _comparison_by_policy(gate, "no_control")
        internal = _comparison_by_policy(gate, "internal_rules")
        out[label] = {
            "passed": bool(gate.get("passed", False)),
            "interpretation": _interpret_gate(gate),
            "reasons": gate.get("reasons", []),
            "vs_no_control": {
                "PFV_mean_reduction_pct": no_control.get("PFV_mean_reduction_pct"),
                "TFV_mean_reduction_pct": no_control.get("TFV_mean_reduction_pct"),
                "peak_mean_reduction_pct": no_control.get("peak_mean_reduction_pct"),
                "PFV_worse_frac_noninferiority": no_control.get("PFV_worse_frac_noninferiority"),
            },
            "vs_internal_rules": {
                "PFV_mean_reduction_pct": internal.get("PFV_mean_reduction_pct"),
                "TFV_mean_reduction_pct": internal.get("TFV_mean_reduction_pct"),
                "peak_mean_reduction_pct": internal.get("peak_mean_reduction_pct"),
            },
        }
    out["main_explanation"] = (
        "The 26-asset v8 line used exact local phase reliability plus a horizon surrogate and achieved strong TFV/peak "
        "repair, but it still missed the strict no-control PFV non-inferiority gate. The current 36-asset temporal line "
        "fails more fundamentally because candidate effects are not yet reliable enough and accepted actions cause large "
        "PFV/TFV/peak deterioration versus no-control."
    )
    return out
