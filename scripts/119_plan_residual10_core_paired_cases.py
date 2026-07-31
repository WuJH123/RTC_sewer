from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.hierarchical_core26_residual10 import (
    assert_residual_only_changes_residual_columns,
    core_residual_ids,
)
from sewerrtc.experiments.targeted_joint_pairs import (
    action_window,
    event_pattern,
    event_return_period,
    materialize_candidate,
    sequence_diagnostics,
)
from sewerrtc.experiments.tier2_residual_v8 import (
    BINARY_RESIDUAL_PUMPS_V8,
    RESIDUAL_ACTUATORS_V8,
    binary_toggle_profile,
    phase_start_min,
    stable_hash,
    temporal_delta_profile,
)
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _case_hash(payload: Any) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:20]


def _normalise_rainfall_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_id"] = out["event_id"].astype(str)
    if "rain_id" not in out:
        out["rain_id"] = out["event_id"].str.split("_", n=1).str[0]
    if "pattern" not in out:
        out["pattern"] = out["event_id"].map(event_pattern)
    if "duration_min" not in out:
        out["duration_min"] = out["event_id"].str.extract(r"_D([0-9]+)_", expand=False).astype(float)
    return out


def _find_no_control_details(root: Path, event_ids: set[str]) -> dict[str, Path]:
    best: dict[str, Path] = {}
    for path in root.rglob("*__no_control_detail.csv"):
        event_id = path.name.split("__", 1)[0]
        if event_id not in event_ids:
            continue
        old = best.get(event_id)
        if old is None or path.stat().st_mtime > old.stat().st_mtime:
            best[event_id] = path
    return best


def _profile_values(profile: str, magnitude: float, direction: float, *, horizon: int) -> list[float]:
    value = float(magnitude) * float(direction)
    out = np.zeros(int(horizon), dtype=np.float32)
    if profile == "pulse":
        out[0 : max(1, min(3, horizon))] = value
    elif profile == "early_then_restore":
        out[0 : max(1, min(4, horizon))] = value
        if horizon > 4:
            out[4:] = 0.0
    elif profile == "late":
        out[max(1, horizon // 2) :] = value
    else:
        out[:] = value
    return out.astype(float).tolist()


def _core_specs(template_path: Path, *, horizon: int, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    for template in payload.get("templates", [])[: int(limit)]:
        actuators = [str(value) for value in template.get("actuators", [])]
        magnitude = float(template.get("magnitude", 0.08))
        direction = float(template.get("direction", -1.0))
        profile = str(template.get("profile", "hold"))
        signed = {
            actuator: _profile_values(profile, magnitude, direction, horizon=horizon)
            for actuator in actuators
        }
        specs.append(
            {
                "family": "core26_v8_exact",
                "kind": "core26_template",
                "mode": str(template.get("label", f"core_template_{len(specs)}")),
                "core_template_id": str(template.get("label", f"core_template_{len(specs)}")),
                "actuators": actuators,
                "signed_profiles": signed,
                "target_profiles": {},
                "horizon_steps": int(horizon),
                "tier": 1,
            }
        )
    if not specs:
        raise ValueError(f"no templates in {template_path}")
    return specs


def _residual_specs(reference_core_seq: np.ndarray, action_ids: list[str], phase: str, *, horizon: int) -> list[dict[str, Any]]:
    idx = {actuator: position for position, actuator in enumerate(action_ids)}
    specs: list[dict[str, Any]] = []

    def append(mode: str, signed: dict[str, list[float]] | None = None, targets: dict[str, list[float]] | None = None, role: str = "fit") -> None:
        residual_ids = sorted(set((signed or {}).keys()) | set((targets or {}).keys()))
        specs.append(
            {
                "family": "residual10_core_conditioned",
                "kind": "residual10_delta",
                "mode": mode,
                "residual_actuators": residual_ids,
                "actuators": residual_ids,
                "signed_profiles": signed or {},
                "target_profiles": targets or {},
                "horizon_steps": int(horizon),
                "intended_evidence_role": role,
                "tier": 2,
            }
        )

    for actuator in RESIDUAL_ACTUATORS_V8[:8]:
        if actuator not in idx:
            continue
        setting = float(np.median(reference_core_seq[:, idx[actuator]]))
        directions = [-1.0, 1.0]
        if setting <= 0.03:
            directions = [1.0]
        elif setting >= 0.97:
            directions = [-1.0]
        for magnitude in (0.05, 0.10, 0.20, 0.35):
            for direction in directions:
                role = "offline_safety_boundary" if magnitude >= 0.35 else "fit_deployment"
                for variant in ("hold", "delayed", "late"):
                    delta = float(direction) * float(magnitude)
                    append(
                        f"{actuator}_{delta:+.2f}_{phase}_{variant}",
                        signed={actuator: temporal_delta_profile(delta, phase, horizon_steps=horizon, variant=variant)},
                        role=role,
                    )
    for pump in BINARY_RESIDUAL_PUMPS_V8:
        if pump not in idx:
            continue
        append(
            f"{pump}_binary_toggle_{phase}",
            targets={pump: binary_toggle_profile(reference_core_seq[:, idx[pump]], phase)},
            role="fit_deployment",
        )
    for inlet, outlet in (("RTC_IN_01", "RTC_OUT_01"), ("RTC_IN_02", "RTC_OUT_02"), ("RTC_IN_03", "RTC_OUT_03")):
        if inlet in idx and outlet in idx:
            append(
                f"{inlet}_{outlet}_interlock_retain_release_{phase}",
                signed={
                    inlet: temporal_delta_profile(-0.10, phase, horizon_steps=horizon, variant="hold"),
                    outlet: temporal_delta_profile(0.10, phase, horizon_steps=horizon, variant="late"),
                },
                role="fit_deployment",
            )
            append(
                f"{inlet}_{outlet}_unsafe_same_direction_{phase}",
                signed={
                    inlet: temporal_delta_profile(0.35, phase, horizon_steps=horizon, variant="hold"),
                    outlet: temporal_delta_profile(0.35, phase, horizon_steps=horizon, variant="hold"),
                },
                role="offline_safety_boundary",
            )
    return specs


def _select_events(rain: pd.DataFrame, details: dict[str, Path], *, max_events: int, seed: int) -> pd.DataFrame:
    frame = rain[rain["event_id"].isin(details)].copy()
    if frame.empty:
        raise ValueError("no rainfall events have reusable no_control details")
    severe = {"T50", "T75", "T100"}
    frame["sort_key"] = frame.apply(
        lambda r: (
            0 if str(r["rain_id"]) in severe else 1,
            str(r["pattern"]),
            stable_hash(str(r["event_id"]), seed=seed),
        ),
        axis=1,
    )
    selected = []
    counts: dict[tuple[str, str], int] = {}
    for _, row in frame.sort_values("sort_key").iterrows():
        key = (str(row["rain_id"]), str(row["pattern"]))
        if counts.get(key, 0) >= 2:
            continue
        selected.append(row)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= int(max_events):
            break
    out = pd.DataFrame(selected).drop(columns=["sort_key"], errors="ignore").reset_index(drop=True)
    roles = []
    for i in range(len(out)):
        roles.append("locked_validation" if i % 5 == 0 else "fit")
    out["event_role"] = roles
    out["split"] = np.where(out["event_role"].eq("locked_validation"), "validation", "train")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan same-state Core26-vs-Residual10 paired cases.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_hierarchical_eventbudget_h120_v2.yaml")
    parser.add_argument("--out-dir", default="outputs/project6_36_residual10_core_paired_h120_v1/paired_plan")
    parser.add_argument("--no-control-root", default="outputs/closed_loop_paired_no_controls")
    parser.add_argument("--max-events", type=int, default=48)
    parser.add_argument("--max-logical-pairs", type=int, default=720)
    parser.add_argument("--core-template-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    rain = _normalise_rainfall_table(pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"))
    details = _find_no_control_details(root / args.no_control_root, set(rain["event_id"].astype(str)))
    selected = _select_events(rain, details, max_events=args.max_events, seed=args.seed)
    actuator_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    enabled_path = cfg_path(cfg, "network.control_enabled_actuator_ids_file")
    enabled = [line.strip() for line in enabled_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    actuators = actuator_table[actuator_table["actuator_id"].astype(str).isin(set(enabled))].copy()
    action_ids = actuators["actuator_id"].astype(str).tolist()
    residual10_config = (
        (cfg.get("controller", {}) or {})
        .get("temporal_joint", {})
        .get("hierarchical", {})
        .get("residual_actuator_ids", RESIDUAL_ACTUATORS_V8)
    )
    core26_ids, residual10_ids = core_residual_ids(actuators, residual10_config)
    priority_nodes = [line.strip() for line in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    horizon = 6
    binary_pumps = set(BINARY_RESIDUAL_PUMPS_V8)
    template_path = root / cfg["controller"]["temporal_joint"]["hierarchical"]["core26_policy_path"]
    core_specs = _core_specs(template_path, horizon=horizon, limit=args.core_template_limit)
    rows: list[dict[str, Any]] = []
    bucket_target = max(2, int(np.ceil(float(args.max_logical_pairs) / max(1, len(selected) * 3))))
    for event_row in selected.itertuples(index=False):
        reference_detail = details[str(event_row.event_id)]
        reference_frame = pd.read_csv(reference_detail)
        for phase in ("rising", "peak", "recession"):
            bucket_added = 0
            start_min = phase_start_min(float(event_row.duration_min), phase)
            try:
                no_control_seq = action_window(reference_frame, action_ids=action_ids, start_min=start_min, horizon_steps=horizon)
            except Exception:
                continue
            for core_spec in core_specs:
                try:
                    core_seq = materialize_candidate(no_control_seq, action_ids=action_ids, specification=core_spec)
                except Exception:
                    continue
                core_diag = sequence_diagnostics(core_seq, no_control_seq, action_ids=action_ids, binary_pump_ids=binary_pumps, minimum_effective_delta=0.01)
                if not core_diag["valid"]:
                    continue
                residual_specs = _residual_specs(core_seq, action_ids, phase, horizon=horizon)
                ordered_specs = sorted(residual_specs, key=lambda spec: stable_hash([event_row.event_id, phase, core_spec["mode"], spec["mode"]], seed=args.seed))
                for residual_spec in ordered_specs[: max(4, min(20, len(ordered_specs)))]:
                    try:
                        candidate_seq = materialize_candidate(core_seq, action_ids=action_ids, specification=residual_spec)
                        assert_residual_only_changes_residual_columns(
                            core_seq,
                            candidate_seq,
                            canonical_action_ids=action_ids,
                            residual_actuator_ids=residual10_ids,
                        )
                    except Exception:
                        continue
                    diag = sequence_diagnostics(candidate_seq, core_seq, action_ids=action_ids, binary_pump_ids=binary_pumps, minimum_effective_delta=0.02)
                    if not diag["valid"]:
                        continue
                    pair_payload = [event_row.event_id, phase, start_min, core_spec["mode"], residual_spec["mode"]]
                    pair_id = _case_hash(pair_payload)
                    base = {
                        "pair_id": pair_id,
                        "event_id": str(event_row.event_id),
                        "rain_id": str(event_row.rain_id),
                        "rain_pattern": str(event_row.pattern),
                        "duration_min": float(event_row.duration_min),
                        "event_role": str(event_row.event_role),
                        "split": str(event_row.split),
                        "phase": phase,
                        "checkpoint_id": f"{event_row.event_id}|{phase}|{float(start_min):.3f}",
                        "override_start_min": float(start_min),
                        "horizon_steps": horizon,
                        "reference_detail": str(reference_detail),
                        "core_template_id": str(core_spec["core_template_id"]),
                        "residual_mode": str(residual_spec["mode"]),
                        "residual_actuator_ids": ",".join(diag["changed_actuator_ids"]),
                        "changed_actuator_count": int(diag["changed_actuator_count"]),
                        "action_l1_difference": float(diag["action_l1_difference"]),
                        "action_linf_difference": float(diag["action_linf_difference"]),
                        "intended_evidence_role": str(residual_spec.get("intended_evidence_role", "fit")),
                        "materialized_core26_action_sequence": _json(core_seq.astype(float).tolist()),
                        "materialized_candidate_action_sequence": _json(candidate_seq.astype(float).tolist()),
                        "residual_specification": _json(residual_spec),
                        "core_specification": _json(core_spec),
                    }
                    rows.append({**base, "branch": "A", "case_id": f"{pair_id}__core26", "executed_action_sequence": _json(core_spec)})
                    rows.append({**base, "branch": "B", "case_id": f"{pair_id}__residual10", "executed_action_sequence": _json({**core_spec, "family": "core26_plus_residual10", "kind": "core26_plus_residual10", "mode": f"{core_spec['mode']}__{residual_spec['mode']}", "signed_profiles": {**core_spec.get("signed_profiles", {}), **residual_spec.get("signed_profiles", {})}, "target_profiles": {**core_spec.get("target_profiles", {}), **residual_spec.get("target_profiles", {})}, "residual_actuators": residual_spec.get("residual_actuators", [])})})
                    bucket_added += 1
                    if bucket_added >= bucket_target:
                        break
                if bucket_added >= bucket_target:
                    break
    if len(rows) // 2 > int(args.max_logical_pairs):
        pair_order = (
            pd.DataFrame(rows)
            .query("branch == 'B'")
            .assign(_rank=lambda frame: frame.apply(lambda r: stable_hash([r["event_id"], r["phase"], r["residual_mode"]], seed=args.seed), axis=1))
            .sort_values(["event_id", "phase", "_rank"])["pair_id"]
            .drop_duplicates()
            .head(int(args.max_logical_pairs))
            .tolist()
        )
        keep = set(pair_order)
        rows = [row for row in rows if row["pair_id"] in keep]
    manifest = pd.DataFrame(rows)
    manifest_path = out / "residual10_core_paired_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    branch_b = manifest[manifest["branch"].eq("B")].copy()
    report = {
        "manifest": str(manifest_path),
        "logical_pairs": int(branch_b["pair_id"].nunique()) if len(branch_b) else 0,
        "physical_cases": int(len(manifest)),
        "events": int(branch_b["event_id"].nunique()) if len(branch_b) else 0,
        "core26_count": int(len(core26_ids)),
        "residual10_count": int(len(residual10_ids)),
        "residual10_ids": list(residual10_ids),
        "branch_b_only_changes_residual10": True,
        "event_roles": branch_b["event_role"].value_counts().astype(int).to_dict() if len(branch_b) else {},
        "phases": branch_b["phase"].value_counts().astype(int).to_dict() if len(branch_b) else {},
        "evidence_roles": branch_b["intended_evidence_role"].value_counts().astype(int).to_dict() if len(branch_b) else {},
    }
    (out / "targeted_manifest_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
