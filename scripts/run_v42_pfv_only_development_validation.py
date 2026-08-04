"""Run a small authoritative PFV-only development validation.

Uses only the already revealed Calibration inputs and an explicitly marked
development calibration. Results are diagnostic and are never written to
Formal paper-execution or Policy-Lock evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plan_v42_pfv_only_fresh_calibration import _forcing
from sewerrtc.v4.v42_formal_runtime import FormalEventInput, run_baseline_event
from sewerrtc.v4.v42_formal_runtime_safe import run_proposed_event


def _choose_events(root: Path, events: list) -> list:
    inventory = pd.read_csv(root / "outputs/project6_dual_reference_v4/final_v4/inventory/event_inventory.csv")
    paths = dict(zip(inventory["event_id"].astype(str), inventory["rainfall_path"].astype(str)))
    rows = []
    for event in events:
        path = Path(paths.get(event.event_id, ""))
        if not path.exists():
            raise FileNotFoundError(f"revealed Calibration rainfall forcing missing: {event.event_id}: {path}")
        rows.append({"event": event, **_forcing(path)})
    rows.sort(key=lambda x: (float(x["total_depth_mm"]), float(x["peak_intensity_mm_h"]), x["event"].event_id))
    return [rows[i]["event"] for i in (0, len(rows) // 2, len(rows) - 1)]


def _load_revealed_events(root: Path) -> list[FormalEventInput]:
    path = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/evaluation_inputs/calibration_case_manifest.csv"
    frame = pd.read_csv(path, low_memory=False)
    result = []
    for (event_id, rainfall_sha), group in frame.groupby(["event_id", "rainfall_sha256"], sort=True):
        if group["inp_path"].astype(str).nunique() != 1:
            raise RuntimeError(f"revealed Calibration event has inconsistent INP paths: {event_id}")
        row = group.iloc[0]
        inp = Path(str(row["inp_path"]))
        if not inp.exists():
            raise FileNotFoundError(inp)
        result.append(
            FormalEventInput(
                role="calibration",
                event_id=str(event_id),
                rainfall_sha256=str(rainfall_sha),
                inp_path=inp.resolve(),
                rain_duration_min=int(row["rain_duration_min"]),
                simulation_duration_min=int(row["simulation_duration_min"]),
            )
        )
    if len(result) < 3:
        raise RuntimeError(f"revealed Calibration contains fewer than 3 unique events: {len(result)}")
    return result


def _kmax(detail_path: Path) -> int:
    detail = pd.read_csv(detail_path, low_memory=False)
    action = [c for c in detail.columns if c.startswith("a:")]
    if not action:
        return 0
    values = detail[action].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if len(values) < 2:
        return 0
    changed = np.abs(np.diff(values, axis=0)) > 1.0e-6
    return int(changed.sum(axis=1).max()) if len(changed) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    root_default = Path(__file__).resolve().parents[1]
    ap.add_argument("--project-root", type=Path, default=root_default)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--max-candidate-sequences", type=int, default=64)
    ap.add_argument(
        "--development-calibration",
        type=Path,
        default=root_default / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/diagnostics/PFV_ONLY_DEVELOPMENT_CALIBRATION.json",
    )
    args = ap.parse_args()
    cal = json.loads(args.development_calibration.read_text(encoding="utf-8"))
    if cal.get("development_only") is not True or cal.get("formal_mainline_authorized") is True:
        raise RuntimeError("development validation requires an explicitly development-only PFV calibration")
    events = _choose_events(args.project_root, _load_revealed_events(args.project_root))
    out_root = args.project_root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/diagnostics/PFV_ONLY_DEVELOPMENT_SWMM"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for pos, event in enumerate(events, start=1):
        print(json.dumps({"stage": "pfv_only_development_swmm", "event": event.event_id, "position": pos, "total": len(events)}, ensure_ascii=False), flush=True)
        event_dir = out_root / event.event_id
        proposed = run_proposed_event(
            event,
            project_root=args.project_root,
            output_dir=event_dir / "Proposed",
            state_source="true_state",
            device=args.device,
            max_candidate_sequences=args.max_candidate_sequences,
            step2_calibration_path=args.development_calibration,
        )
        baselines = {
            strategy: run_baseline_event(
                event,
                strategy=strategy,
                project_root=args.project_root,
                output_dir=event_dir / strategy,
            )
            for strategy in ("No-control", "Internal", "Hold")
        }
        decision_path = Path(str(proposed["decision_path"]))
        decisions = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = [x for x in decisions if str(x.get("selected_id")) != "frozen_hold_readback"]
        proposed_kpi = proposed["kpis"]
        no_control_kpi = baselines["No-control"]["kpis"]
        internal_kpi = baselines["Internal"]["kpis"]
        rows.append(
            {
                "event": event.event_id,
                "rainfall_sha256": event.rainfall_sha256,
                "decision_count": len(decisions),
                "non_hold_decisions": len(selected),
                "fallback_rate": float(proposed.get("fallback_rate", 1.0)),
                "PFV_Proposed": float(proposed_kpi["PFV"]),
                "PFV_No_control": float(no_control_kpi["PFV"]),
                "PFV_allowance": 100.0 + 0.05 * float(no_control_kpi["PFV"]),
                "PFV_pass": float(proposed_kpi["PFV"] - no_control_kpi["PFV"]) <= 100.0 + 0.05 * float(no_control_kpi["PFV"]),
                "TFV_Proposed": float(proposed_kpi["TFV"]),
                "TFV_Internal": float(internal_kpi["TFV"]),
                "TFV_delta_vs_Internal": float(proposed_kpi["TFV"] - internal_kpi["TFV"]),
                "Peak_Proposed": float(proposed_kpi["peak_TFV_rate"]),
                "Peak_Internal": float(internal_kpi["peak_TFV_rate"]),
                "Peak_delta_vs_Internal": float(proposed_kpi["peak_TFV_rate"] - internal_kpi["peak_TFV_rate"]),
                "action_changes": float(proposed_kpi.get("action_changes", 0.0)),
                "K_max": _kmax(Path(str(proposed["detail_path"]))),
                "engineering_violations": int(sum(not bool(x.get("target_write_verified", False)) for x in decisions)),
                "authority": "authoritative_SWMM_development_only",
            }
        )
        (out_root / "PFV_ONLY_DEVELOPMENT_SWMM_PROGRESS.json").write_text(json.dumps({"completed_events": pos, "total_events": len(events), "last_event": event.event_id}, indent=2), encoding="utf-8")
    table = pd.DataFrame(rows)
    csv_path = out_root / "DEVELOPMENT_PFV_ONLY_TFV_MIN_VALIDATION.csv"
    json_path = out_root / "DEVELOPMENT_PFV_ONLY_TFV_MIN_VALIDATION.json"
    table.to_csv(csv_path, index=False)
    summary = {
        "status": "development_only_pass",
        "formal_mainline_authorized": False,
        "events": table["event"].tolist(),
        "non_hold_decisions": int(table["non_hold_decisions"].sum()),
        "fallback_rate": float(table["fallback_rate"].mean()),
        "pfv_pass_all_events": bool(table["PFV_pass"].all()),
        "engineering_violations": int(table["engineering_violations"].sum()),
        "output_csv": str(csv_path),
        "development_calibration": str(args.development_calibration),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
