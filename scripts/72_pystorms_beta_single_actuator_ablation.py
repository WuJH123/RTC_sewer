from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.pyswmm_runner import run_swmm_trajectory


def _event_table(cfg: dict, event_ids: list[str], max_events: int) -> pd.DataFrame:
    rain = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    if event_ids:
        order = {event_id: i for i, event_id in enumerate(event_ids)}
        rain = rain[rain["event_id"].astype(str).isin(order)].copy()
        rain["_order"] = rain["event_id"].astype(str).map(order)
        rain = rain.sort_values("_order").drop(columns=["_order"])
    elif (cfg.get("rainfall", {}) or {}).get("representative_event_ids"):
        ids = [str(x) for x in cfg["rainfall"]["representative_event_ids"]]
        order = {event_id: i for i, event_id in enumerate(ids)}
        rain = rain[rain["event_id"].astype(str).isin(order)].copy()
        rain["_order"] = rain["event_id"].astype(str).map(order)
        rain = rain.sort_values("_order").drop(columns=["_order"])
    if max_events > 0:
        rain = rain.head(max_events)
    if rain.empty:
        raise ValueError("No rainfall events selected for single-actuator ablation.")
    return rain


def _fixed_action_from_columns(actuators: pd.DataFrame, policy_col: str = "no_control_setting") -> dict[str, float]:
    out: dict[str, float] = {}
    ids = actuators["actuator_id"].astype(str).tolist()
    if policy_col in actuators:
        vals = pd.to_numeric(actuators[policy_col], errors="coerce")
        for aid, val in zip(ids, vals):
            if pd.notna(val):
                out[aid] = float(val)
    if not out:
        out = {aid: 1.0 for aid in ids}
    return out


def _with_fixed_policy(actuators: pd.DataFrame, policy_id: str, settings: dict[str, float]) -> pd.DataFrame:
    out = actuators.copy()
    col = f"{policy_id}_setting"
    out[col] = pd.NA
    for i, aid in out["actuator_id"].astype(str).items():
        if aid in settings:
            out.loc[i, col] = float(settings[aid])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/open_pystorms_beta.yaml")
    ap.add_argument("--event-ids", default="")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--settings", default="0,1", help="Comma-separated settings to test for each actuator.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(
        Path(args.out_dir)
        if args.out_dir
        else cfg_path(cfg, "outputs.diagnostics") / "single_actuator_ablation"
    )
    inp_dir = ensure_dir(out_dir / "event_inp")
    detail_dir = ensure_dir(out_dir / "details")
    event_ids = [x.strip() for x in args.event_ids.split(",") if x.strip()]
    rain = _event_table(cfg, event_ids, int(args.max_events))
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    priority = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    base_settings = _fixed_action_from_columns(actuators, "no_control_setting")
    settings_to_test = [float(x.strip()) for x in args.settings.split(",") if x.strip()]

    rows: list[dict] = []
    for _, ev in rain.iterrows():
        event_id = str(ev["event_id"])
        event_inp = inp_dir / f"{event_id}__no_controls.inp"
        if not event_inp.exists():
            mutate_inp_for_event(
                cfg_path(cfg, "network.inp"),
                ev["rainfall_csv"],
                event_inp,
                int(ev["simulation_duration_min"]),
                strip_controls=True,
            )
        base_detail = detail_dir / f"{event_id}__ablation_base_no_control_detail.csv"
        if args.resume and base_detail.exists():
            base_row = None
        else:
            base_act = _with_fixed_policy(actuators, "ablation_base_no_control", base_settings)
            base_row = run_swmm_trajectory(
                event_inp,
                "ablation_base_no_control",
                base_act,
                priority,
                base_detail,
                event_id,
                int(ev["duration_min"]),
                int(cfg["experiment"]["control_step_sec"]),
                int(cfg["experiment"]["random_seed"]),
                simulation_duration_min=int(ev["simulation_duration_min"]),
                recession_min=int(cfg["experiment"]["recession_min"]),
            )
            rows.append({**base_row, "actuator_id": "", "tested_setting": "", "base_policy": "no_control"})
        if base_row is None:
            base_df = pd.read_csv(base_detail)
            from sewerrtc.simulation.kpi_metrics import compute_kpis

            base_row = compute_kpis(base_df, priority, dt_sec=int(cfg["experiment"]["control_step_sec"]))
        for aid in actuators["actuator_id"].astype(str):
            for setting in settings_to_test:
                if abs(float(base_settings.get(aid, 1.0)) - float(setting)) < 1e-9:
                    continue
                policy_id = f"ablate_{aid}_{str(setting).replace('.', 'p')}"
                detail = detail_dir / f"{event_id}__{policy_id}_detail.csv"
                if args.resume and detail.exists():
                    from sewerrtc.simulation.kpi_metrics import compute_kpis

                    row = compute_kpis(pd.read_csv(detail), priority, dt_sec=int(cfg["experiment"]["control_step_sec"]))
                    row.update({"event_id": event_id, "policy_id": policy_id, "detail_file": str(detail)})
                else:
                    fixed = dict(base_settings)
                    fixed[aid] = float(setting)
                    row = run_swmm_trajectory(
                        event_inp,
                        policy_id,
                        _with_fixed_policy(actuators, policy_id, fixed),
                        priority,
                        detail,
                        event_id,
                        int(ev["duration_min"]),
                        int(cfg["experiment"]["control_step_sec"]),
                        int(cfg["experiment"]["random_seed"]),
                        simulation_duration_min=int(ev["simulation_duration_min"]),
                        recession_min=int(cfg["experiment"]["recession_min"]),
                    )
                row.update(
                    {
                        "actuator_id": aid,
                        "tested_setting": float(setting),
                        "base_policy": "no_control",
                        "base_PFV": float(base_row.get("PFV", 0.0)),
                        "base_TFV": float(base_row.get("TFV", 0.0)),
                        "base_peak_TFV_rate": float(base_row.get("peak_TFV_rate", 0.0)),
                        "delta_PFV": float(row.get("PFV", 0.0)) - float(base_row.get("PFV", 0.0)),
                        "delta_TFV": float(row.get("TFV", 0.0)) - float(base_row.get("TFV", 0.0)),
                        "delta_peak_TFV_rate": float(row.get("peak_TFV_rate", 0.0)) - float(base_row.get("peak_TFV_rate", 0.0)),
                    }
                )
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "single_actuator_ablation_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "events": rain["event_id"].astype(str).tolist(),
        "actuators": actuators["actuator_id"].astype(str).tolist(),
        "base_settings": base_settings,
        "settings_tested": settings_to_test,
        "summary": str(out_dir / "single_actuator_ablation_summary.csv"),
    }
    (out_dir / "single_actuator_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
