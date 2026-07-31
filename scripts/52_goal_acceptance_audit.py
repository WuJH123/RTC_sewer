from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, load_config


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_json(path: Path) -> dict:
    if not _exists_nonempty(path):
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not _exists_nonempty(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _row(requirement: str, passed: bool, evidence: str, detail: str = "") -> dict:
    return {
        "requirement": requirement,
        "passed": bool(passed),
        "evidence": evidence,
        "detail": detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    eval_dir = root / "outputs" / "evaluation"
    surrogate_dir = root / "outputs" / "surrogate"
    model_dir = root / "outputs" / "models_paired_no_controls"
    generic_dir = root / "outputs" / "generic_rtc"
    network_dir = root / "outputs" / "network"
    logs_dir = root / "outputs" / "logs"
    rows = []

    risk_table = _read_csv(eval_dir / "risk_stratified_event_table.csv")
    risk_counts = risk_table.get("event_risk_class", pd.Series(dtype=str)).value_counts().to_dict() if not risk_table.empty else {}
    rows.append(
        _row(
            "risk_stratified_event_table.csv generated",
            _exists_nonempty(eval_dir / "risk_stratified_event_table.csv"),
            str(eval_dir / "risk_stratified_event_table.csv"),
            f"rows={len(risk_table)}",
        )
    )
    rows.append(
        _row(
            "high/medium/low risk event counts are present",
            all(k in risk_counts for k in ["high_risk_event", "medium_risk_event", "low_risk_event"]),
            str(eval_dir / "risk_stratified_event_table.csv"),
            json.dumps({str(k): int(v) for k, v in risk_counts.items()}, ensure_ascii=False),
        )
    )

    horizon_path = root / cfg.get("horizon_surrogate", {}).get("output_dataset", "data/surrogate/horizon_mpc_dataset.parquet")
    horizon_audit = _read_json(surrogate_dir / "horizon_dataset_audit.json")
    rows.append(
        _row(
            "formal horizon_mpc_dataset.parquet generated",
            _exists_nonempty(horizon_path),
            str(horizon_path),
            f"samples={horizon_audit.get('samples', 'missing')}",
        )
    )
    val = _read_csv(surrogate_dir / "horizon_surrogate_validation.csv")
    targets = set(val.get("target", pd.Series(dtype=str)).astype(str).tolist()) if not val.empty else set()
    rows.append(
        _row(
            "horizon surrogate outputs PFV_H/TFV_H/peak_TFV_rate_H",
            {"PFV_H", "TFV_H", "peak_TFV_rate_H"}.issubset(targets),
            str(surrogate_dir / "horizon_surrogate_validation.csv"),
            ",".join(sorted(targets)),
        )
    )
    unc = _read_json(surrogate_dir / "uncertainty_gate_validation_summary.json")
    can_output = set(map(str, unc.get("can_output", [])))
    rows.append(
        _row(
            "uncertainty gate outputs p50/p90-style deltas",
            {"delta_PFV_p50", "delta_PFV_p90", "delta_TFV_p90", "delta_peak_p90"}.issubset(can_output),
            str(surrogate_dir / "uncertainty_gate_validation_summary.json"),
            ",".join(sorted(can_output)),
        )
    )

    main_table = _read_csv(eval_dir / "water_research_main_table.csv")
    summary = _read_json(eval_dir / "risk_stratified_summary.json")
    low_false = summary.get("low_risk_false_intervention_rate")
    high_pfv = summary.get("high_risk_PFV_mean_reduction_pct")
    high_tfv = summary.get("high_risk_TFV_mean_reduction_pct")
    high_peak = summary.get("high_risk_peak_mean_reduction_pct")
    max_false = float((cfg.get("intervention_policy", {}) or {}).get("max_false_intervention_rate_low_risk", 0.05))
    rows.append(
        _row(
            "NativeShield low-risk false intervention below configured target",
            low_false is not None and float(low_false) <= max_false,
            str(eval_dir / "risk_stratified_summary.json"),
            f"low_false={low_false}; target<={max_false}",
        )
    )
    rows.append(
        _row(
            "high-risk PFV_mean_reduction_pct > 0",
            high_pfv is not None and float(high_pfv) > 0.0,
            str(eval_dir / "risk_stratified_summary.json"),
            f"high_pfv={high_pfv}",
        )
    )
    rows.append(
        _row(
            "high-risk TFV and peak not clearly worsened",
            high_tfv is not None and high_peak is not None and float(high_tfv) >= -0.5 and float(high_peak) >= -1.0,
            str(eval_dir / "risk_stratified_summary.json"),
            f"high_tfv={high_tfv}; high_peak={high_peak}",
        )
    )
    rows.append(
        _row(
            "Generic RTC smoke can run without native rules",
            bool(_read_json(generic_dir / "generic_smoke_summary.json").get("does_not_require_native_rules", False)),
            str(generic_dir / "generic_smoke_summary.json"),
        )
    )
    rows.append(
        _row(
            "water_research_main_table.csv generated",
            _exists_nonempty(eval_dir / "water_research_main_table.csv") and not main_table.empty,
            str(eval_dir / "water_research_main_table.csv"),
            f"rows={len(main_table)}",
        )
    )
    rows.append(
        _row(
            "influence-domain candidate files generated",
            _exists_nonempty(network_dir / "priority_to_actuator_candidates.csv"),
            str(network_dir / "priority_to_actuator_candidates.csv"),
        )
    )
    rows.append(
        _row(
            "stage scripts emitted logs",
            logs_dir.exists() and any(logs_dir.glob("stage*.log")),
            str(logs_dir),
        )
    )

    out = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else eval_dir / "goal_acceptance_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    result = {
        "passed": bool(out["passed"].all()) if not out.empty else False,
        "passed_count": int(out["passed"].sum()) if not out.empty else 0,
        "total": int(len(out)),
        "failed": out.loc[~out["passed"], ["requirement", "detail"]].to_dict(orient="records"),
        "output": str(out_path),
    }
    (eval_dir / "goal_acceptance_audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
