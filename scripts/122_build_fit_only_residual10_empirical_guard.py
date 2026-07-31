from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import ensure_dir


def _return_group(rain_id: str) -> str:
    digits = "".join(ch for ch in str(rain_id) if ch.isdigit())
    value = int(digits or 0)
    if value <= 10:
        return "T5_T10"
    if value <= 30:
        return "T20_T30"
    return "T50_plus"


def _direction(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    if not arr.size or np.nanmax(np.abs(arr)) < 1.0e-7:
        return "none"
    return "increase" if float(np.nanmean(arr)) > 0.0 else "decrease"


def _pfv_noninferior(delta_pfv: pd.Series, reference_pfv: pd.Series | None, *, absolute_margin: float, relative_margin: float) -> pd.Series:
    delta = pd.to_numeric(delta_pfv, errors="coerce").fillna(np.inf)
    if reference_pfv is None:
        margin = pd.Series(float(absolute_margin), index=delta.index)
    else:
        reference = pd.to_numeric(reference_pfv, errors="coerce").fillna(0.0).clip(lower=0.0)
        margin = pd.Series(
            np.maximum(float(absolute_margin), reference.to_numpy(float) * float(relative_margin)),
            index=delta.index,
        )
    return delta <= margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fit-only Residual10 empirical reliability guard.")
    parser.add_argument("--manifest", default="outputs/project6_36_residual10_core_paired_h120_v1/paired_plan/residual10_core_paired_manifest.csv")
    parser.add_argument("--dataset-audit", default="outputs/project6_36_residual10_core_paired_h120_v1/effect_dataset/residual10_core_effect_audit.csv")
    parser.add_argument("--out-dir", default="outputs/project6_36_residual10_core_paired_h120_v1/empirical_guard")
    parser.add_argument("--pfv-margin-m3", type=float, default=100.0)
    parser.add_argument("--pfv-rel-margin", type=float, default=0.02)
    parser.add_argument("--tfv-deadband-m3", type=float, default=100.0)
    parser.add_argument("--peak-margin", type=float, default=0.0)
    args = parser.parse_args()

    root = Path.cwd()
    manifest_path = root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    audit_path = root / args.dataset_audit if not Path(args.dataset_audit).is_absolute() else Path(args.dataset_audit)
    manifest = pd.read_csv(manifest_path)
    audit = pd.read_csv(audit_path)
    branch_b = manifest[manifest["branch"].astype(str).eq("B")].copy()
    frame = branch_b.merge(audit, on=["pair_id", "event_id", "split", "phase"], how="inner", suffixes=("", "_audit"))
    frame = frame[frame["split"].astype(str).eq("train")].copy()
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        spec = json.loads(row.residual_specification)
        signed = spec.get("signed_profiles", {}) or {}
        targets = spec.get("target_profiles", {}) or {}
        residual_ids = sorted(set(signed) | set(targets))
        for actuator_id in residual_ids:
            profile = signed.get(actuator_id, targets.get(actuator_id, []))
            rows.append(
                {
                    "actuator_id": str(actuator_id),
                    "template_id": str(row.residual_mode),
                    "event_phase": str(row.phase),
                    "return_period_group": _return_group(str(row.rain_id)),
                    "rainfall_pattern": str(row.rain_pattern),
                    "action_direction": _direction(profile),
                    "event_id": str(row.event_id),
                    "pair_id": str(row.pair_id),
                    "reference_pfv_h120": float(getattr(row, "reference_pfv_h120", np.nan)),
                    "delta_pfv_h120": float(row.delta_pfv_h120),
                    "delta_tfv_h120": float(row.delta_tfv_h120),
                    "delta_peak_h120": float(row.delta_peak_h120),
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise RuntimeError("no fit rows available for empirical guard")
    guard_rows = []
    group_cols = ["actuator_id", "template_id", "event_phase", "return_period_group", "rainfall_pattern", "action_direction"]
    for keys, group in expanded.groupby(group_cols, dropna=False):
        reference = group["reference_pfv_h120"] if "reference_pfv_h120" in group else None
        pfv_noninferior = _pfv_noninferior(
            group["delta_pfv_h120"],
            reference,
            absolute_margin=float(args.pfv_margin_m3),
            relative_margin=float(args.pfv_rel_margin),
        )
        guard_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "sample_rows": int(len(group)),
                "sample_events": int(group["event_id"].nunique()),
                "PFV_noninferior_fraction": float(pfv_noninferior.mean()),
                "pfv_abs_margin_m3": float(args.pfv_margin_m3),
                "pfv_rel_margin": float(args.pfv_rel_margin),
                "TFV_improved_fraction": float((group["delta_tfv_h120"] < -float(args.tfv_deadband_m3)).mean()),
                "peak_safe_fraction": float((group["delta_peak_h120"] <= float(args.peak_margin)).mean()),
                "median_delta_pfv_h120": float(group["delta_pfv_h120"].median()),
                "median_delta_tfv_h120": float(group["delta_tfv_h120"].median()),
                "median_delta_peak_h120": float(group["delta_peak_h120"].median()),
                "q10_delta_tfv_h120": float(group["delta_tfv_h120"].quantile(0.10)),
                "q90_delta_peak_h120": float(group["delta_peak_h120"].quantile(0.90)),
                "unknown_combo_default_allow": False,
            }
        )
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    guard = pd.DataFrame(guard_rows).sort_values(["sample_events", "TFV_improved_fraction", "peak_safe_fraction"], ascending=[False, False, False])
    guard_path = out / "residual10_fit_only_empirical_guard.csv"
    guard.to_csv(guard_path, index=False)
    expanded.to_csv(out / "residual10_fit_only_guard_source_rows.csv", index=False)
    report = {
        "guard": str(guard_path),
        "source_rows": int(len(expanded)),
        "groups": int(len(guard)),
        "fit_events": int(expanded["event_id"].nunique()),
        "unknown_combinations_default": "deny_residual_control",
    }
    (out / "empirical_guard_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
