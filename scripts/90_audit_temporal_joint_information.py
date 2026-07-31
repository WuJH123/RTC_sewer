from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.experiments.effect_audit import resolve_audit_metadata
from sewerrtc.experiments.targeted_joint_pairs import event_pattern, event_return_period
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _direction(value: float, tolerance: float) -> str:
    if value < -float(tolerance):
        return "improved"
    if value > float(tolerance):
        return "worsened"
    return "deadband"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--dataset", default="outputs/project6_36_temporal_joint_v1/effect_dataset/same_state_raw_joint_36.npz")
    parser.add_argument("--manifest", default="outputs/project6_36_temporal_joint_v1/paired_plan/joint_action_case_manifest.csv")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v1/effect_dataset")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    dataset_path = root / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    manifest_path = root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    data = np.load(dataset_path, allow_pickle=True)
    manifest = pd.read_csv(manifest_path)
    manifest["pair_id"] = manifest["pair_id"].astype(str)
    candidates = manifest[manifest["branch"].astype(str).eq("B")].drop_duplicates("pair_id").set_index("pair_id")
    event_ids = data["event_ids"].astype(str)
    pair_ids = data["pair_ids"].astype(str)
    action_ids = data["action_ids"].astype(str).tolist()
    candidate_actions = data["candidate_action_seq"].astype(np.float64)
    reference_actions = data["reference_action_seq"].astype(np.float64)
    residual = candidate_actions - reference_actions
    reference_risk = data["reference_risk_rate_seq"].astype(np.float64)
    delta_risk = data["delta_risk_rate_seq"].astype(np.float64)
    candidate_risk = reference_risk + delta_risk
    delta_pfv = delta_risk[:, :, 0].sum(axis=1) * 300.0
    delta_tfv = delta_risk[:, :, 1].sum(axis=1) * 300.0
    delta_peak = candidate_risk[:, :, 1].max(axis=1) - reference_risk[:, :, 1].max(axis=1)
    reference_pfv = reference_risk[:, :, 0].sum(axis=1) * 300.0
    temporal_cfg = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}))
    validation = temporal_cfg.get("training_validation", {}) or {}
    safety = temporal_cfg.get("safety", {}) or {}
    pfv_deadband = float(validation.get("pfv_direction_tolerance_m3", 1.0))
    tfv_deadband = float(validation.get("tfv_direction_tolerance_m3", 100.0))
    peak_deadband = float(validation.get("peak_direction_tolerance", 0.1))
    pfv_abs = float(safety.get("pfv_abs_margin_m3", 100.0))
    pfv_rel = float(safety.get("pfv_rel_margin", 0.005))
    peak_margin = float(safety.get("peak_margin", 0.0))
    embedded_split = data["split"].astype(str) if "split" in data.files else None
    events = sorted(set(event_ids))
    validation_events = (
        set(event_ids[embedded_split == "validation"])
        if embedded_split is not None
        else set(events[::4])
    )
    embedded_metadata = {
        key: data[key]
        for key in ("phase", "checkpoint_id", "candidate_kind", "candidate_family", "source_dataset")
        if key in data.files
    }
    rows = []
    for index, (event_id, pair_id) in enumerate(zip(event_ids, pair_ids)):
        metadata = resolve_audit_metadata(
            pair_id=str(pair_id),
            event_id=str(event_id),
            row_index=index,
            candidates=candidates,
            data=embedded_metadata,
        )
        specification = metadata["specification"]
        changed = np.abs(residual[index]) > 1.0e-7
        changed_positions = np.flatnonzero(changed.any(axis=0))
        changed_ids = [action_ids[position] for position in changed_positions]
        actual_delta = {
            action_ids[position]: residual[index, :, position].astype(float).tolist()
            for position in changed_positions
        }
        margin = max(pfv_abs, pfv_rel * max(0.0, float(reference_pfv[index])))
        rows.append({
            "event_id": event_id,
            "checkpoint_id": metadata["checkpoint_id"],
            "split": str(embedded_split[index]) if embedded_split is not None else ("validation" if event_id in validation_events else "train"),
            "return_period": event_return_period(event_id),
            "rain_pattern": event_pattern(event_id),
            "storm_phase": metadata["phase"],
            "candidate_kind": metadata["candidate_kind"],
            "candidate_mode": metadata["candidate_mode"],
            "actuator_ids": ",".join(changed_ids or map(str, specification.get("actuators", []))),
            "requested_action_delta": json.dumps({
                key: value for key, value in specification.items()
                if key in {"delta", "signed_profile", "signed_profiles", "target_profile", "target_profiles"}
            }, sort_keys=True),
            "actual_delta_after_clipping": json.dumps(actual_delta, sort_keys=True),
            "changed_actuator_count": int(len(changed_positions)),
            "changed_time_step_count": int(changed.any(axis=1).sum()),
            "max_simultaneous_changes": int(changed.sum(axis=1).max(initial=0)),
            "action_L1_difference": float(np.abs(residual[index]).sum()),
            "action_Linf_difference": float(np.abs(residual[index]).max(initial=0.0)),
            "is_noop": not bool(changed.any()),
            "delta_PFV_m3": float(delta_pfv[index]),
            "delta_TFV_m3": float(delta_tfv[index]),
            "delta_peak_TFV_rate": float(delta_peak[index]),
            "PFV_direction": _direction(float(delta_pfv[index]), pfv_deadband),
            "PFV_noninferiority_margin_m3": margin,
            "PFV_noninferiority": "safe" if float(delta_pfv[index]) <= margin else "unsafe",
            "TFV_outcome": _direction(float(delta_tfv[index]), tfv_deadband),
            "peak_outcome": "safe" if float(delta_peak[index]) <= peak_margin else "unsafe",
            "peak_direction": _direction(float(delta_peak[index]), peak_deadband),
            "pair_id": pair_id,
            "manifest_match": bool(metadata["manifest_match"]),
            "metadata_source": "manifest+embedded_dataset" if metadata["manifest_match"] else "embedded_dataset",
            "source_dataset": metadata["source_dataset"],
        })
    audit = pd.DataFrame(rows)
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    audit.to_csv(out / "paired_information_audit.csv", index=False)
    audit[audit["is_noop"]].to_csv(out / "noop_candidate_audit.csv", index=False)
    categories = ["PFV_direction", "PFV_noninferiority", "TFV_outcome", "peak_outcome", "peak_direction"]
    support_rows = []
    for (split, event_id), group in audit.groupby(["split", "event_id"], sort=True):
        row = {
            "split": split,
            "event_id": event_id,
            "return_period": group["return_period"].iloc[0],
            "rain_pattern": group["rain_pattern"].iloc[0],
            "rows": len(group),
            "non_noop_rows": int((~group["is_noop"]).sum()),
        }
        for category in categories:
            for label, count in group[category].value_counts().items():
                row[f"{category}__{label}"] = int(count)
        support_rows.append(row)
    support = pd.DataFrame(support_rows).fillna(0)
    support.to_csv(out / "event_label_support.csv", index=False)
    summary = {
        "dataset": str(dataset_path),
        "rows": len(audit),
        "events": int(audit["event_id"].nunique()),
        "noops": int(audit["is_noop"].sum()),
        "noop_fraction": float(audit["is_noop"].mean()),
        "effective_rows": int((~audit["is_noop"]).sum()),
        "manifest_matched_rows": int(audit["manifest_match"].sum()),
        "embedded_metadata_fallback_rows": int((~audit["manifest_match"]).sum()),
        "source_dataset_rows": audit["source_dataset"].value_counts().to_dict(),
        "split_event_counts": audit.groupby("split")["event_id"].nunique().to_dict(),
        "label_counts": {category: audit[category].value_counts().to_dict() for category in categories},
        "deadbands": {"PFV_m3": pfv_deadband, "TFV_m3": tfv_deadband, "peak": peak_deadband},
        "pfv_noninferiority": {"absolute_m3": pfv_abs, "relative": pfv_rel},
    }
    (out / "paired_information_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
