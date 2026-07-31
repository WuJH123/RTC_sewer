from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.data.peak_label_semantics import repair_paired_risk_rate_sequences

from sewerrtc.data.targeted_dataset_metadata import resolve_old_metadata

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


ARRAY_KEYS = (
    "state",
    "candidate_action_seq",
    "reference_action_seq",
    "rain_seq",
    "reference_risk_rate_seq",
    "delta_risk_rate_seq",
    "priority_depth_seq",
    "storage_level_seq",
    "target_state_seq",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window(frame: pd.DataFrame, start_min: float, horizon: int) -> pd.DataFrame:
    times = pd.to_numeric(frame["elapsed_min"], errors="coerce").to_numpy(float)
    start = int(np.searchsorted(times, float(start_min), side="left"))
    result = frame.iloc[start : start + horizon].copy()
    if len(result) != horizon:
        raise ValueError(f"incomplete horizon at {start_min}: {len(result)}/{horizon}")
    return result


def _new_payload(
    *,
    results: pd.DataFrame,
    node_ids: list[str],
    action_ids: list[str],
    priority_ids: list[str],
    storage_ids: list[str],
    horizon: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, list] = {key: [] for key in ("event_ids", "pair_ids", *ARRAY_KEYS)}
    metadata: dict[str, list] = {
        "split": [],
        "candidate_kind": [],
        "candidate_family": [],
        "phase": [],
        "checkpoint_id": [],
        "source_dataset": [],
    }
    failures: list[dict[str, str]] = []
    for row in results.itertuples(index=False):
        try:
            candidate = _window(pd.read_csv(row.candidate_detail), float(row.override_start_min), horizon)
            reference = _window(pd.read_csv(row.reference_detail), float(row.override_start_min), horizon)
            state_columns = [f"h:{node_id}" for node_id in node_ids]
            action_columns = [f"a:{actuator_id}" for actuator_id in action_ids]
            candidate_state = candidate[state_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            reference_state = reference[state_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            candidate_action = candidate[action_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            reference_action = reference[action_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            planned = np.asarray(json.loads(str(row.materialized_candidate_action_sequence)), dtype=np.float32)
            planned_reference = np.asarray(json.loads(str(row.materialized_reference_action_sequence)), dtype=np.float32)
            if candidate_action.shape != (horizon, len(action_ids)):
                raise ValueError(f"candidate action shape is {candidate_action.shape}")
            if not np.allclose(candidate_action, planned, atol=1.0e-6):
                raise ValueError("realized candidate action does not match preflight materialization")
            if not np.allclose(reference_action, planned_reference, atol=1.0e-6):
                raise ValueError("realized reference action does not match preflight materialization")
            if np.max(np.abs(candidate_action - reference_action)) <= 1.0e-7:
                raise ValueError("targeted candidate became a no-op during SWMM execution")

            flood_columns = [column for column in reference.columns if column.startswith("flood:")]
            priority_columns = [f"flood:{node_id}" for node_id in priority_ids if f"flood:{node_id}" in reference]
            reference_pfv = reference[priority_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            candidate_pfv = candidate[priority_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            reference_tfv = reference[flood_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            candidate_tfv = candidate[flood_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            reference_risk = np.stack([reference_pfv, reference_tfv, reference_tfv], axis=1)[None, ...]
            candidate_risk = np.stack([candidate_pfv, candidate_tfv, candidate_tfv], axis=1)[None, ...]
            reference_risk, delta_risk = repair_paired_risk_rate_sequences(
                reference_risk,
                candidate_risk - reference_risk,
            )
            reference_risk, delta_risk = reference_risk[0], delta_risk[0]

            arrays["event_ids"].append(str(row.event_id))
            arrays["pair_ids"].append(str(row.pair_id))
            arrays["state"].append(reference_state[0])
            arrays["candidate_action_seq"].append(candidate_action)
            arrays["reference_action_seq"].append(reference_action)
            arrays["rain_seq"].append(reference[["rainfall_mm_h"]].to_numpy(np.float32))
            arrays["reference_risk_rate_seq"].append(reference_risk)
            arrays["delta_risk_rate_seq"].append(delta_risk)
            priority_depth_columns = [f"h:{node_id}" for node_id in priority_ids if f"h:{node_id}" in candidate]
            arrays["priority_depth_seq"].append(
                candidate[priority_depth_columns].mean(axis=1).to_numpy(np.float32)
            )
            storage_columns = [f"h:{node_id}" for node_id in storage_ids if f"h:{node_id}" in candidate]
            arrays["storage_level_seq"].append(
                candidate[storage_columns].mean(axis=1).to_numpy(np.float32)
                if storage_columns
                else np.zeros(horizon, dtype=np.float32)
            )
            arrays["target_state_seq"].append(candidate_state)
            metadata["split"].append(str(row.split))
            metadata["candidate_kind"].append(str(row.candidate_kind))
            metadata["candidate_family"].append(str(row.candidate_family))
            metadata["phase"].append(str(row.phase))
            metadata["checkpoint_id"].append(str(row.checkpoint_id))
            metadata["source_dataset"].append("targeted_informative_v2")
        except Exception as exc:
            failures.append({"case_id": str(getattr(row, "case_id", "unknown")), "error": repr(exc)})
    if failures:
        raise RuntimeError(f"targeted dataset rejected {len(failures)} cases; first={failures[0]}")
    return {**{key: np.asarray(value) for key, value in arrays.items()}, **{key: np.asarray(value) for key, value in metadata.items()}}


def _balanced_noop_indices(event_ids: np.ndarray, noop_indices: np.ndarray, count: int) -> np.ndarray:
    by_event: dict[str, list[int]] = {}
    for index in noop_indices.tolist():
        by_event.setdefault(str(event_ids[index]), []).append(int(index))
    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for event_id in sorted(by_event):
            if by_event[event_id] and len(selected) < count:
                selected.append(by_event[event_id].pop(0))
                progressed = True
        if not progressed:
            break
    return np.asarray(selected, dtype=int)


def _label_audit(payload: dict[str, np.ndarray], *, pfv_abs: float, pfv_rel: float, tfv_deadband: float, peak_margin: float) -> pd.DataFrame:
    reference = payload["reference_risk_rate_seq"].astype(np.float64)
    delta = payload["delta_risk_rate_seq"].astype(np.float64)
    candidate = reference + delta
    delta_pfv = delta[:, :, 0].sum(axis=1) * 300.0
    delta_tfv = delta[:, :, 1].sum(axis=1) * 300.0
    delta_peak = candidate[:, :, 1].max(axis=1) - reference[:, :, 1].max(axis=1)
    reference_pfv = reference[:, :, 0].sum(axis=1) * 300.0
    margin = np.maximum(float(pfv_abs), float(pfv_rel) * np.maximum(reference_pfv, 0.0))
    residual = payload["candidate_action_seq"] - payload["reference_action_seq"]
    return pd.DataFrame({
        "event_id": payload["event_ids"].astype(str),
        "pair_id": payload["pair_ids"].astype(str),
        "split": payload["split"].astype(str),
        "source_dataset": payload["source_dataset"].astype(str),
        "candidate_family": payload["candidate_family"].astype(str),
        "phase": payload["phase"].astype(str),
        "is_noop": np.abs(residual).max(axis=(1, 2)) <= 1.0e-7,
        "delta_PFV_m3": delta_pfv,
        "delta_TFV_m3": delta_tfv,
        "delta_peak": delta_peak,
        "PFV_noninferiority_margin_m3": margin,
        "PFV_noninferior": delta_pfv <= margin,
        "TFV_improved": delta_tfv < -float(tfv_deadband),
        "TFV_worsened": delta_tfv > float(tfv_deadband),
        "peak_safe": delta_peak <= float(peak_margin),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--old-dataset", default="outputs/project6_36_temporal_joint_v1/effect_dataset/same_state_raw_joint_36.npz")
    parser.add_argument("--case-dir", default="outputs/project6_36_temporal_joint_v2/paired_cases")
    parser.add_argument("--correction-case-dir", default="")
    parser.add_argument("--extra-case-dir", action="append", default=[])
    parser.add_argument("--manifest", default="outputs/project6_36_temporal_joint_v2/joint_data_plan/targeted_informative_paired_manifest.csv")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v2/effect_dataset")
    parser.add_argument("--noop-fraction", type=float, default=0.06)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    old_path = root / args.old_dataset if not Path(args.old_dataset).is_absolute() else Path(args.old_dataset)
    case_dir = root / args.case_dir if not Path(args.case_dir).is_absolute() else Path(args.case_dir)
    manifest_path = root / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    old = np.load(old_path, allow_pickle=True)
    results = pd.read_csv(case_dir / "paired_candidate_results.csv")
    correction_case_dir = None
    if args.correction_case_dir:
        correction_case_dir = root / args.correction_case_dir if not Path(args.correction_case_dir).is_absolute() else Path(args.correction_case_dir)
        correction_results = pd.read_csv(correction_case_dir / "paired_candidate_results.csv")
        results = pd.concat([results, correction_results], ignore_index=True).drop_duplicates("case_id", keep="last")
    extra_case_dirs: list[Path] = []
    for value in args.extra_case_dir:
        extra_dir = root / value if not Path(value).is_absolute() else Path(value)
        extra_case_dirs.append(extra_dir)
        extra_results = pd.read_csv(extra_dir / "paired_candidate_results.csv")
        results = pd.concat([results, extra_results], ignore_index=True).drop_duplicates("case_id", keep="last")
    manifest = pd.read_csv(manifest_path)
    expected = manifest[manifest["branch"].astype(str).eq("B")]["execution_case_id"].astype(str).nunique()
    if len(results) != expected or results["execution_case_id"].astype(str).nunique() != expected:
        raise RuntimeError(f"targeted SWMM output is incomplete: {len(results)}/{expected}")
    failures_path = case_dir / "failures.csv"
    if failures_path.exists() and failures_path.stat().st_size > 0:
        try:
            failures = pd.read_csv(failures_path)
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame()
        if len(failures):
            raise RuntimeError("targeted SWMM output contains failures")
    if correction_case_dir is not None:
        correction_failures_path = correction_case_dir / "failures.csv"
        if correction_failures_path.exists() and correction_failures_path.stat().st_size > 0:
            try:
                correction_failures = pd.read_csv(correction_failures_path)
            except pd.errors.EmptyDataError:
                correction_failures = pd.DataFrame()
            if len(correction_failures):
                raise RuntimeError("alignment-correction SWMM output contains failures")
    for extra_dir in extra_case_dirs:
        extra_failures_path = extra_dir / "failures.csv"
        if extra_failures_path.exists() and extra_failures_path.stat().st_size > 0:
            try:
                extra_failures = pd.read_csv(extra_failures_path)
            except pd.errors.EmptyDataError:
                extra_failures = pd.DataFrame()
            if len(extra_failures):
                raise RuntimeError(f"extra SWMM output contains failures: {extra_dir}")

    node_ids = old["node_ids"].astype(str).tolist()
    action_ids = old["action_ids"].astype(str).tolist()
    horizon = int(old["horizon_steps"].item())
    node_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    storage_ids = node_table[node_table["node_type"].astype(str).str.lower().eq("storage")]["node_id"].astype(str).tolist()
    priority_ids = [line.strip() for line in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines() if line.strip()]
    new = _new_payload(
        results=results,
        node_ids=node_ids,
        action_ids=action_ids,
        priority_ids=priority_ids,
        storage_ids=storage_ids,
        horizon=horizon,
    )

    old_residual = old["candidate_action_seq"] - old["reference_action_seq"]
    old_noop = np.max(np.abs(old_residual), axis=(1, 2)) <= 1.0e-7
    active_indices = np.flatnonzero(~old_noop)
    available_noops = np.flatnonzero(old_noop)
    base_count = len(active_indices) + len(new["event_ids"])
    requested_noops = int(round(float(args.noop_fraction) * base_count / max(1.0e-9, 1.0 - float(args.noop_fraction))))
    selected_noops = _balanced_noop_indices(old["event_ids"].astype(str), available_noops, min(requested_noops, len(available_noops)))
    old_indices = np.sort(np.concatenate([active_indices, selected_noops]))

    old_manifest = pd.read_csv(root / "outputs/project6_36_temporal_joint_v1/paired_plan/joint_action_case_manifest.csv")
    old_metadata = resolve_old_metadata(old, old_manifest)
    payload: dict[str, np.ndarray] = {}
    for key in ("event_ids", "pair_ids", *ARRAY_KEYS):
        payload[key] = np.concatenate([old[key][old_indices], new[key]], axis=0)
    for key in old_metadata:
        payload[key] = np.concatenate([old_metadata[key][old_indices], new[key]], axis=0)
    payload.update({
        "node_ids": np.asarray(node_ids),
        "action_ids": np.asarray(action_ids),
        "label_semantics": np.asarray("same_state_candidate_minus_no_control"),
        "horizon_steps": np.asarray(horizon),
    })

    temporal = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}))
    validation = temporal.get("training_validation", {}) or {}
    safety = temporal.get("safety", {}) or {}
    audit = _label_audit(
        payload,
        pfv_abs=float(safety.get("pfv_abs_margin_m3", 100.0)),
        pfv_rel=float(safety.get("pfv_rel_margin", 0.005)),
        tfv_deadband=float(validation.get("tfv_direction_tolerance_m3", 100.0)),
        peak_margin=float(safety.get("peak_margin", 0.0)),
    )
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    dataset_path = out / "same_state_raw_joint_36_v3.npz"
    np.savez_compressed(dataset_path, **payload)
    audit.to_csv(out / "v3_effect_label_audit.csv", index=False)
    support = audit.groupby(["split", "source_dataset"], dropna=False).agg(
        rows=("event_id", "size"),
        events=("event_id", "nunique"),
        noops=("is_noop", "sum"),
        PFV_safe=("PFV_noninferior", "sum"),
        TFV_improved=("TFV_improved", "sum"),
        TFV_worsened=("TFV_worsened", "sum"),
        peak_safe=("peak_safe", "sum"),
    ).reset_index()
    support["PFV_unsafe"] = support["rows"] - support["PFV_safe"]
    support["peak_unsafe"] = support["rows"] - support["peak_safe"]
    support.to_csv(out / "v3_event_label_support.csv", index=False)
    noop_fraction = float(audit["is_noop"].mean())
    report = {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "old_dataset": str(old_path),
        "old_dataset_sha256": _sha256(old_path),
        "targeted_manifest": str(manifest_path),
        "targeted_manifest_sha256": _sha256(manifest_path),
        "samples": len(audit),
        "events": int(audit["event_id"].nunique()),
        "old_active_rows": int(len(active_indices)),
        "old_noops_available": int(len(available_noops)),
        "old_noops_retained": int(len(selected_noops)),
        "new_targeted_rows": int(len(new["event_ids"])),
        "final_noop_fraction": noop_fraction,
        "noop_fraction_in_5_to_10_percent": bool(0.05 <= noop_fraction <= 0.10),
        "train_events": sorted(audit.loc[audit["split"].eq("train"), "event_id"].unique().tolist()),
        "validation_events": sorted(audit.loc[audit["split"].eq("validation"), "event_id"].unique().tolist()),
        "event_group_split_disjoint": not bool(
            set(audit.loc[audit["split"].eq("train"), "event_id"])
            & set(audit.loc[audit["split"].eq("validation"), "event_id"])
        ),
        "PFV_unsafe_rows": int((~audit["PFV_noninferior"]).sum()),
        "PFV_unsafe_events": int(audit.loc[~audit["PFV_noninferior"], "event_id"].nunique()),
        "TFV_improved_rows": int(audit["TFV_improved"].sum()),
        "TFV_worsened_rows": int(audit["TFV_worsened"].sum()),
        "peak_unsafe_rows": int((~audit["peak_safe"]).sum()),
        "formal_calibration_leakage": False,
        "old_dataset_overwritten": False,
    }
    (out / "v3_dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["noop_fraction_in_5_to_10_percent"] or not report["event_group_split_disjoint"]:
        raise SystemExit("v3 dataset integrity gate failed")


if __name__ == "__main__":
    main()
