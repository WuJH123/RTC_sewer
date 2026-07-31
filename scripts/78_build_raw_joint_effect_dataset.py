from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.data.peak_label_semantics import repair_paired_risk_rate_sequences

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _source_parts(value: str) -> tuple[str, int]:
    parts = str(value).split(":", 2)
    return parts[0], int(parts[1])


def _sample_source_rows(cache_path: Path, max_files_per_event: int, samples_per_file: int, seed: int) -> list[tuple[str, str, int]]:
    data = np.load(cache_path, allow_pickle=True)
    sources = data["sources"].astype(str)
    events = data["event_ids"].astype(str)
    policies = data["policy_ids"].astype(str)
    grouped: dict[str, list[tuple[str, str, int]]] = {}
    seen: set[tuple[str, str]] = set()
    for source, event, policy in zip(sources, events, policies):
        name, row = _source_parts(source)
        key = (event, name)
        if policy == "no_control" or key in seen:
            continue
        seen.add(key)
        grouped.setdefault(event, []).append((event, name, row))
    rng = np.random.default_rng(seed)
    selected: list[tuple[str, str, int]] = []
    for event, values in sorted(grouped.items()):
        take = values[:]
        rng.shuffle(take)
        selected.extend(take[:max_files_per_event])
    return selected


def _read_event_details(path: Path, reference: Path, node_ids: list[str], priority_nodes: list[str], storage_nodes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = {"event_id", "policy_id", "elapsed_min", "rainfall_mm_h", "phase"}
    needed.update(f"h:{node}" for node in node_ids)
    needed.update(f"flood:{node}" for node in node_ids)
    needed.update(f"a:{aid}" for aid in [])
    candidate_header = pd.read_csv(path, nrows=0).columns.tolist()
    action_cols = [col for col in candidate_header if col.startswith("a:")]
    usecols = [col for col in candidate_header if col in needed or col in action_cols]
    candidate = pd.read_csv(path, usecols=usecols, low_memory=False)
    ref_header = pd.read_csv(reference, nrows=0).columns.tolist()
    ref_usecols = [col for col in ref_header if col in needed or col in action_cols]
    ref = pd.read_csv(reference, usecols=ref_usecols, low_memory=False)
    if len(candidate) != len(ref):
        raise ValueError(f"paired detail row mismatch: {path.name}")
    return candidate, ref


def _matrix(frame: pd.DataFrame, columns: list[str], rows: slice) -> np.ndarray:
    return frame.reindex(columns=columns).iloc[rows].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--max-files-per-event", type=int, default=4)
    ap.add_argument("--samples-per-file", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--event-ids", default="", help="Comma-separated event shard for resumable dataset construction.")
    ap.add_argument("--out-dir", default="outputs/cache_raw_joint")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    cache_path = cfg_path(cfg, "outputs.cache") / "transition_cache.npz"
    nodes = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    node_ids = nodes["node_id"].astype(str).tolist()
    priority_nodes = [x.strip() for x in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    storage_nodes = nodes.loc[nodes["node_type"].astype(str).eq("storage"), "node_id"].astype(str).tolist()
    selected = _sample_source_rows(cache_path, int(args.max_files_per_event), int(args.samples_per_file), int(args.seed))
    requested_events = {value.strip() for value in args.event_ids.split(",") if value.strip()}
    if requested_events:
        selected = [item for item in selected if item[0] in requested_events]
    trajectory_dir = cfg_path(cfg, "outputs.data_bank_train") / "trajectories"
    horizon = int((cfg.get("horizon_surrogate", {}) or {}).get("horizon_steps", 6))
    dt = int((cfg.get("experiment", {}) or {}).get("control_step_sec", 300))
    hcols = [f"h:{node}" for node in node_ids]
    fcols = [f"flood:{node}" for node in node_ids]
    pr_indices = [node_ids.index(node) for node in priority_nodes if node in node_ids]
    st_indices = [node_ids.index(node) for node in storage_nodes if node in node_ids]
    rng = np.random.default_rng(int(args.seed))
    output: dict[str, list[np.ndarray] | list[str] | list[int]] = {key: [] for key in [
        "state", "candidate_action_seq", "reference_action_seq", "rain_seq", "reference_risk_rate_seq",
        "delta_risk_rate_seq", "priority_depth_seq", "storage_level_seq", "target_state_seq", "event_ids", "source_files", "row_index",
    ]}
    failures = []
    for index, (event_id, name, _) in enumerate(selected, 1):
        candidate_path = trajectory_dir / name
        reference_path = trajectory_dir / f"{event_id}__no_control_detail.csv"
        if not candidate_path.exists() or not reference_path.exists():
            failures.append({"event_id": event_id, "source": name, "error": "missing_detail"}); continue
        try:
            candidate, reference = _read_event_details(candidate_path, reference_path, node_ids, priority_nodes, storage_nodes)
            action_cols = [col for col in candidate.columns if col.startswith("a:")]
            if len(action_cols) != 109:
                raise ValueError(f"expected 109 actions, found {len(action_cols)}")
            valid_starts = np.arange(0, max(0, len(candidate) - horizon - 1), dtype=int)
            if not len(valid_starts):
                continue
            take = rng.choice(valid_starts, size=min(int(args.samples_per_file), len(valid_starts)), replace=False)
            for start in sorted(take.tolist()):
                state = _matrix(reference, hcols, slice(start, start + 1))[0]
                candidate_actions = _matrix(candidate, action_cols, slice(start, start + horizon))
                reference_actions = _matrix(reference, action_cols, slice(start, start + horizon))
                rain = pd.to_numeric(reference.get("rainfall_mm_h", pd.Series(0.0, index=reference.index)), errors="coerce").fillna(0.0).to_numpy(np.float32)[start:start+horizon, None]
                c_flood = _matrix(candidate, fcols, slice(start + 1, start + 1 + horizon))
                r_flood = _matrix(reference, fcols, slice(start + 1, start + 1 + horizon))
                c_depth = _matrix(candidate, hcols, slice(start + 1, start + 1 + horizon))
                r_rate = np.stack([
                    r_flood[:, pr_indices].sum(axis=1) if pr_indices else np.zeros(horizon),
                    r_flood.sum(axis=1), r_flood.sum(axis=1),
                ], axis=1).astype(np.float32)
                d_rate = np.stack([
                    (c_flood[:, pr_indices].sum(axis=1) - r_flood[:, pr_indices].sum(axis=1)) if pr_indices else np.zeros(horizon),
                    c_flood.sum(axis=1) - r_flood.sum(axis=1),
                    c_flood.sum(axis=1) - r_flood.sum(axis=1),
                ], axis=1).astype(np.float32)
                r_rate, d_rate = repair_paired_risk_rate_sequences(
                    r_rate[None, ...], d_rate[None, ...]
                )
                r_rate, d_rate = r_rate[0], d_rate[0]
                output["state"].append(state); output["candidate_action_seq"].append(candidate_actions)
                output["reference_action_seq"].append(reference_actions); output["rain_seq"].append(rain)
                output["reference_risk_rate_seq"].append(r_rate); output["delta_risk_rate_seq"].append(d_rate)
                output["priority_depth_seq"].append(c_depth[:, pr_indices].mean(axis=1) if pr_indices else c_depth.mean(axis=1))
                output["storage_level_seq"].append(c_depth[:, st_indices].mean(axis=1) if st_indices else c_depth.mean(axis=1))
                output["target_state_seq"].append(c_depth); output["event_ids"].append(event_id)
                output["source_files"].append(name); output["row_index"].append(start)
        except Exception as exc:
            failures.append({"event_id": event_id, "source": name, "error": repr(exc)})
        if index % 100 == 0: print(f"[raw_joint_dataset] files={index}/{len(selected)} samples={len(output['event_ids'])} failures={len(failures)}", flush=True)
    out = ensure_dir(root / args.out_dir)
    dataset = out / "raw_joint_no_control_effect_dataset.npz"
    np.savez_compressed(dataset, **{key: np.asarray(value) for key, value in output.items()})
    report = {
        "dataset": str(dataset), "samples": len(output["event_ids"]), "events": len(set(output["event_ids"])),
        "selected_source_files": len(selected), "failures": failures[:50], "failure_count": len(failures),
        "horizon_steps": horizon, "node_count": len(node_ids), "action_count": 109,
        "target_semantics": "candidate trajectory minus same-event same-time no_control trajectory; paired trajectory effect, not reset-state SWMM counterfactual",
        "cache_source": str(cache_path), "source_index_used_for_selection": True,
        "requested_events": sorted(requested_events),
    }
    (out / "raw_joint_no_control_effect_dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
