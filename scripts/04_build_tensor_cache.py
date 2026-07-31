from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.data.tensor_cache import build_transition_cache
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _validate_action_schema(trajectory_dir: Path, manifest_path: Path) -> dict:
    """Fail closed when a detail file is not in the declared action space.

    Filling a missing action column with a default setting is reasonable for a
    permissive exploratory cache, but invalid for the formal 36-asset study:
    it silently changes the meaning of a sampled control trajectory.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing trajectory action-schema manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = [f"a:{aid}" for aid in manifest.get("actuator_ids", [])]
    if not expected:
        raise ValueError(f"Action-schema manifest has no actuator IDs: {manifest_path}")
    violations = []
    files = sorted(trajectory_dir.glob("*_detail.csv"))
    for path in files:
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as exc:
            violations.append({"detail_file": str(path), "error": f"unreadable_header:{exc!r}"})
            continue
        observed = [col for col in columns if str(col).startswith("a:")]
        if observed != expected:
            violations.append(
                {
                    "detail_file": str(path),
                    "expected_action_count": len(expected),
                    "observed_action_count": len(observed),
                    "missing": [col for col in expected if col not in observed],
                    "extra": [col for col in observed if col not in expected],
                    "order_matches": observed == expected,
                }
            )
            if len(violations) >= 20:
                break
    if violations:
        raise ValueError(
            "Trajectory action schema is inconsistent with action_scope_manifest; "
            f"examples={violations[:3]}"
        )
    return {"detail_files_checked": len(files), "action_count": len(expected), "action_columns": expected}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--time-stride", type=int, default=1)
    ap.add_argument("--horizon-steps", type=int, default=0)
    ap.add_argument("--baseline-policy", default="")
    ap.add_argument("--reference-policies", default="")
    ap.add_argument("--no-current-event-filter", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg_path(cfg, "outputs.cache"))
    trajectory_dir = cfg_path(cfg, "outputs.data_bank_train") / "trajectories"
    action_schema = _validate_action_schema(
        trajectory_dir,
        cfg_path(cfg, "outputs.data_bank_train") / "action_scope_manifest.json",
    )
    priority_nodes = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    allowed_event_ids = None
    rainfall_event_table = cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"
    if not args.no_current_event_filter:
        rain = pd.read_csv(rainfall_event_table)
        allowed_event_ids = set(rain["event_id"].astype(str)) if "event_id" in rain else set()
    training_cfg = cfg.get("training", {}) or {}
    schedule_path = cfg_path(cfg, "outputs.data_bank_train") / "trajectory_schedule.csv"
    allowed_event_policy_keys = None
    if schedule_path.exists():
        schedule = pd.read_csv(schedule_path)
        allowed_event_policy_keys = set(zip(schedule["event_id"].astype(str), schedule["policy_id"].astype(str)))
    reference_policies = args.reference_policies.strip()
    if not reference_policies:
        configured = training_cfg.get("cache_reference_policies", ["no_control", "efd_storage_priority", "auto_rbc"])
        reference_policies = ",".join(str(x) for x in configured)
    meta = build_transition_cache(
        trajectory_dir,
        out / "transition_cache.npz",
        args.max_files,
        args.time_stride,
        args.horizon_steps or int(cfg["training"].get("surrogate_horizon_steps", 6)),
        priority_nodes,
        int(cfg["experiment"].get("control_step_sec", 300)),
        args.baseline_policy,
        allowed_event_ids=allowed_event_ids,
        allowed_event_policy_keys=allowed_event_policy_keys,
        reference_policies=reference_policies,
    )
    meta["rainfall_event_table"] = str(rainfall_event_table)
    meta["current_event_filter"] = not args.no_current_event_filter
    meta["trajectory_schedule"] = str(schedule_path) if schedule_path.exists() else ""
    meta["action_schema_validation"] = action_schema
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
