"""Build isolated causal-history manifests for fresh PFV-only Calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--raw-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    raw = pd.read_parquet(args.raw_manifest)
    required = {
        "state_key",
        "event_id",
        "rainfall_sha256",
        "split_group_key",
        "checkpoint_min",
        "source_detail_path_candidate",
        "source_detail_path_no_control",
        "candidate_action_sha256",
        "training_admission_authorized",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f"fresh raw manifest missing columns: {missing}")
    if raw.empty:
        raise RuntimeError("fresh GAT input requires a non-empty raw manifest")
    if not raw["training_admission_authorized"].astype(bool).all():
        raise RuntimeError("fresh GAT input requires independently raw-admitted rows")

    states: list[dict] = []
    windows: list[dict] = []
    history_hashes: set[str] = set()
    for state_key, group in raw.groupby("state_key", sort=True):
        group = group.sort_values("candidate_action_sha256", kind="mergesort")
        if group["candidate_action_sha256"].astype(str).nunique() < 3:
            raise RuntimeError(f"state {state_key} has fewer than 3 candidate actions")
        for column in ("event_id", "rainfall_sha256", "split_group_key", "checkpoint_min"):
            if group[column].astype(str).nunique() != 1:
                raise RuntimeError(f"state {state_key} has inconsistent {column}")
        history_values = group["source_detail_path_no_control"].astype(str).unique().tolist()
        if len(history_values) != 1:
            raise RuntimeError(
                f"state {state_key} does not share exactly one causal No-control history source"
            )

        first = group.iloc[0]
        checkpoint = float(first["checkpoint_min"])
        if checkpoint < 120.0:
            raise RuntimeError(f"state {state_key} is below checkpoint gate")
        history = Path(history_values[0]).resolve()
        candidate = Path(str(first["source_detail_path_candidate"])).resolve()
        if not history.is_file() or not candidate.is_file():
            raise FileNotFoundError(f"missing fresh history/candidate detail for {state_key}")
        history_sha = sha256_file(history)
        history_hashes.add(history_sha)
        rainfall = str(first["split_group_key"])
        event = str(first["event_id"])
        for index in range(13):
            anchor = checkpoint - 60.0 + 5.0 * index
            windows.append(
                {
                    # Physical identity must follow file content, not a machine-specific
                    # absolute path. This keeps the causal-history lineage stable when
                    # the same authoritative trajectory is moved or mounted elsewhere.
                    "physical_identity_sha256": history_sha,
                    "detail_path": str(history),
                    "detail_sha256": history_sha,
                    "event_id": event,
                    "rainfall_sha256": str(first["rainfall_sha256"]),
                    "split_group_key": rainfall,
                    "anchor_min": anchor,
                    "history_start_min": anchor - 60.0,
                    "history_end_min": anchor,
                    "frame_count": 13,
                    "frame_interval_min": 5,
                    "action_authority": "actual_readback_setting",
                    "requested_action_fallback_allowed": False,
                    "target_authority": "full_network_SWMM_depth_truth",
                    "future_hydraulic_truth_in_input": False,
                    "source_role": "fresh_calibration_history",
                    "step1_domain_role": "target_formal",
                    "formal_generation_id": FORMAL_GENERATION_ID,
                    "formal_split": "fresh_calibration",
                }
            )
        states.append(
            {
                "formal_generation_id": FORMAL_GENERATION_ID,
                "development_only": False,
                "formal_mainline_authorized": False,
                "rainfall_sha256": str(first["rainfall_sha256"]),
                "rainfall_group_key": rainfall,
                "event_id": event,
                "state_key": str(state_key),
                "checkpoint_min": checkpoint,
                "candidate_detail_path": str(candidate),
                "history_detail_path": str(history),
                "history_detail_sha256": history_sha,
                "history_start_min": checkpoint - 120.0,
                "history_end_min": checkpoint,
                "history_match_level": "same_state_shared_no_control_history",
                "candidate_count": int(group["candidate_action_sha256"].astype(str).nunique()),
                "compatible": True,
                "failure_reason": "",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_path = args.output_dir / "FRESH_PFV_ONLY_STEP1_WINDOW_MANIFEST.parquet"
    source_path = args.output_dir / "FRESH_PFV_ONLY_HISTORY_SOURCE_MANIFEST.parquet"
    pd.DataFrame(windows).to_parquet(window_path, index=False)
    pd.DataFrame(states).to_parquet(source_path, index=False)
    audit = {
        "status": "pass",
        "formal_mainline_authorized": False,
        "raw_manifest": str(args.raw_manifest.resolve()),
        "raw_manifest_sha256": sha256_file(args.raw_manifest),
        "window_manifest": str(window_path.resolve()),
        "history_source_manifest": str(source_path.resolve()),
        "rainfall_groups": int(raw["split_group_key"].astype(str).nunique()),
        "states": int(len(states)),
        "window_rows": int(len(windows)),
        "unique_history_content_hashes": int(len(history_hashes)),
        "candidate_actions_min": int(min(row["candidate_count"] for row in states)),
        "history_identity_authority": "sha256_file_content",
        "history_source_is_candidate_outcome_source": False,
        "future_hydraulic_truth_in_input": False,
    }
    (args.output_dir / "FRESH_PFV_ONLY_GAT_INPUT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
