"""Offline lineage reconciliation for the fresh PFV-only Calibration pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.v4 import v42_formal_runtime as base_runtime
from sewerrtc.v4 import v42_pfv_tfv_runtime_patch as production
from sewerrtc.control.action_sequence_generator import generate_action_sequences


def _sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.round(np.asarray(value, dtype=np.float32), 6))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _json_array(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def _production_pool(
    base: np.ndarray, actuators: pd.DataFrame, influence: pd.DataFrame, cap: int
) -> tuple[list[tuple[str, np.ndarray, object, int, bool]], dict[str, int]]:
    reference = np.repeat(
        np.asarray(base, dtype=np.float32)[None, :],
        base_runtime.HORIZON_STEPS,
        axis=0,
    )
    raw = production._global_tfv_sequences(base, actuators)
    raw.extend(
        generate_action_sequences(
            base,
            actuators,
            base_runtime.HORIZON_STEPS,
            max_delta=production.GLOBAL_SINGLE_DELTA,
            include_hold=True,
            max_sequences=0,
            group_limit=8,
            reference_sequence=reference,
            priority_to_actuators=influence,
        )
    )
    return production._project_dedupe_and_cap(
        raw, base=base, actuators=actuators, requested_cap=cap
    )


def main() -> int:
    root = ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=root)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2/FRESH_PFV_ONLY_GAT_MANIFEST.parquet",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2/CALIBRATION12_PRODUCTION_CANDIDATE_RECONCILIATION.json",
    )
    ap.add_argument("--requested-cap", type=int, default=64)
    args = ap.parse_args()

    frame = pd.read_parquet(args.manifest)
    required = {
        "state_key",
        "action_candidate_readback",
        "action_dynamic_internal_readback",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"fresh GAT manifest missing {missing}")
    actuators = base_runtime.load_actuators(args.project_root)
    graph = base_runtime.load_graph_assets(args.project_root)
    influence = base_runtime._build_priority_influence_map(
        args.project_root, graph, actuators
    )
    rows = []
    for state_key, group in frame.groupby("state_key", sort=True):
        group = group.sort_values("case_id")
        existing: dict[str, str] = {}
        for _, row in group.iterrows():
            sequence = _json_array(row["action_candidate_readback"])
            if sequence.ndim != 2 or sequence.shape[1] != len(actuators):
                raise ValueError(f"{state_key}: invalid candidate action shape {sequence.shape}")
            existing[_sha(sequence[: base_runtime.CONTROLLABLE_PREFIX_STEPS])] = str(
                row["case_id"]
            )
        base = _json_array(group.iloc[0]["action_dynamic_internal_readback"])[0]
        projected, stats = _production_pool(
            base, actuators, influence, int(args.requested_cap)
        )
        production_hashes = {
            _sha(item[1][: base_runtime.CONTROLLABLE_PREFIX_STEPS]) for item in projected
        }
        existing_hashes = set(existing)
        rows.append(
            {
                "state_key": str(state_key),
                "event_id": str(group.iloc[0]["event_id"]),
                "existing_candidate_rows": int(len(group)),
                "existing_distinct_h3": int(len(existing_hashes)),
                "production_raw_candidate_count": int(stats["raw_candidate_count"]),
                "production_projected_unique_candidate_count": int(
                    stats["projected_unique_candidate_count"]
                ),
                "requested_candidate_cap": int(stats["requested_candidate_cap"]),
                "effective_candidate_cap": int(stats["effective_candidate_cap"]),
                "exact_h3_action_matches": int(len(existing_hashes & production_hashes)),
                "unmatched_production_candidates": int(
                    len(production_hashes - existing_hashes)
                ),
                "old_only_candidates": int(len(existing_hashes - production_hashes)),
                "existing_h3_sha256": sorted(existing_hashes),
                "production_h3_sha256": sorted(production_hashes),
            }
        )

    total_existing = sum(int(row["existing_candidate_rows"]) for row in rows)
    total_matches = sum(int(row["exact_h3_action_matches"]) for row in rows)
    payload = {
        "status": "pass",
        "formal_mainline_authorized": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": base_runtime.sha256_file(args.manifest),
        "states": len(rows),
        "existing_candidate_rows": total_existing,
        "exact_h3_action_matches": total_matches,
        "unmatched_production_candidates": sum(
            int(row["unmatched_production_candidates"]) for row in rows
        ),
        "old_only_candidates": sum(int(row["old_only_candidates"]) for row in rows),
        "hash_mismatches": sum(
            int(row["unmatched_production_candidates"])
            + int(row["old_only_candidates"])
            for row in rows
        ),
        "requested_candidate_cap": int(args.requested_cap),
        "candidate_generator_sha256": base_runtime.sha256_file(
            root / "sewerrtc/control/action_sequence_generator.py"
        ),
        "global_search_adapter_sha256": base_runtime.sha256_file(
            root / "sewerrtc/v4/v42_pfv_tfv_runtime_patch.py"
        ),
        "engineering_projector_sha256": base_runtime.sha256_file(
            root / "sewerrtc/v4/v42_formal_runtime.py"
        ),
        "selector_sha256": base_runtime.sha256_file(
            root / "sewerrtc/control/pfvfirst_mpc_v42.py"
        ),
        "by_state": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
