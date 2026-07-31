from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _correct(spec: dict, phase: str) -> dict:
    out = dict(spec)
    kind = str(out.get("kind", ""))
    mode = str(out.get("mode", ""))
    actuators = list(out.get("actuators", []))
    if kind == "single_continuous":
        if mode == "small_ramp":
            out.update(mode="ramp_restrict_restore", signed_profile=[-0.05, -0.10, -0.10, -0.10, 0.0, 0.0])
        else:
            out.update(mode="delayed_release_restore", signed_profile=[0.0, 0.0, 0.20, 0.20, 0.0, 0.0])
    elif kind == "single_binary":
        target = 1.0 if mode == "off_to_on_hold" else 0.0
        out["target_profile"] = [target, target, target, target, target, target]
    elif kind == "add350_continuous_profile":
        profiles = {
            "ramp_up": [0.05, 0.10, 0.15, 0.20, 0.10, 0.0],
            "ramp_down": [-0.05, -0.10, -0.15, -0.20, -0.10, 0.0],
            "hold_then_release": [-0.10, -0.10, -0.10, 0.10, 0.10, 0.0],
        }
        out["signed_profile"] = profiles[mode]
    elif kind == "storage_interlock":
        number = str(actuators[0]).rsplit("_", 1)[-1]
        inlet, outlet = f"RTC_IN_{number}", f"RTC_OUT_{number}"
        out["actuators"] = [inlet, outlet]
        if str(actuators[0]).startswith("RTC_IN"):
            out.update(
                mode="retain_then_restore",
                signed_profiles={inlet: [0.10, 0.20, 0.20, 0.0, 0.0, 0.0], outlet: [-0.10, -0.20, -0.20, 0.0, 0.0, 0.0]},
            )
        else:
            out.update(
                mode="delayed_release",
                signed_profiles={inlet: [0.0, 0.0, -0.10, -0.10, 0.0, 0.0], outlet: [0.0, 0.0, 0.10, 0.10, 0.0, 0.0]},
            )
    elif kind == "hydraulic_pair":
        sign = -1.0 if phase in {"rising", "peak"} else 1.0
        out["signed_profiles"] = {
            actuators[0]: [0.10 * sign] * 4 + [0.0, 0.0],
            actuators[1]: [-0.10 * sign] * 4 + [0.0, 0.0],
        }
        out["mode"] = "paired_opposite_direction_restore"
    out["sequence_semantics"] = "relative_to_same_state_no_control_reference"
    out["horizon_steps"] = 6
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/project6_36_fulltrain_v1/joint_data_plan/joint_action_case_manifest.csv")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v1/paired_plan")
    args = parser.parse_args()
    source = Path(args.source)
    frame = pd.read_csv(source)
    corrected_by_pair: dict[str, dict] = {}
    for row in frame[frame["branch"].eq("B")].itertuples(index=False):
        corrected_by_pair[str(row.pair_id)] = _correct(json.loads(row.executed_action_sequence), str(row.phase))
    rows = []
    for row in frame.itertuples(index=False):
        candidate = corrected_by_pair[str(row.pair_id)]
        executed = {"mode": "default_no_control", "horizon_steps": 6} if row.branch == "A" else candidate
        record = row._asdict()
        record["source_case_id"] = record["case_id"]
        record["candidate_action_sequence"] = json.dumps(candidate, sort_keys=True)
        record["executed_action_sequence"] = json.dumps(executed, sort_keys=True)
        record["case_id"] = _hash({"source": record["source_case_id"], "executed": executed, "schema": "temporal_joint_v2"})
        record["execution_case_id"] = _hash({"event": record["event_id"], "split": record["split_timestamp_fraction"], "branch": record["branch"], "executed": executed, "schema": "temporal_joint_v2"})
        record["status"] = "validated_plan_not_started"
        rows.append(record)
    result = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "joint_action_case_manifest.csv", index=False)
    candidates = result[result["branch"].eq("B")]
    report = {
        "source_manifest": str(source),
        "logical_rows": len(result),
        "paired_experiments": int(candidates["pair_id"].nunique()),
        "unique_physical_cases": int(result["execution_case_id"].nunique()),
        "candidate_kind_counts": candidates["executed_action_sequence"].map(lambda value: json.loads(value)["kind"]).value_counts().to_dict(),
        "all_candidates_have_explicit_sequence_semantics": bool(candidates["executed_action_sequence"].map(lambda value: "sequence_semantics" in json.loads(value)).all()),
        "old_manifest_modified": False,
        "execution": "plan_only_no_swmm_started",
    }
    (out / "manifest_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
