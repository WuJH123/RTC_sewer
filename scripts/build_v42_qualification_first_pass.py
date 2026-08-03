"""Build a small, diverse, development-only V4.2 qualification population.

The qualification pass exists to exercise the complete 28-stage software chain
before the expensive Formal production run.  It reuses already admitted Formal
F2 *development* assets, never consumes untouched Formal evaluation rainfalls,
and writes to an isolated output root.  Qualification artifacts can diagnose
wiring/runtime problems but can never authorize Formal paper evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QUALIFICATION_CONTRACT = "PROJECT6_V42_QUALIFICATION_FIRST_PASS_V1"
RAW_REQUIRED_GATES = (
    "training_admission_authorized",
    "raw_independent_oracle_all_pass",
    "same_state_raw_verified",
    "same_forcing_raw_verified",
    "actual_readback_verified",
    "h120_window_complete",
    "kpi_recompute_ok",
)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"input table is empty: {path}")
    return frame


def _rank(values: list[str], seed: int, salt: str) -> list[str]:
    return sorted(
        {str(v) for v in values},
        key=lambda value: (hashlib.sha256(f"{salt}:{seed}:{value}".encode()).hexdigest(), value),
    )


def _spread(frame: pd.DataFrame, limit: int, *, order_columns: list[str]) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    ordered = frame.sort_values(order_columns, kind="mergesort").reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, num=limit, dtype=int)
    return ordered.iloc[np.unique(indices)].copy()


def _cap_step1(
    frame: pd.DataFrame,
    *,
    windows_per_run: int,
    windows_per_group: int,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, run in frame.groupby("physical_identity_sha256", sort=True):
        pieces.append(_spread(run, windows_per_run, order_columns=["anchor_min", "detail_path"]))
    capped = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()
    pieces = []
    for _, group in capped.groupby("split_group_key", sort=True):
        pieces.append(_spread(group, windows_per_group, order_columns=["anchor_min", "detail_path"]))
    return pd.concat(pieces, ignore_index=True) if pieces else capped.iloc[0:0].copy()


def _action_hash(row: pd.Series) -> str:
    for key in ("candidate_action_sha256", "actual_candidate_action_sha256", "h3_action_sha256"):
        value = str(row.get(key, "")).strip()
        if value and value.lower() != "nan":
            return value
    raw = row.get("action_candidate_readback")
    try:
        array = np.asarray(json.loads(str(raw)), dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if array.ndim == 1 and array.size % 36 == 0:
        array = array.reshape(-1, 36)
    if array.ndim != 2 or array.shape[1] != 36:
        return ""
    h3 = np.ascontiguousarray(array[:3], dtype=np.float64)
    return hashlib.sha256(h3.tobytes()).hexdigest()


def _bool_all(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and bool(frame[column].fillna(False).astype(bool).all())


def _choose_step2_state(group: pd.DataFrame, candidates: int, seed: int) -> pd.DataFrame | None:
    viable: list[tuple[str, pd.DataFrame]] = []
    for state_key, state in group.groupby("state_key", sort=True):
        state = state.copy()
        state["qualification_candidate_action_sha256"] = state.apply(_action_hash, axis=1)
        state = state[state["qualification_candidate_action_sha256"].astype(bool)].copy()
        state = state.drop_duplicates("qualification_candidate_action_sha256", keep="first")
        if len(state) >= candidates:
            viable.append((str(state_key), state))
    if not viable:
        return None
    state_key, state = sorted(
        viable,
        key=lambda item: hashlib.sha256(f"qualification-state:{seed}:{item[0]}".encode()).hexdigest(),
    )[0]
    state = state.sort_values("qualification_candidate_action_sha256", kind="mergesort").head(candidates).copy()
    state["qualification_selected_state_key"] = state_key
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/v42_qualification_first_pass.json",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = config["selection"]
    seed = int(selection["seed"])
    root = args.project_root
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    out = args.output_root or (
        root / str(config["output_relative_root"])
    )
    out.mkdir(parents=True, exist_ok=True)

    step1_path = formal / "prepare/FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet"
    raw_path = formal / "step2/FORMAL_F2_STEP2_RAW_MANIFEST.parquet"
    step1 = _read(step1_path)
    raw = _read(raw_path)

    step1_required = {
        "detail_path",
        "anchor_min",
        "split_group_key",
        "physical_identity_sha256",
        "step1_domain_role",
        "formal_split",
    }
    missing = sorted(step1_required - set(step1.columns))
    if missing:
        raise KeyError(f"Formal Step1 manifest missing required columns: {missing}")

    target_train = step1[
        step1["step1_domain_role"].astype(str).eq("target_formal")
        & step1["formal_split"].astype(str).eq("train")
    ].copy()
    target_validation = step1[
        step1["step1_domain_role"].astype(str).eq("target_formal")
        & step1["formal_split"].astype(str).eq("validation")
    ].copy()
    auxiliary = step1[step1["step1_domain_role"].astype(str).eq("auxiliary_pretrain")].copy()

    train_group_count = int(selection["step1_train_rainfall_groups"])
    validation_group_count = int(selection["step1_validation_rainfall_groups"])
    train_groups = _rank(target_train["split_group_key"].astype(str).unique().tolist(), seed, "step1-train")[:train_group_count]
    validation_groups = _rank(target_validation["split_group_key"].astype(str).unique().tolist(), seed, "step1-validation")[:validation_group_count]
    if len(train_groups) < train_group_count:
        raise RuntimeError(f"qualification Step1 has only {len(train_groups)} train groups; required {train_group_count}")
    if len(validation_groups) < validation_group_count:
        raise RuntimeError(
            f"qualification Step1 has only {len(validation_groups)} validation groups; required {validation_group_count}"
        )

    target_train = target_train[target_train["split_group_key"].astype(str).isin(train_groups)].copy()
    target_validation = target_validation[target_validation["split_group_key"].astype(str).isin(validation_groups)].copy()
    selected_step1 = pd.concat([target_train, target_validation, auxiliary], ignore_index=True)
    selected_step1 = _cap_step1(
        selected_step1,
        windows_per_run=int(selection["step1_windows_per_physical_run"]),
        windows_per_group=int(selection["step1_windows_per_group"]),
    )
    selected_step1["qualification_only"] = True
    selected_step1["development_only"] = True
    selected_step1["formal_mainline_authorized"] = False
    selected_step1["qualification_contract_id"] = QUALIFICATION_CONTRACT
    selected_step1_path = out / "QUALIFICATION_STEP1_WINDOW_MANIFEST.parquet"
    selected_step1.to_parquet(selected_step1_path, index=False)

    raw_required = {"split_group_key", "state_key", "event_id"}
    missing = sorted(raw_required - set(raw.columns))
    if missing:
        raise KeyError(f"Formal Raw Step2 manifest missing required columns: {missing}")
    failed_gates = [column for column in RAW_REQUIRED_GATES if not _bool_all(raw, column)]
    if failed_gates:
        raise RuntimeError(f"Formal Raw Step2 input is not fully admitted: {failed_gates}")

    candidates_per_state = int(selection["step2_candidates_per_state"])
    eligible_groups: dict[str, pd.DataFrame] = {}
    for group_key, group in raw.groupby("split_group_key", sort=True):
        chosen = _choose_step2_state(group, candidates_per_state, seed)
        if chosen is not None:
            eligible_groups[str(group_key)] = chosen

    required_step2_groups = int(selection["step2_rainfall_groups"])
    ranked_groups = _rank(list(eligible_groups), seed, "step2-qualification")
    train_step2_groups = ranked_groups[:required_step2_groups]
    if len(train_step2_groups) < required_step2_groups:
        raise RuntimeError(
            f"qualification Step2 has only {len(train_step2_groups)} viable groups; required {required_step2_groups}"
        )
    selected_step2 = pd.concat([eligible_groups[group] for group in train_step2_groups], ignore_index=True)
    selected_step2["qualification_only"] = True
    selected_step2["development_only"] = True
    selected_step2["formal_mainline_authorized"] = False
    selected_step2["qualification_contract_id"] = QUALIFICATION_CONTRACT
    selected_step2["qualification_outfall_supervision_deferred"] = True
    selected_step2_path = out / "QUALIFICATION_STEP2_RAW_MANIFEST.parquet"
    selected_step2.to_parquet(selected_step2_path, index=False)

    remaining_groups = [group for group in ranked_groups if group not in set(train_step2_groups)]
    requested_eval = {
        "qualification_calibration": int(selection["qualification_calibration_groups"]),
        "qualification_challenge": int(selection["qualification_challenge_groups"]),
        "qualification_locked": int(selection["qualification_locked_groups"]),
        "qualification_blind": int(selection["qualification_blind_groups"]),
    }
    if len(remaining_groups) < sum(requested_eval.values()):
        raise RuntimeError(
            "not enough revealed development rainfall groups remain for isolated qualification evaluation"
        )
    eval_rows: list[dict[str, Any]] = []
    offset = 0
    for role, count in requested_eval.items():
        groups = remaining_groups[offset : offset + count]
        offset += count
        for group in groups:
            source = eligible_groups[group].iloc[0]
            eval_rows.append(
                {
                    "qualification_contract_id": QUALIFICATION_CONTRACT,
                    "qualification_role": role,
                    "rainfall_sha256": group,
                    "event_id": str(source.get("event_id", "")),
                    "state_key": str(source.get("state_key", "")),
                    "checkpoint_min": float(source.get("checkpoint_min", math.nan)),
                    "qualification_only": True,
                    "development_only": True,
                    "formal_untouched_event": False,
                    "formal_mainline_authorized": False,
                }
            )
    evaluation = pd.DataFrame(eval_rows)
    evaluation_path = out / "QUALIFICATION_DEVELOPMENT_EVALUATION_PLAN.csv"
    evaluation.to_csv(evaluation_path, index=False)

    step1_group_counts = selected_step1.groupby(["formal_split", "step1_domain_role"])["split_group_key"].nunique().to_dict()
    audit = {
        "contract_id": QUALIFICATION_CONTRACT,
        "status": "pass",
        "qualification_only": True,
        "development_only": True,
        "formal_mainline_authorized": False,
        "formal_outputs_overwritten": False,
        "source_formal_step1_manifest": str(step1_path),
        "source_formal_raw_step2_manifest": str(raw_path),
        "step1_manifest": str(selected_step1_path),
        "step1_rows": int(len(selected_step1)),
        "step1_train_groups": int(step1_group_counts.get(("train", "target_formal"), 0)),
        "step1_validation_groups": int(step1_group_counts.get(("validation", "target_formal"), 0)),
        "step1_auxiliary_groups": int(
            selected_step1.loc[selected_step1["step1_domain_role"].astype(str).eq("auxiliary_pretrain"), "split_group_key"].nunique()
        ),
        "step2_manifest": str(selected_step2_path),
        "step2_rows": int(len(selected_step2)),
        "step2_rainfall_groups": int(selected_step2["split_group_key"].astype(str).nunique()),
        "step2_states": int(selected_step2["state_key"].astype(str).nunique()),
        "step2_candidates_per_state_min": int(selected_step2.groupby("state_key").size().min()),
        "qualification_evaluation_plan": str(evaluation_path),
        "qualification_evaluation_group_counts": evaluation.groupby("qualification_role")["rainfall_sha256"].nunique().to_dict(),
        "formal_untouched_events_consumed": False,
        "deferred_formal_blockers": [
            "explicit_outfall_flow_supervision",
            "full Formal multi-seed production training",
            "new untouched Calibration/Locked/Challenge/Formal-Blind evidence",
        ],
        "next": "run qualification Step1 seeds, causal GAT bridge, qualification Step2 seeds, then micro authoritative closed-loop stages",
    }
    (out / "QUALIFICATION_PREPARE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
