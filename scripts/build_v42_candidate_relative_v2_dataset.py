"""Build the bounded candidate-relative V2 Step-2 development dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.control.authoritative_control_metrics_v42 import rolling_pfv_budget_metric
from sewerrtc.control.experience_bank_v42 import decode_sequence, decode_signature
from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS


def _stable_rank(values: list[str], seed: int = 42) -> list[str]:
    # Must match the frozen Formal Step2 split authority exactly.
    return sorted(values, key=lambda x: (hashlib.sha256(f"formal-f2:{seed}:{x}".encode()).hexdigest(), x))


def _split(groups: list[str]) -> dict[str, str]:
    ranked = _stable_rank(groups)
    return {g: ("validation" if i < 8 else "holdout" if i < 16 else "train") for i, g in enumerate(ranked)}


def _finite_array(value: object, *, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(json.loads(str(value)) if isinstance(value, str) else value, dtype=np.float32)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"expected ndim={ndim}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite array")
    return arr


def _trajectory_summary(candidate: object, internal: object, storage: object | None, facility: object | None) -> tuple[list[float], list[bool]]:
    c_depth = _finite_array(candidate, ndim=2)
    i_depth = _finite_array(internal, ndim=2)
    if c_depth.shape != i_depth.shape:
        raise ValueError("depth residual shape mismatch")
    values = [
        (c_depth - i_depth).mean(axis=1),
        np.zeros(c_depth.shape[0], dtype=np.float32),
        np.zeros(c_depth.shape[0], dtype=np.float32),
        np.zeros(c_depth.shape[0], dtype=np.float32),
    ]
    masks = [True, False, False, False]
    if storage is not None and internal is not None:
        c = _finite_array(storage, ndim=2)
        # The caller supplies the internal storage separately through the
        # packed tuple; this branch is replaced in _target_fill_rows.
        values[2] = c.mean(axis=1)
        masks[2] = True
    if facility is not None:
        c = _finite_array(facility, ndim=2)
        values[3] = c.mean(axis=1)
        masks[3] = True
    return np.concatenate(values).astype(float).tolist(), masks


def _target_fill_labels(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for row in frame.itertuples(index=False):
        candidate_path = Path(str(row.source_detail_path_candidate))
        no_control_path = Path(str(row.source_detail_path_no_control))
        if not candidate_path.is_absolute():
            candidate_path = project_root / candidate_path
        if not no_control_path.is_absolute():
            no_control_path = project_root / no_control_path
        candidate = pd.read_csv(candidate_path)
        no_control = pd.read_csv(no_control_path)
        g = rolling_pfv_budget_metric(
            candidate,
            no_control,
            priority_nodes=PFV_CORE_8_IDS,
            checkpoint_min=float(row.checkpoint_min),
            relative_margin=0.05,
        )
        c_depth = _finite_array(row.trajectory_depth_candidate, ndim=2)
        i_depth = _finite_array(row.trajectory_depth_dynamic_internal, ndim=2)
        c_flood = _finite_array(row.trajectory_flood_candidate, ndim=2)
        i_flood = _finite_array(row.trajectory_flood_dynamic_internal, ndim=2)
        aux = [
            (c_depth - i_depth).mean(axis=1),
            (c_flood - i_flood).sum(axis=1),
        ]
        masks = [True, True]
        for c_name, i_name, key, pos in (
            ("trajectory_storage_volume_candidate", "trajectory_storage_volume_dynamic_internal", "storage", 2),
            ("trajectory_facility_flow_candidate", "trajectory_facility_flow_dynamic_internal", "facility", 3),
        ):
            c_value = getattr(row, c_name, None)
            i_value = getattr(row, i_name, None)
            available = bool(getattr(row, f"{key}_supervised_available", True))
            if c_value is not None and i_value is not None and available:
                c = _finite_array(c_value, ndim=2)
                i = _finite_array(i_value, ndim=2)
                aux.append((c - i).mean(axis=1))
                masks.append(True)
            else:
                aux.append(np.zeros(12, dtype=np.float32))
                masks.append(False)
        rows.append({
            "state_key": str(row.state_key),
            "rainfall_sha256": str(row.split_group_key),
            "candidate_action_sha256": str(row.candidate_action_sha256),
            "candidate_action_json": json.dumps(_finite_array(row.action_candidate_readback, ndim=2).tolist(), separators=(",", ":")),
            "state_signature_json": None,
            "g_pfv": float(g),
            "delta_tfv": float(row.tfv_delta),
            "source": "target_fill",
            "trajectory_aux_json": json.dumps(np.asarray(aux, dtype=np.float32).T.tolist(), separators=(",", ":")),
            "trajectory_aux_mask_json": json.dumps(masks, separators=(",", ":")),
        })
    return pd.DataFrame(rows)


def _pair_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = [_finite_array(v, ndim=2)[:3] for v in frame.candidate_action_json]
    action_h3 = np.stack([x.reshape(-1) for x in actions]).astype(np.float32)
    diverse: list[dict] = []
    local: list[dict] = []
    for state, positions in frame.groupby("state_key", sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        if len(idx) < 2:
            continue
        # Pair extremes and deterministic neighbours first, then fill with a
        # stable pseudo-random sample.  This keeps the dataset bounded.
        order_t = idx[np.argsort(frame.iloc[idx].delta_tfv.to_numpy(float))]
        order_g = idx[np.argsort(frame.iloc[idx].g_pfv.to_numpy(float))]
        seed = int(hashlib.sha256(str(state).encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        pairs: set[tuple[int, int]] = set()
        for a, b in zip(order_t[:8], order_t[-8:]):
            if a != b:
                pairs.add(tuple(sorted((int(a), int(b)))))
        for a, b in zip(order_g[:8], order_g[-8:]):
            if a != b:
                pairs.add(tuple(sorted((int(a), int(b)))))
        while len(pairs) < min(32, len(idx) * (len(idx) - 1) // 2):
            a, b = rng.choice(idx, size=2, replace=False)
            pairs.add(tuple(sorted((int(a), int(b)))))
        for a, b in sorted(pairs)[:32]:
            diverse.append({
                "row_i": a, "row_j": b, "state_key": str(state),
                "action_distance": float(np.mean(np.abs(action_h3[a] - action_h3[b]))),
                "pfv_diff": float(frame.iloc[b].g_pfv - frame.iloc[a].g_pfv),
                "tfv_diff": float(frame.iloc[b].delta_tfv - frame.iloc[a].delta_tfv),
                "local": False,
            })
        distances = np.mean(np.abs(action_h3[idx, None, :] - action_h3[None, idx, :]), axis=2)
        np.fill_diagonal(distances, np.inf)
        local_pairs: set[tuple[int, int]] = set()
        for pos in range(len(idx)):
            for nearest in np.argsort(distances[pos])[:3]:
                a, b = sorted((int(idx[pos]), int(idx[nearest])))
                if a != b:
                    local_pairs.add((a, b))
        for a, b in sorted(local_pairs)[:24]:
            local.append({
                "row_i": a, "row_j": b, "state_key": str(state),
                "action_distance": float(np.mean(np.abs(action_h3[a] - action_h3[b]))),
                "pfv_diff": float(frame.iloc[b].g_pfv - frame.iloc[a].g_pfv),
                "tfv_diff": float(frame.iloc[b].delta_tfv - frame.iloc[a].delta_tfv),
                "local": True,
            })
    return pd.DataFrame(diverse), pd.DataFrame(local)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--experience-bank", type=Path, required=True)
    ap.add_argument("--target-fill-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bank = pd.read_parquet(args.experience_bank, columns=[
        "state_key", "rainfall_sha256", "state_signature_json", "candidate_action_sha256",
        "candidate_action_json", "pfv_budget_metric_m3", "tfv_candidate_m3", "tfv_internal_m3",
    ])
    bank = bank.rename(columns={"pfv_budget_metric_m3": "g_pfv"})
    bank["delta_tfv"] = bank.tfv_candidate_m3 - bank.tfv_internal_m3
    bank["source"] = "experience_bank"
    bank["trajectory_aux_json"] = json.dumps(np.zeros((12, 4), dtype=np.float32).tolist(), separators=(",", ":"))
    bank["trajectory_aux_mask_json"] = json.dumps([False, False, False, False], separators=(",", ":"))
    bank = bank[["state_key", "rainfall_sha256", "candidate_action_sha256", "candidate_action_json", "state_signature_json", "g_pfv", "delta_tfv", "source", "trajectory_aux_json", "trajectory_aux_mask_json"]]

    target = pd.read_parquet(args.target_fill_manifest)
    target = _target_fill_labels(target, args.project_root)
    signatures = bank.drop_duplicates("state_key").set_index("state_key").state_signature_json.to_dict()
    target["state_signature_json"] = target.state_key.map(signatures)
    if target.state_signature_json.isna().any():
        raise RuntimeError("target-fill state has no causal experience-bank signature")
    target = target[["state_key", "rainfall_sha256", "candidate_action_sha256", "candidate_action_json", "state_signature_json", "g_pfv", "delta_tfv", "source", "trajectory_aux_json", "trajectory_aux_mask_json"]]
    bank_key_to_index = {
        (str(row.state_key), str(row.candidate_action_sha256)): int(index)
        for index, row in bank.iterrows()
    }
    target_fill_matched = 0
    target_fill_added = 0
    label_max_abs = 0.0
    for row in target.itertuples(index=False):
        key = (str(row.state_key), str(row.candidate_action_sha256))
        if key in bank_key_to_index:
            target_fill_matched += 1
            index = bank_key_to_index[key]
            label_max_abs = max(
                label_max_abs,
                abs(float(bank.loc[index, "g_pfv"]) - float(row.g_pfv)),
                abs(float(bank.loc[index, "delta_tfv"]) - float(row.delta_tfv)),
            )
            bank.loc[index, "trajectory_aux_json"] = row.trajectory_aux_json
            bank.loc[index, "trajectory_aux_mask_json"] = row.trajectory_aux_mask_json
            bank.loc[index, "source"] = "experience_bank+target_fill"
        else:
            target_fill_added += 1
    if label_max_abs > 1.0e-2:
        raise RuntimeError(f"target-fill labels disagree with experience bank: {label_max_abs}")
    target = target[[
        key not in bank_key_to_index for key in zip(
            target.state_key.astype(str), target.candidate_action_sha256.astype(str)
        )
    ]]
    frame = pd.concat([bank, target], ignore_index=True)
    frame["state_key"] = frame.state_key.astype(str)
    frame["rainfall_sha256"] = frame.rainfall_sha256.astype(str)
    split = _split(sorted(frame.rainfall_sha256.unique()))
    frame["split"] = frame.rainfall_sha256.map(split)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    frame["g_pfv"] = pd.to_numeric(frame.g_pfv, errors="coerce")
    frame["delta_tfv"] = pd.to_numeric(frame.delta_tfv, errors="coerce")
    if not np.isfinite(frame[["g_pfv", "delta_tfv"]].to_numpy(float)).all():
        raise RuntimeError("non-finite V2 labels")
    target_fill_nontrain = int(
        frame.source.astype(str).str.contains("target_fill")
        .where(frame.split.ne("train"), False)
        .sum()
    )
    if target_fill_nontrain:
        raise RuntimeError(f"target-fill auxiliary rows leaked outside train groups: {target_fill_nontrain}")

    diverse, local = _pair_rows(frame)
    frame.to_parquet(out / "STATE_ACTION_GROUP_DATASET.parquet", index=False)
    diverse.to_parquet(out / "SAME_STATE_ACTION_PAIRS.parquet", index=False)
    local.to_parquet(out / "LOCAL_ACTION_EFFECT_PAIRS.parquet", index=False)

    group_sizes = frame.groupby("state_key").size().to_numpy(float)
    safe = frame.g_pfv <= 100.0 + 1.0e-9
    improving = frame.delta_tfv < 0.0
    summary = {
        "dataset_contract": "CANDIDATE_RELATIVE_DIFFERENTIABLE_CONTROL_SURROGATE_V2",
        "development_only": True,
        "formal_mainline_authorized": False,
        "source_experience_rows": int(len(bank)),
        "source_target_fill_rows_added": int(target_fill_added),
        "target_fill_rows_matched_for_auxiliary_supervision": int(target_fill_matched),
        "target_fill_label_max_abs_disagreement": float(label_max_abs),
        "rows": int(len(frame)),
        "states": int(frame.state_key.nunique()),
        "rainfall_groups": int(frame.rainfall_sha256.nunique()),
        "split_groups": {k: int(frame.loc[frame.split.eq(k), "rainfall_sha256"].nunique()) for k in ("train", "validation", "holdout")},
        "rows_by_split": {k: int((frame.split == k).sum()) for k in ("train", "validation", "holdout")},
        "candidates_per_state": {k: float(v) for k, v in {"min": np.min(group_sizes), "p10": np.percentile(group_sizes, 10), "median": np.median(group_sizes), "p90": np.percentile(group_sizes, 90), "max": np.max(group_sizes)}.items()},
        "same_state_pair_count": int(len(diverse)),
        "local_neighbour_pair_count": int(len(local)),
        "pfv_g_distribution": {k: float(v) for k, v in {"min": frame.g_pfv.min(), "p01": frame.g_pfv.quantile(.01), "median": frame.g_pfv.median(), "p99": frame.g_pfv.quantile(.99), "max": frame.g_pfv.max()}.items()},
        "tfv_delta_distribution": {k: float(v) for k, v in {"min": frame.delta_tfv.min(), "p01": frame.delta_tfv.quantile(.01), "median": frame.delta_tfv.median(), "p99": frame.delta_tfv.quantile(.99), "max": frame.delta_tfv.max()}.items()},
        "pfv_safe_fraction": float(safe.mean()),
        "tfv_improving_fraction": float(improving.mean()),
        "trajectory_aux_rows": int(frame.source.astype(str).str.contains("target_fill").sum()),
        "trajectory_aux_available_rows": int(sum(json.loads(x)[0] for x in frame.trajectory_aux_mask_json)),
        "target_fill_nontrain_rows": target_fill_nontrain,
        "input_contract": "causal_state_signature_plus_candidate_current_no_control_internal_action_differences",
        "future_hydraulic_truth_as_input": False,
        "realized_future_rainfall_as_input": False,
        "files": {"rows": str(out / "STATE_ACTION_GROUP_DATASET.parquet"), "pairs": str(out / "SAME_STATE_ACTION_PAIRS.parquet"), "local_pairs": str(out / "LOCAL_ACTION_EFFECT_PAIRS.parquet")},
    }
    (out / "DATASET_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
