"""Content-based resolver for causal GAT history sources."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_step1_dataset import _detail_extract_window


_EVALUATION_SPLITS = {
    "calibration",
    "challenge",
    "formal_blind",
    "locked_validation",
    "reserved_evaluation",
}


def _read_table(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _normal_text(value: Any) -> str:
    text = str(value)
    return "" if text.lower() in {"", "nan", "none", "nat"} else text


def build_history_index(manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Build a compact path-level catalog from the full Step1 window manifest."""
    frame = _read_table(manifest)
    required = {"split_group_key", "detail_path", "history_start_min", "history_end_min"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"history manifest missing columns: {missing}")
    frame = frame.copy()
    if "formal_split" in frame:
        split = frame["formal_split"].astype(str).str.lower()
        frame = frame.loc[~split.isin(_EVALUATION_SPLITS)].copy()
    if "step1_domain_role" in frame:
        role = frame["step1_domain_role"].astype(str)
        frame = frame.loc[role.isin({"target_formal", "auxiliary_pretrain"})].copy()
    frame["split_group_key"] = frame["split_group_key"].map(_normal_text)
    frame["detail_path"] = frame["detail_path"].map(_normal_text)
    frame["event_id"] = frame.get("event_id", "").map(_normal_text)
    frame["history_start_min"] = pd.to_numeric(frame["history_start_min"], errors="coerce")
    frame["history_end_min"] = pd.to_numeric(frame["history_end_min"], errors="coerce")
    frame = frame[
        frame["split_group_key"].astype(bool)
        & frame["detail_path"].astype(bool)
        & np.isfinite(frame["history_start_min"])
        & np.isfinite(frame["history_end_min"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "split_group_key",
                "event_id",
                "detail_path",
                "history_start_min",
                "history_end_min",
            ]
        )
    keys = ["split_group_key", "event_id", "detail_path"]
    return (
        frame.groupby(keys, dropna=False, as_index=False)
        .agg(
            history_start_min=("history_start_min", "min"),
            history_end_min=("history_end_min", "max"),
        )
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )


def pre_action_signature_components(detail: pd.DataFrame, checkpoint_min: float, graph: Any) -> dict[str, np.ndarray]:
    extracted = _detail_extract_window(detail, float(checkpoint_min), graph.node_ids, graph.facility_ids)
    if extracted is None:
        raise ValueError("detail cannot reconstruct Step1 window at checkpoint")
    return {
        "checkpoint_depth": np.ascontiguousarray(extracted["depth_history"][-1], dtype=np.float64),
        "rainfall_history": np.ascontiguousarray(extracted["rainfall"], dtype=np.float64),
        "pre_action_history": np.ascontiguousarray(extracted["actions"][:-1], dtype=np.float64),
    }


def compare_pre_action_signatures(
    candidate: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
    *,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    names = (
        ("checkpoint_depth", "checkpoint_depth_mismatch"),
        ("rainfall_history", "rainfall_history_mismatch"),
        ("pre_action_history", "pre_action_history_mismatch"),
    )
    result: dict[str, Any] = {}
    mismatch = False
    for key, flag in names:
        left = np.asarray(candidate[key], dtype=np.float64)
        right = np.asarray(history[key], dtype=np.float64)
        same_shape = left.shape == right.shape
        error = float(np.max(np.abs(left - right))) if same_shape and left.size else float("inf")
        bad = not same_shape or not np.allclose(left, right, atol=atol, rtol=0.0)
        result[flag] = bool(bad)
        result[f"{key}_max_abs_error"] = error
        mismatch = mismatch or bad
    result["compatible"] = not mismatch
    return result


def _load_detail(
    load_detail: Callable[..., pd.DataFrame],
    path: str,
    start_min: float | None = None,
    end_min: float | None = None,
) -> pd.DataFrame:
    try:
        return load_detail(Path(path), start_min, end_min)
    except (KeyError, TypeError):
        return load_detail(path)  # type: ignore[arg-type]


def _history_candidates(index: pd.DataFrame, rainfall_group: str, event_id: str, checkpoint_min: float) -> pd.DataFrame:
    subset = index[index["split_group_key"].astype(str).eq(str(rainfall_group))].copy()
    subset = subset[
        (subset["history_start_min"] <= float(checkpoint_min) - 120.0 + 1.0e-6)
        & (subset["history_end_min"] >= float(checkpoint_min) - 1.0e-6)
    ].copy()
    if subset.empty:
        return subset
    event_id = _normal_text(event_id)
    subset["match_level"] = np.where(
        event_id and subset["event_id"].astype(str).eq(event_id),
        "same_event",
        "same_rainfall_verified_prefix",
    )
    subset["match_rank"] = np.where(subset["match_level"].eq("same_event"), 0, 1)
    return subset.sort_values(
        ["match_rank", "history_start_min", "detail_path"], kind="mergesort"
    ).drop_duplicates("detail_path", keep="first")


def resolve_compatible_history(
    *,
    history_index: pd.DataFrame,
    rainfall_group: str,
    event_id: str,
    checkpoint_min: float,
    candidate_signature: dict[str, np.ndarray],
    load_detail: Callable[[Path], pd.DataFrame],
    graph: Any,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    candidates = _history_candidates(history_index, rainfall_group, event_id, checkpoint_min)
    diagnostics: list[dict[str, Any]] = []
    for row in candidates.to_dict("records"):
        path = str(row["detail_path"])
        try:
            detail = _load_detail(
                load_detail,
                path,
                float(checkpoint_min) - 120.0,
                float(checkpoint_min),
            )
            comparison = compare_pre_action_signatures(
                candidate_signature,
                pre_action_signature_components(detail, checkpoint_min, graph),
                atol=atol,
            )
            if not comparison["compatible"]:
                diagnostics.append({"path": path, "failure_reason": "prefix_signature_mismatch", **comparison})
                continue
            for anchor in (float(checkpoint_min) - 60.0 + 5.0 * i for i in range(13)):
                if _detail_extract_window(detail, anchor, graph.node_ids, graph.facility_ids) is None:
                    raise ValueError(f"missing_gat_anchor:{anchor:.6f}")
            return {
                "compatible": True,
                "history_detail_path": path,
                "history_event_id": _normal_text(row.get("event_id", "")),
                "history_start_min": float(row["history_start_min"]),
                "history_end_min": float(row["history_end_min"]),
                "history_match_level": str(row["match_level"]),
                "candidate_pre_action_signature": candidate_signature,
                "history_pre_action_signature": pre_action_signature_components(detail, checkpoint_min, graph),
                "comparison": comparison,
                "diagnostics": diagnostics,
            }
        except Exception as exc:
            diagnostics.append({"path": path, "failure_reason": "history_validation_failed", "error": str(exc)})
    if not candidates.empty:
        start = candidates["history_start_min"].min()
        reason = "prefix_signature_mismatch" if diagnostics else "history_validation_failed"
    else:
        start = None
        reason = "coverage_start_too_late"
    return {
        "compatible": False,
        "failure_reason": reason,
        "required_history_start_min": float(checkpoint_min) - 120.0,
        "best_history_start_min": None if pd.isna(start) else float(start),
        "diagnostics": diagnostics,
    }


def _candidate_action_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "candidate_action_sha256",
        "qualification_candidate_action_sha256",
        "actual_candidate_action_sha256",
    ):
        if column in frame:
            values = frame[column].map(_normal_text)
            if values.astype(bool).any():
                return column
    return None


def choose_history_compatible_state(
    group: pd.DataFrame,
    *,
    history_index: pd.DataFrame,
    load_detail: Callable[[Path], pd.DataFrame],
    graph: Any,
    required_candidates: int = 3,
    min_checkpoint_min: float = 120.0,
) -> dict[str, Any] | None:
    failures: list[dict[str, Any]] = []
    for state_key, state in group.groupby("state_key", sort=True):
        checkpoints = pd.to_numeric(state.get("checkpoint_min"), errors="coerce").dropna()
        if checkpoints.empty or checkpoints.nunique() != 1 or float(checkpoints.iloc[0]) < min_checkpoint_min:
            continue
        action_column = _candidate_action_column(state)
        if action_column is None:
            failures.append({"state_key": str(state_key), "failure_reason": "missing_actual_candidate_action"})
            continue
        state = state.copy()
        state[action_column] = state[action_column].map(_normal_text)
        state = state[state[action_column].astype(bool)].drop_duplicates(action_column, keep="first")
        if len(state) < required_candidates:
            failures.append({"state_key": str(state_key), "failure_reason": "fewer_than_required_candidates"})
            continue
        first = state.iloc[0]
        candidate_path = str(first.get("source_detail_path_candidate", ""))
        if not candidate_path:
            failures.append({"state_key": str(state_key), "failure_reason": "missing_candidate_detail"})
            continue
        checkpoint = float(checkpoints.iloc[0])
        try:
            candidate_signature = pre_action_signature_components(
                _load_detail(load_detail, candidate_path, checkpoint - 60.0, checkpoint), checkpoint, graph
            )
            history = resolve_compatible_history(
                history_index=history_index,
                rainfall_group=str(first["split_group_key"]),
                event_id=_normal_text(first.get("event_id", "")),
                checkpoint_min=checkpoint,
                candidate_signature=candidate_signature,
                load_detail=load_detail,
                graph=graph,
            )
        except Exception as exc:
            history = {"compatible": False, "failure_reason": "candidate_history_unreadable", "error": str(exc)}
        if history.get("compatible"):
            selected = state.sort_values(action_column, kind="mergesort").head(required_candidates).copy()
            selected["qualification_candidate_action_sha256"] = selected[action_column].astype(str)
            return {
                "compatible": True,
                "state_key": str(state_key),
                "state": selected,
                "history": history,
                "failure_diagnostics": failures,
            }
        failures.append({"state_key": str(state_key), **history})
    if not failures:
        return None
    return {"compatible": False, "failure_reason": "no_history_compatible_state", "failures": failures}
