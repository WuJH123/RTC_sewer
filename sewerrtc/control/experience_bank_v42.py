"""Authoritative state-action experience bank for V4.2 hybrid MPC.

The bank is development/training infrastructure.  It stores only causal state
features available at the decision time plus action sequences and outcomes
recomputed from authoritative SWMM detail files.  It is used to warm-start the
online differentiable search; it is never an online source of future truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256


EXPERIENCE_BANK_CONTRACT = "V42_AUTHORITATIVE_EXPERIENCE_BANK_V1"


def _array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


def state_signature(
    *,
    state_history: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
) -> np.ndarray:
    """Return a compact causal signature for nearest-state warm-start retrieval.

    The signature deliberately uses distributional summaries rather than node
    identities so it can be computed for every state without exposing any
    future hydraulic truth.  The differentiable model still receives the full
    graph state; this signature is only a retrieval index.
    """
    history = np.asarray(state_history, dtype=float)
    if history.ndim == 3 and history.shape[-1] == 1:
        history = history[..., 0]
    if history.ndim != 2:
        raise ValueError("state_history must be [T,N] or [T,N,1]")
    rain = np.asarray(rainfall_forecast, dtype=float).reshape(-1)
    action = np.asarray(current_action, dtype=float).reshape(-1)
    if not np.isfinite(history).all() or not np.isfinite(rain).all() or not np.isfinite(action).all():
        raise ValueError("state signature inputs must be finite")
    current = history[-1]
    early = history[max(0, len(history) - 3)]
    trend = current - early
    q = np.quantile(current, [0.25, 0.50, 0.75, 0.90, 0.99])
    tq = np.quantile(trend, [0.25, 0.50, 0.75, 0.90])
    rain_h3 = rain[:3]
    rain_h6 = rain[:6]
    features = np.asarray(
        [
            float(np.mean(current)), float(np.std(current)), *q.tolist(), float(np.max(current)),
            float(np.mean(trend)), float(np.std(trend)), *tq.tolist(), float(np.max(trend)),
            float(np.sum(rain_h3)), float(np.max(rain_h3)) if rain_h3.size else 0.0,
            float(np.sum(rain_h6)), float(np.max(rain_h6)) if rain_h6.size else 0.0,
            float(np.sum(rain)), float(np.max(rain)) if rain.size else 0.0,
            float(np.mean(action)), float(np.std(action)),
            float(np.mean(action <= 1.0e-6)), float(np.mean(action >= 1.0 - 1.0e-6)),
        ],
        dtype=np.float32,
    )
    return features


def encode_signature(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value, dtype=np.float32).round(7).tolist(), separators=(",", ":"))


def decode_signature(value: Any) -> np.ndarray:
    return _array(value).reshape(-1)


def encode_sequence(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value, dtype=np.float32).round(7).tolist(), separators=(",", ":"))


def decode_sequence(value: Any) -> np.ndarray:
    result = _array(value)
    if result.ndim != 2:
        raise ValueError("candidate sequence must be 2D")
    return result


@dataclass(frozen=True)
class ExperienceRetrievalConfig:
    nearest_states: int = 8
    actions_per_state: int = 3
    max_warm_starts: int = 16
    min_action_l1_distance: float = 0.01
    require_pfv_feasible: bool = True
    prefer_tfv_improving: bool = True


class AuthoritativeExperienceBank:
    """Read-only nearest-state index over authoritative development actions."""

    REQUIRED = {
        "state_key",
        "candidate_action_sha256",
        "candidate_action_json",
        "state_signature_json",
        "pfv_feasible",
        "tfv_reduction_pct",
    }

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED - set(frame.columns))
        if missing:
            raise KeyError(f"experience bank missing columns: {missing}")
        work = frame.copy()
        work["state_key"] = work["state_key"].astype(str)
        work["candidate_action_sha256"] = work["candidate_action_sha256"].astype(str)
        work["tfv_reduction_pct"] = pd.to_numeric(work["tfv_reduction_pct"], errors="coerce")
        work["pfv_feasible"] = work["pfv_feasible"].astype(bool)
        work = work.drop_duplicates(["state_key", "candidate_action_sha256"], keep="last")
        signatures = []
        valid_rows = []
        for idx, value in work["state_signature_json"].items():
            try:
                signatures.append(decode_signature(value))
                valid_rows.append(idx)
            except Exception:
                continue
        if not valid_rows:
            raise ValueError("experience bank has no valid state signatures")
        work = work.loc[valid_rows].reset_index(drop=True)
        width = len(signatures[0])
        if any(len(item) != width for item in signatures):
            raise ValueError("experience bank signatures have inconsistent widths")
        self.frame = work
        self.signatures = np.stack(signatures).astype(np.float32)
        self.center = np.nanmedian(self.signatures, axis=0)
        q25 = np.nanquantile(self.signatures, 0.25, axis=0)
        q75 = np.nanquantile(self.signatures, 0.75, axis=0)
        self.scale = np.maximum(q75 - q25, 1.0e-6)

    @classmethod
    def load(cls, path: str | Path) -> "AuthoritativeExperienceBank":
        source = Path(path)
        if source.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(source)
        else:
            frame = pd.read_csv(source, low_memory=False)
        return cls(frame)

    def _distance(self, signature: np.ndarray) -> np.ndarray:
        query = np.asarray(signature, dtype=float).reshape(-1)
        if query.size != self.signatures.shape[1]:
            raise ValueError("query signature width differs from experience bank")
        z_bank = (self.signatures - self.center[None, :]) / self.scale[None, :]
        z_query = (query - self.center) / self.scale
        return np.sqrt(np.mean((z_bank - z_query[None, :]) ** 2, axis=1))

    def retrieve(
        self,
        *,
        signature: np.ndarray,
        current_action: np.ndarray,
        config: ExperienceRetrievalConfig = ExperienceRetrievalConfig(),
    ) -> list[dict[str, Any]]:
        """Return diverse, authoritative warm starts from nearby causal states."""
        distance = self._distance(signature)
        work = self.frame.copy()
        work["_distance"] = distance
        if config.require_pfv_feasible:
            work = work[work["pfv_feasible"]]
        if work.empty:
            return []
        state_distance = work.groupby("state_key", sort=False)["_distance"].min().sort_values()
        state_keys = state_distance.head(max(1, int(config.nearest_states))).index.astype(str).tolist()
        work = work[work["state_key"].isin(state_keys)].copy()
        work["_improving"] = (work["tfv_reduction_pct"] > 0.0).astype(int)
        sort_columns = ["state_key"]
        ascending = [True]
        if config.prefer_tfv_improving:
            sort_columns.append("_improving")
            ascending.append(False)
        sort_columns += ["tfv_reduction_pct", "_distance", "candidate_action_sha256"]
        ascending += [False, True, True]
        work = work.sort_values(sort_columns, ascending=ascending, kind="stable")
        work = work.groupby("state_key", sort=False).head(max(1, int(config.actions_per_state)))

        current = np.asarray(current_action, dtype=float).reshape(-1)
        selected: list[dict[str, Any]] = []
        selected_h3: list[np.ndarray] = []
        for row in work.itertuples(index=False):
            try:
                sequence = decode_sequence(row.candidate_action_json)
            except Exception:
                continue
            if sequence.shape[1] != current.size:
                continue
            h3 = sequence[:3]
            if any(float(np.mean(np.abs(h3 - old))) < float(config.min_action_l1_distance) for old in selected_h3):
                continue
            selected_h3.append(h3.copy())
            selected.append(
                {
                    "state_key": str(row.state_key),
                    "candidate_action_sha256": str(row.candidate_action_sha256),
                    "sequence": sequence.astype(np.float32),
                    "tfv_reduction_pct": float(row.tfv_reduction_pct),
                    "pfv_feasible": bool(row.pfv_feasible),
                    "retrieval_distance": float(row._distance),
                }
            )
            if len(selected) >= int(config.max_warm_starts):
                break
        return selected


def action_bank_row(
    *,
    state_key: str,
    state_signature_value: np.ndarray,
    candidate_action: np.ndarray,
    pfv_candidate_m3: float,
    pfv_no_control_m3: float,
    pfv_budget_metric_m3: float,
    pfv_feasible: bool,
    tfv_candidate_m3: float,
    tfv_internal_m3: float,
    tfv_reduction_pct: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence = np.asarray(candidate_action, dtype=np.float32)
    payload = {
        "experience_contract": EXPERIENCE_BANK_CONTRACT,
        "state_key": str(state_key),
        "state_signature_json": encode_signature(state_signature_value),
        "candidate_action_sha256": action_sha256(sequence),
        "candidate_action_json": encode_sequence(sequence),
        "pfv_candidate_m3": float(pfv_candidate_m3),
        "pfv_no_control_m3": float(pfv_no_control_m3),
        "pfv_budget_metric_m3": float(pfv_budget_metric_m3),
        "pfv_feasible": bool(pfv_feasible),
        "tfv_candidate_m3": float(tfv_candidate_m3),
        "tfv_internal_m3": float(tfv_internal_m3),
        "tfv_reduction_pct": float(tfv_reduction_pct),
    }
    if extra:
        payload.update(extra)
    return payload
