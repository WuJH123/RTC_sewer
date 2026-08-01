"""Warm-up-safe wrapper for the V4.2 fast E2E dataset selection.

A Step2 state at time t contains 13 reconstructed Step1 frames from t-60..t.
Each Step1 reconstruction itself consumes the preceding 60 minutes. Therefore a
truly causal integrated sample needs raw observations back to t-120.  Filtering
checkpoint times *before* rainfall/state selection prevents late pipeline
attrition from silently shrinking a nominal 64+ rainfall pilot below its target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .v42_fast_e2e import (
    DEFAULT_CANDIDATES_PER_STATE,
    DEFAULT_TARGET_RAINFALL_GROUPS,
    MIN_RAINFALL_GROUPS,
    PREFERRED_SOURCE_TOKENS,
    build_fast_step2_dataset_64plus,
)


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def build_warm_fast_step2_dataset_64plus(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    case_manifest: str | Path,
    split_manifest: str | Path,
    working_dir: str | Path,
    target_groups: int = DEFAULT_TARGET_RAINFALL_GROUPS,
    min_groups: int = MIN_RAINFALL_GROUPS,
    candidates_per_state: int = DEFAULT_CANDIDATES_PER_STATE,
    preferred_source_tokens: Sequence[str] = PREFERRED_SOURCE_TOKENS,
    seed: int = 42,
    min_checkpoint_min: float = 120.0,
):
    work = Path(working_dir)
    work.mkdir(parents=True, exist_ok=True)
    cases = _read(case_manifest)
    if "checkpoint_min" not in cases.columns:
        raise KeyError("case manifest missing checkpoint_min required by causal warm-up gate")
    checkpoint = pd.to_numeric(cases["checkpoint_min"], errors="coerce")
    finite = checkpoint.notna()
    late = finite & checkpoint.ge(float(min_checkpoint_min))
    filtered = cases.loc[late].copy()
    if filtered.empty:
        raise RuntimeError(
            f"no Step2 cases remain after checkpoint >= {min_checkpoint_min} min causal warm-up gate"
        )
    filtered_path = work / "step2_fast_e2e_warm_case_pool.parquet"
    filtered.to_parquet(filtered_path, index=False)
    audit = {
        "stage": "fast_e2e_causal_warmup_prefilter",
        "development_only": True,
        "input_cases": int(len(cases)),
        "finite_checkpoint_cases": int(finite.sum()),
        "minimum_checkpoint_min": float(min_checkpoint_min),
        "retained_cases": int(len(filtered)),
        "blocked_early_or_missing_checkpoint_cases": int(len(cases) - len(filtered)),
        "reason": (
            "Step2 needs reconstructed states at t-60..t and each Step1 state needs its own "
            "60-minute causal observation history; formal online input must not synthesize missing warm-up."
        ),
    }
    (work / "step2_fast_e2e_warmup_prefilter.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    return build_fast_step2_dataset_64plus(
        project_root=project_root,
        physical_manifest=physical_manifest,
        case_manifest=filtered_path,
        split_manifest=split_manifest,
        working_dir=work,
        target_groups=target_groups,
        min_groups=min_groups,
        candidates_per_state=candidates_per_state,
        preferred_source_tokens=preferred_source_tokens,
        seed=seed,
    )
