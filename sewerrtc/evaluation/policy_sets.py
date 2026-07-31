from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_PAPER_POLICY_SET = (
    "proposed_temporal_joint_36",
    "internal_rules",
    "efd_storage_priority",
    "auto_rbc",
    "no_control",
)

DEFAULT_DIAGNOSTIC_POLICY_SET = (
    "all_open",
    "random_safe",
    "efd_static",
    "proposed_native_shield",
)

POLICY_ALIASES = {
    "proposed_pfv_first_mpc": "proposed_gat_mpc",
    "proposed": "proposed_gat_mpc",
    "proposed_dual_reference_v4": "proposed_dual_reference_v4",
    "gat_mpc": "proposed_gat_mpc",
    "generic_gat_mpc": "proposed_gat_mpc",
    "temporal_joint_36": "proposed_temporal_joint_36",
    "proposed_hierarchical_v8_residual_36": "proposed_temporal_joint_36",
    "retrofit_hierarchical36": "retrofit_hierarchical36",
    "hierarchical_core26_residual10": "retrofit_hierarchical36",
    "native_shield": "proposed_native_shield",
    "no_control_diagnostic": "no_control",
    "public_mpc": "official_mpc",
    "pystorms_beta_mpc": "official_mpc",
    "mpc_rules_beta": "official_mpc",
}


def normalize_policy_id(policy_id: object) -> str:
    text = str(policy_id or "").strip()
    return POLICY_ALIASES.get(text, text)


def _as_policy_list(value: object, default: Iterable[str]) -> list[str]:
    if value is None:
        raw = list(default)
    elif isinstance(value, str):
        raw = [p.strip() for p in value.split(",")]
    else:
        raw = [str(p).strip() for p in value]  # type: ignore[arg-type]
    out: list[str] = []
    for policy in raw:
        norm = normalize_policy_id(policy)
        if norm and norm not in out:
            out.append(norm)
    return out


def paper_policy_ids(cfg: dict | None = None) -> list[str]:
    evaluation = (cfg or {}).get("evaluation", {}) or {}
    return _as_policy_list(evaluation.get("paper_policy_set"), DEFAULT_PAPER_POLICY_SET)


def diagnostic_policy_ids(cfg: dict | None = None) -> list[str]:
    evaluation = (cfg or {}).get("evaluation", {}) or {}
    paper = set(paper_policy_ids(cfg))
    out = _as_policy_list(evaluation.get("diagnostic_policy_set"), DEFAULT_DIAGNOSTIC_POLICY_SET)
    return [policy for policy in out if policy not in paper]


def paper_baseline_policy_ids(cfg: dict | None = None) -> list[str]:
    proposed = {"proposed_gat_mpc", "proposed_temporal_joint_36", "proposed_native_shield", "retrofit_hierarchical36", "proposed_dual_reference_v4"}
    return [policy for policy in paper_policy_ids(cfg) if policy not in proposed]


def split_policy_set_frames(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "policy_id" not in df:
        return df.copy(), df.iloc[0:0].copy()
    paper = set(paper_policy_ids(cfg))
    diagnostic = set(diagnostic_policy_ids(cfg))
    work = df.copy()
    work["policy_id"] = work["policy_id"].map(normalize_policy_id)
    main = work[work["policy_id"].isin(paper)].copy()
    diag = work[work["policy_id"].isin(diagnostic)].copy()
    return main, diag
