"""Content-addressed, development-only recovery of historical V4.2 trajectories."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\RTC_sewer\Project6")
DATA = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse"
OUT = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus"
ROLES = {"candidate", "no_control", "dynamic_internal", "hold_previous"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def boolcol(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(False, index=df.index)
    return df[name].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def forcing_value(row: pd.Series) -> bool:
    for name in ("same_forcing_pass_y", "same_forcing_pass", "same_forcing_pass_x"):
        if name in row.index and str(row[name]).lower() in {"true", "1", "yes"}:
            return True
    return False


def rainfall_fingerprint(path: Path) -> tuple[str, int, float | None, float | None]:
    try:
        head = pd.read_csv(path, nrows=0)
        cols = {c.lower(): c for c in head.columns}
        if "elapsed_min" not in cols or "rainfall_mm_h" not in cols:
            return "", 0, None, None
        d = pd.read_csv(path, usecols=[cols["elapsed_min"], cols["rainfall_mm_h"]])
        d.columns = ["elapsed_min", "rainfall_mm_h"]
        d = d.apply(pd.to_numeric, errors="coerce").dropna()
        if d.empty:
            return "", 0, None, None
        payload = "\n".join(f"{a:.9f},{b:.9f}" for a, b in d.to_numpy())
        return sha(payload), len(d), float(d.elapsed_min.min()), float(d.elapsed_min.max())
    except Exception:
        return "", 0, None, None


def main() -> int:
    physical = pd.read_parquet(DATA / "reusable_pool_manifest.parquet")
    split = pd.read_parquet(DATA / "split_group_manifest.parquet")
    split_map = split.set_index("physical_identity_sha256")["split_group_key"].astype(str).to_dict()
    rows: list[dict[str, object]] = []
    for row in physical.itertuples(index=False):
        path = Path(str(row.detail_path))
        old_fp, n, tmin, tmax = rainfall_fingerprint(path) if path.exists() else ("", 0, None, None)
        rain = old_fp or str(getattr(row, "rainfall_sha256", ""))
        role = str(getattr(row, "branch_role", ""))
        checkpoint = getattr(row, "checkpoint_min", None)
        checkpoint_text = "" if pd.isna(checkpoint) else f"{float(checkpoint):.6f}"
        action = str(getattr(row, "action_readback_sha256", ""))
        detail = str(getattr(row, "detail_sha256", ""))
        stable = sha("|".join((detail, role, checkpoint_text, rain, action)))
        prefix = str(getattr(row, "prefix_hash_match", ""))
        state = sha("|".join((rain, checkpoint_text, str(getattr(row, "network_sha256", "")), prefix)))
        rows.append({
            "path": str(path), "detail_sha256": detail, "row_count": n,
            "elapsed_min_min": tmin, "elapsed_min_max": tmax,
            "rainfall_series_sha256": rain, "old_rainfall_sha256": str(getattr(row, "rainfall_sha256", "")),
            "old_split_group_key": split_map.get(str(getattr(row, "physical_identity_sha256", "")), ""),
            "checkpoint_min": checkpoint, "branch_role": role, "action_readback_sha256": action,
            "network_sha256": str(getattr(row, "network_sha256", "")), "domain_id": str(getattr(row, "domain_id", "")),
            "source_role": str(getattr(row, "source_role", "")), "physical_identity_sha256": str(getattr(row, "physical_identity_sha256", "")),
            "stable_trajectory_key": stable, "counterfactual_state_key": state,
            "finite_pass": bool(getattr(row, "available_finite_pass", False)),
            "core_pass": bool(getattr(row, "core_trajectory_complete", False)),
            "windowable_13x12": bool(getattr(row, "windowable_13x12", False)),
        })
    catalog = pd.DataFrame(rows)
    valid = catalog["rainfall_series_sha256"].ne("")
    step1 = catalog[valid & catalog.windowable_13x12 & catalog.core_pass & catalog.finite_pass].copy()
    step1["rainfall_group_key"] = step1["rainfall_series_sha256"]
    cases: list[dict[str, object]] = []
    for state, group in step1.groupby("counterfactual_state_key"):
        refs = {r: group[group.branch_role.eq(r)] for r in ROLES if not group[group.branch_role.eq(r)].empty}
        if not {"no_control", "dynamic_internal", "hold_previous"}.issubset(refs):
            continue
        candidates = group[group.branch_role.eq("candidate")].drop_duplicates("action_readback_sha256")
        for cand in candidates.itertuples(index=False):
            cases.append({"rainfall_group_key": cand.rainfall_group_key, "counterfactual_state_key": state, "checkpoint_min": cand.checkpoint_min, "candidate_trajectory_key": cand.stable_trajectory_key, "no_control_trajectory_key": refs["no_control"].iloc[0].stable_trajectory_key, "dynamic_internal_trajectory_key": refs["dynamic_internal"].iloc[0].stable_trajectory_key, "hold_previous_trajectory_key": refs["hold_previous"].iloc[0].stable_trajectory_key, "candidate_action_signature": cand.action_readback_sha256, "same_forcing_pass": True, "same_state_pass": True, "finite_pass": True, "core_trajectory_pass": True, "domain_id": cand.domain_id, "source_role": cand.source_role, "development_only": True})
    virtual = pd.DataFrame(cases)
    if virtual.empty:
        virtual = pd.DataFrame(columns=["rainfall_group_key", "counterfactual_state_key", "candidate_action_signature"])
    virtual = virtual[virtual.checkpoint_min.ge(120)] if "checkpoint_min" in virtual else virtual
    state_counts = virtual.groupby(["rainfall_group_key", "counterfactual_state_key"]).candidate_action_signature.nunique() if not virtual.empty else pd.Series(dtype=int)
    usable_states = state_counts[state_counts.ge(3)]
    funnel = pd.DataFrame([
        {"stage": "raw_discovered_trajectory_files", "count": int(len(catalog))},
        {"stage": "hydraulic_trajectory_files", "count": int(valid.sum())},
        {"stage": "finite_hydraulic_trajectories", "count": int((valid & catalog.finite_pass).sum())},
        {"stage": "unique_recomputed_rainfall_fingerprints", "count": int(catalog.loc[valid, "rainfall_series_sha256"].nunique())},
        {"stage": "step1_causal_trajectories", "count": int(len(step1))},
        {"stage": "candidate_trajectories", "count": int((step1.branch_role == "candidate").sum())},
        {"stage": "no_control_trajectories", "count": int((step1.branch_role == "no_control").sum())},
        {"stage": "dynamic_internal_trajectories", "count": int((step1.branch_role == "dynamic_internal").sum())},
        {"stage": "hold_previous_trajectories", "count": int((step1.branch_role == "hold_previous").sum())},
        {"stage": "virtual_four_reference_cases", "count": int(len(virtual))},
        {"stage": "states_with_ge3_distinct_candidates", "count": int(len(usable_states))},
        {"stage": "usable_rainfall_groups_after_checkpoint_and_candidates", "count": int(usable_states.index.get_level_values(0).nunique()) if len(usable_states) else 0},
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(OUT / "CONTENT_ADDRESSED_TRAJECTORY_INDEX.parquet", index=False)
    virtual.to_parquet(OUT / "VIRTUAL_FOUR_REFERENCE_CASES.parquet", index=False)
    funnel.to_csv(OUT / "RECOVERED_POOL_FUNNEL.csv", index=False)
    audit = {"development_only": True, "catalog_rows": int(len(catalog)), "recomputed_rainfall_groups": int(catalog.loc[valid, "rainfall_series_sha256"].nunique()), "old_rainfall_groups": int(catalog.loc[valid, "old_rainfall_sha256"].replace("", np.nan).nunique()), "old_split_groups": int(catalog.old_split_group_key.replace("", np.nan).nunique()), "virtual_cases": int(len(virtual)), "funnel": funnel.to_dict("records"), "outputs": {"virtual_cases": str(OUT / "VIRTUAL_FOUR_REFERENCE_CASES.parquet")}}
    (OUT / "CONTENT_ADDRESSED_TRAJECTORY_AUDIT.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "CONTENT_ADDRESSED_TRAJECTORY_AUDIT.md").write_text("# Content-addressed trajectory recovery\n\nDevelopment-only.\n\n```text\n" + funnel.to_string(index=False) + "\n```\n", encoding="utf-8")
    (OUT / "RECOVERED_POOL_FUNNEL.json").write_text(json.dumps(funnel.to_dict("records"), indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
