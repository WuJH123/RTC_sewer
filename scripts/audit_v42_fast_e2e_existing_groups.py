"""Read-only census of existing groups usable by the development fast-E2E line."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\RTC_sewer\Project6")
DATA = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse"
OUT = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus"


def boolean(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(False, index=df.index)
    return df[name].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def forcing(df: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    cols = [c for c in ("same_forcing_pass_y", "same_forcing_pass", "same_forcing_pass_x") if c in df]
    if not cols:
        return pd.Series(False, index=df.index), {"missing": len(df)}
    values = {c: boolean(df, c) for c in cols}
    resolved = values[cols[0]].copy()
    audit = {"/".join(("true" if values[cols[0]].iloc[i] else "false", "true" if values[cols[-1]].iloc[i] else "false")): 0 for i in range(0)}
    for i in df.index:
        key = "/".join("true" if values[c].loc[i] else "false" for c in cols)
        audit[key] = audit.get(key, 0) + 1
    return resolved, audit


def fingerprint_census(physical: pd.DataFrame) -> dict[str, object]:
    raw = physical.get("rainfall_sha256", pd.Series(dtype=str)).fillna("").astype(str)
    split = physical.get("split_group_key", pd.Series(dtype=str)).fillna("").astype(str)
    return {
        "rainfall_sha256_unique": int(raw[raw.ne("")].nunique()),
        "split_group_key_unique": int(split[split.ne("")].nunique()),
        "rainfall_to_split_pairs": int(pd.DataFrame({"rainfall": raw, "split": split}).drop_duplicates().shape[0]),
    }


def filesystem_census() -> dict[str, object]:
    tokens = ("train1600", "gate5r", "v4")
    roles = ("candidate", "no_control", "dynamic_internal", "hold_previous")
    rows: list[dict[str, object]] = []
    roots = [p for p in (ROOT / "outputs").iterdir() if p.is_dir() and any(t in p.name.lower() for t in tokens)]
    listed: list[str] = []
    for root in roots:
        result = subprocess.run(["rg", "--files", str(root)], capture_output=True, text=True, check=True)
        listed.extend(result.stdout.splitlines())
    for listed_path in listed:
        path = Path(listed_path)
        low_path = str(path).lower()
        if not any(t in low_path for t in tokens):
            continue
        for name in [path.name]:
            low = name.lower()
            if not low.endswith(".csv") or not any(r in low for r in roles):
                continue
            try:
                header = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception as exc:
                header = [f"READ_ERROR:{type(exc).__name__}"]
            rows.append({"path": str(path), "source_family": next(t for t in tokens if t in str(path).lower()), "role": next(r for r in roles if r in low), "header": json.dumps(header), "size_bytes": path.stat().st_size})
    frame = pd.DataFrame(rows)
    return {"on_disk_files": int(len(frame)), "by_source_family": frame.groupby("source_family").size().to_dict() if not frame.empty else {}, "by_role": frame.groupby("role").size().to_dict() if not frame.empty else {}, "files": frame.to_dict("records")}


def main() -> int:
    physical = pd.read_parquet(DATA / "reusable_pool_manifest.parquet")
    cases = pd.read_parquet(DATA / "reusable_case_manifest.parquet")
    split = pd.read_parquet(DATA / "split_group_manifest.parquet")
    split_by_id = split.set_index("physical_identity_sha256")["split_group_key"].astype(str).to_dict()
    physical = physical.copy()
    physical["split_group_key"] = physical["physical_identity_sha256"].astype(str).map(split_by_id).fillna("")
    forcing_ok, forcing_counts = forcing(cases)
    core = boolean(cases, "four_reference_complete") & boolean(cases, "same_state_numeric_pass") & forcing_ok & boolean(cases, "four_reference_finite_pass") & boolean(cases, "core_trajectory_targets")
    not_reserved = ~cases.get("source_role", pd.Series("", index=cases.index)).astype(str).eq("reserved_evaluation")
    tiers = {
        "TARGET_STRICT_CORE": boolean(cases, "eligible_counterfactual_flood"),
        "SOURCE_STRICT_CORE": boolean(cases, "eligible_source_domain_counterfactual_aux"),
        "DEVELOPMENT_COMPATIBLE_CORE": core & not_reserved,
        "TARGET_STRICT_FULL": boolean(cases, "eligible_formal_all_target"),
    }
    split_map = split.set_index("physical_identity_sha256")["split_group_key"].astype(str).to_dict()
    by_id = physical.set_index("physical_identity_sha256")
    def group_for(row: pd.Series) -> str:
        try:
            ids = json.loads(row.get("branch_physical_ids", "[]"))
        except Exception:
            ids = []
        groups = {split_map.get(str(i), "") for i in ids}
        groups.discard("")
        return next(iter(groups)) if len(groups) == 1 else ""
    cases = cases.copy()
    cases["audit_split_group_key"] = cases.apply(group_for, axis=1)
    cases["checkpoint_ge_120"] = pd.to_numeric(cases["checkpoint_min"], errors="coerce").ge(120)
    cases["audit_state_key"] = cases["event_id"].astype(str) + "|" + cases["checkpoint_min"].astype(str)
    case_rows = []
    for name, mask in tiers.items():
        sub = cases[mask & cases.checkpoint_ge_120]
        counts = sub.groupby(["audit_split_group_key", "audit_state_key"]).size()
        eligible = counts[counts.ge(3)]
        case_rows.append({"tier": name, "cases": int(mask.sum()), "rainfall_sha256_unique": int(cases.loc[mask, "rainfall_sha256"].nunique()), "split_groups_ge120": int(sub.audit_split_group_key.nunique()), "states_ge120": int(sub.audit_state_key.nunique()), "groups_with_ge3_candidates": int(eligible.index.get_level_values(0).nunique()) if len(eligible) else 0, "states_with_ge3_candidates": int(len(eligible)), "source_experiments": int(cases.loc[mask, "source_experiment"].nunique())})
    audit = {"development_only": True, "admission_tiers": case_rows, "forcing_resolution_counts": forcing_counts, "fingerprint_census": fingerprint_census(physical), "filesystem_census": filesystem_census(), "physical_rows": int(len(physical)), "case_rows": int(len(cases)), "fast_selector_must_not_relabel_domain": True}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case_rows).to_csv(OUT / "fast_e2e_existing_group_audit.csv", index=False)
    (OUT / "fast_e2e_existing_group_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# FAST-E2E existing-group admission audit", "", "Development-only, read-only audit.", "", "```text", pd.DataFrame(case_rows).to_string(index=False), "```", "", f"Fingerprint census: `{json.dumps(audit['fingerprint_census'], sort_keys=True)}`", f"Forcing resolution: `{json.dumps(forcing_counts, sort_keys=True)}`"]
    (OUT / "FAST_E2E_EXISTING_GROUP_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
