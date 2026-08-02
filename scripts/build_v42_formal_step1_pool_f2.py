"""Expand Formal F2 Step1 with physically compatible *training* trajectories.

Formal Step1 does not require four-reference labels, but it must still respect the
frozen F2 rainfall ledger.  In particular, Calibration/Locked/Challenge/Blind
and unused-untouched rainfalls are never converted to auxiliary pretraining just
because a structured source (for example the opportunity pool) can resolve a
physical trajectory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, explicit_step1_roles, read_table, sha256_file, text
from sewerrtc.v4.v42_step1_dataset import _build_usecols, load_graph_assets

ROLE_ALIASES = ("candidate", "no_control", "dynamic_internal", "dynamic_internal_rules", "hold_previous")
TRAINABLE_LEDGER_ROLES = {"train", "auxiliary"}


def _index(root: Path) -> dict[str, list[Path]]:
    try:
        result = subprocess.run(
            ["rg", "--files", "-uu", "-g", "completion.json", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        paths = [Path(x) for x in result.stdout.splitlines() if x.strip()] if result.returncode in (0, 1) else []
    except FileNotFoundError:
        paths = list(root.rglob("completion.json"))
    out: dict[str, list[Path]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_id = text(payload.get("case_id", ""))
        if case_id:
            out.setdefault(case_id, []).append(path)
    return out


def _detail(completion: Path) -> Path | None:
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except Exception:
        return None
    branches = payload.get("branches", {})
    if not isinstance(branches, dict):
        return None
    for role in ROLE_ALIASES:
        value = branches.get(role)
        if value is None:
            continue
        raw = (
            value
            if isinstance(value, str)
            else text(value.get("detail_path") or value.get("path") or value.get("detail"))
            if isinstance(value, dict)
            else ""
        )
        if not raw:
            continue
        q = Path(raw)
        for candidate in (q, completion.parent / q, completion.parent / q.name):
            if candidate.exists():
                return candidate.resolve()
    return None


def _anchors(path: Path, limit: int) -> list[float]:
    elapsed = pd.to_numeric(pd.read_csv(path, usecols=["elapsed_min"]).elapsed_min, errors="coerce").dropna().to_numpy(float)
    times = {round(float(v), 6) for v in elapsed}
    valid = [a for a in sorted(times) if {round(a - 60 + 5 * i, 6) for i in range(13)}.issubset(times)]
    if len(valid) <= limit:
        return valid
    idx = np.linspace(0, len(valid) - 1, limit, dtype=int)
    return [valid[i] for i in sorted(set(idx))]


def _prefix(a: Path, b: Path) -> int:
    n = 0
    for x, y in zip(a.resolve().parts, b.resolve().parts):
        if x.casefold() != y.casefold():
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--source-rows",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_SOURCE_ROWS.parquet",
    )
    ap.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_EVENT_LEDGER.csv",
    )
    ap.add_argument(
        "--base-step1-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet",
    )
    ap.add_argument(
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet",
    )
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/project6_dual_reference_v4")
    ap.add_argument("--max-windows-per-physical-run", type=int, default=4)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--min-target-train-groups", type=int, default=65)
    args = ap.parse_args()

    source_rows = read_table(args.source_rows)
    ledger = read_table(args.ledger)
    role_map = dict(zip(ledger.rainfall_group_key.astype(str), ledger.formal_f2_role.astype(str)))

    graph = load_graph_assets(args.project_root)
    required = set(_build_usecols(graph.node_ids, graph.facility_ids))
    completion_index = _index(args.output_root)
    records: list[dict] = []
    failures: list[dict] = []
    sha_cache: dict[str, str] = {}

    # Legacy Step1 windows are retained only if their rainfall is explicitly
    # training/auxiliary in the frozen F2 ledger. Unknown and untouched groups
    # are fail-closed rather than silently becoming auxiliary pretraining.
    base_rows = 0
    base_rows_excluded_by_ledger = 0
    if args.base_step1_manifest.exists():
        base = read_table(args.base_step1_manifest)
        if "rainfall_sha256" in base.columns:
            rainfall = base.rainfall_sha256.fillna("").astype(str).str.strip()
            use = rainfall.ne("")
            if use.any():
                base.loc[use, "split_group_key"] = rainfall.loc[use]
        base_roles = base.get("split_group_key", pd.Series("", index=base.index)).astype(str).map(role_map).fillna("excluded")
        keep = base_roles.isin(TRAINABLE_LEDGER_ROLES)
        base_rows = int(keep.sum())
        base_rows_excluded_by_ledger = int((~keep).sum())
        for _, row in base.loc[keep].iterrows():
            data = row.to_dict()
            data["source_dataset"] = text(data.get("source_dataset", "legacy_step1_manifest"))
            records.append(data)

    allowed = source_rows.get("formal_step1_allowed", pd.Series(False, index=source_rows.index)).astype(bool)
    source_roles = source_rows.get("rainfall_group_key", pd.Series("", index=source_rows.index)).astype(str).map(role_map).fillna("excluded")
    eligible = source_rows.loc[allowed & source_roles.isin(TRAINABLE_LEDGER_ROLES)].copy()
    excluded_eligible_rows = int((allowed & ~source_roles.isin(TRAINABLE_LEDGER_ROLES)).sum())

    for _, row in eligible.iterrows():
        group = text(row.get("rainfall_group_key", ""))
        case_id = text(row.get("case_id", ""))
        raw = text(row.get("detail_path", ""))
        path = Path(raw) if raw else None
        try:
            if path is None or not path.exists():
                options = completion_index.get(case_id, [])
                if not options:
                    raise FileNotFoundError(f"no completion for case_id={case_id!r}")
                source_manifest = Path(text(row.get("source_manifest", "")))
                options = sorted(options, key=lambda p: (-_prefix(p, source_manifest), str(p).casefold()))
                path = next((candidate for candidate in (_detail(x) for x in options) if candidate is not None), None)
            if path is None or not path.exists():
                raise FileNotFoundError("no physical detail")
            header = set(map(str, pd.read_csv(path, nrows=0).columns))
            missing = sorted(required - header)
            if missing:
                raise KeyError(f"missing Step1 columns {missing[:8]}")
            key = str(path.resolve())
            sha_cache.setdefault(key, sha256_file(path))
            physical_sha = sha_cache[key]
            for anchor in _anchors(path, args.max_windows_per_physical_run):
                records.append(
                    {
                        "detail_path": key,
                        "anchor_min": float(anchor),
                        "split_group_key": group,
                        "rainfall_sha256": group,
                        "physical_identity_sha256": physical_sha,
                        "source_dataset": text(row.get("source_id", "historical")),
                        "formal_generation_id": FORMAL_GENERATION_ID,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "source_id": text(row.get("source_id", "")),
                    "case_id": case_id,
                    "rainfall_group": group,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("no Formal F2 Step1 windows")
    if "rainfall_sha256" in out.columns:
        rainfall = out.rainfall_sha256.fillna("").astype(str).str.strip()
        use = rainfall.ne("")
        out.loc[use, "split_group_key"] = rainfall.loc[use]

    # Final fail-closed guard: even if an upstream source was misconfigured, an
    # evaluation or untouched rainfall cannot survive into the training manifest.
    out_roles = out.split_group_key.astype(str).map(role_map).fillna("excluded")
    bad = ~out_roles.isin(TRAINABLE_LEDGER_ROLES)
    dropped_after_materialization = int(bad.sum())
    out = out.loc[~bad].copy()

    out = out.drop_duplicates(["physical_identity_sha256", "anchor_min", "split_group_key"]).reset_index(drop=True)
    out = explicit_step1_roles(out, ledger, validation_fraction=args.validation_fraction, split_seed=args.split_seed)
    train = int(
        out.loc[out.step1_domain_role.eq("target_formal") & out.formal_split.eq("train"), "split_group_key"]
        .astype(str)
        .nunique()
    )
    validation = int(
        out.loc[out.step1_domain_role.eq("target_formal") & out.formal_split.eq("validation"), "split_group_key"]
        .astype(str)
        .nunique()
    )
    auxiliary = int(
        out.loc[out.step1_domain_role.eq("auxiliary_pretrain"), "split_group_key"].astype(str).nunique()
    )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_manifest, index=False)
    audit = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_step1_pool",
        "status": "pass" if train >= args.min_target_train_groups else "fail",
        "rows": len(out),
        "physical_runs": int(out.physical_identity_sha256.astype(str).nunique()),
        "target_train_rainfall_groups": train,
        "target_validation_rainfall_groups": validation,
        "auxiliary_rainfall_groups": auxiliary,
        "minimum_target_train_groups": args.min_target_train_groups,
        "base_rows_kept_by_ledger": base_rows,
        "base_rows_excluded_by_ledger": base_rows_excluded_by_ledger,
        "eligible_source_rows_excluded_by_ledger": excluded_eligible_rows,
        "rows_dropped_by_final_ledger_guard": dropped_after_materialization,
        "evaluation_or_unused_rainfalls_enter_training_manifest": False,
        "failed_source_rows": len(failures),
        "failure_examples": failures[:200],
    }
    (args.output_manifest.parent / "FORMAL_F2_STEP1_POOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if audit["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
