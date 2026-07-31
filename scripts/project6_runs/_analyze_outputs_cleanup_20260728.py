"""Read-only cleanup potential analysis for outputs/ (2026-07-28).

Statistics only -- never deletes anything.  Writes:
  cleanup_manifests/outputs_cleanup_analysis_20260728.json   (summary)
  cleanup_manifests/outputs_cleanup_candidates_20260728.csv  (file list)

Classification follows the A/B/C/D precedent (cleanup_confirmed_abcd.ps1)
and the Project6 truth rules: frozen evidence, audits, contracts,
manifests, completion markers and reference caches are NEVER listed.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(r"E:\RTC_sewer\Project6\outputs")
OUT_DIR = Path(r"E:\RTC_sewer\Project6\cleanup_manifests")
STAMP = "20260728"

# --- protected zones: never propose deletion inside these -----------------
PROTECTED_PARTS = (
    "frozen_evidence",       # pilot v1/v2 frozen evidence
    "audits",                # stage status, hashes, gate evidence
    "_cleanup_manifests",
    "references",            # pilot reference cache (reused by pipeline)
    "dataset", "dataset_v2", "dataset_v3",  # built label datasets
    "evaluation", "map", "legacy_oracle", "planning",
    "inventory", "contracts", "golden_v4",
)
PROTECTED_NAMES = (
    "completion.json", "manifest", "audit", "verdict", "freeze",
    "catalog", "contract", "plan", "coverage", "summary",
)

# --- category rules --------------------------------------------------------
# (category, safety, description)
CAT_SWMM_OUT = "A_swmm_out_rpt"          # regenerable SWMM binary/report
CAT_CASE_INP = "B_generated_case_inp"    # auto-generated INP (regenerable)
CAT_DETAIL = "C_branch_detail_csv"       # label recompute basis (caution)
CAT_HOTSTART = "D_hotstart_hsf"          # hotstart caches (no-hotstart contract)
CAT_TMP = "E_tmp_partial"                # tmp/partial/backup leftovers
CAT_LEGACY_HEAVY = "F_legacy_branch_dirs"  # superseded legacy phases heavy files

LEGACY_HEAVY_TOPS = (
    "recovery_capability_v2", "recovery_capability_v1",
    "dual_reference_aug1", "gate5r_informative_v3_exact_native_prefix",
    "gate5r_informative_v1", "gate5r_informative_v2_no_dwf",
    "recovery_validation", "oracle_pareto_20ev", "oracle_pareto_smoke1_fix",
    "oracle_bottleneck_diagnosis", "cl",
)


def is_protected(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if any(z in parts for z in PROTECTED_PARTS):
        return True
    name = path.name.lower()
    return any(k in name for k in PROTECTED_NAMES)


def classify(path: Path, rel: str) -> tuple[str, str] | None:
    ext = path.suffix.lower()
    name = path.name.lower()
    top = rel.split(os.sep, 2)
    v4_top = top[1] if len(top) > 1 and top[0] == "project6_dual_reference_v4" else ""
    if ext in (".out", ".rpt"):
        return CAT_SWMM_OUT, "safe"
    if ext == ".hsf" or "hotstart" in name:
        return CAT_HOTSTART, "safe"
    if name.endswith((".tmp", ".partial", ".bak", ".old")) or name.startswith("~"):
        return CAT_TMP, "safe"
    if name == "case.inp" or (ext == ".inp" and ("case_inp" in rel or "event_inp" in rel)):
        return CAT_CASE_INP, "safe_regenerable"
    if name == "detail.csv" and os.sep + "runs" + os.sep in os.sep + rel:
        return CAT_DETAIL, "caution_label_recompute"
    if v4_top in LEGACY_HEAVY_TOPS and ext in (".inp", ".csv", ".parquet", ".npz", ".npy") \
            and path.stat().st_size > 5 * 1024 * 1024:
        return CAT_LEGACY_HEAVY, "caution_legacy_superseded"
    return None


def main() -> None:
    rows = []
    stats: dict[str, dict] = {}
    total_bytes = 0
    total_files = 0
    protected_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(ROOT):
        d = Path(dirpath)
        for fn in filenames:
            p = d / fn
            try:
                size = p.stat().st_size
            except OSError:
                continue
            total_bytes += size
            total_files += 1
            rel = str(p.relative_to(ROOT))
            if is_protected(p.relative_to(ROOT)):
                protected_bytes += size
                continue
            hit = classify(p, rel)
            if hit is None:
                continue
            cat, safety = hit
            s = stats.setdefault(cat, {"files": 0, "bytes": 0, "safety": safety})
            s["files"] += 1
            s["bytes"] += size
            rows.append((cat, safety, str(p), size))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"outputs_cleanup_candidates_{STAMP}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["category", "safety", "path", "bytes"])
        writer.writerows(rows)

    summary = {
        "generated_at": STAMP,
        "mode": "statistics_only_no_deletion",
        "outputs_total_files": total_files,
        "outputs_total_gb": round(total_bytes / 1024**3, 2),
        "protected_gb": round(protected_bytes / 1024**3, 2),
        "categories": {
            cat: {
                "files": s["files"],
                "gb": round(s["bytes"] / 1024**3, 2),
                "safety": s["safety"],
            }
            for cat, s in sorted(stats.items())
        },
        "total_candidate_gb": round(
            sum(s["bytes"] for s in stats.values()) / 1024**3, 2
        ),
        "safe_only_gb": round(
            sum(
                s["bytes"]
                for s in stats.values()
                if s["safety"].startswith("safe")
            )
            / 1024**3,
            2,
        ),
        "candidates_manifest": str(csv_path),
    }
    json_path = OUT_DIR / f"outputs_cleanup_analysis_{STAMP}.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
