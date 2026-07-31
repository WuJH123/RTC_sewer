"""Archive frozen Peak Boundary evidence before any sewerrtc/v4 code edits.

Copies (never moves) all Peak Boundary scientific evidence into
outputs/project6_dual_reference_v4/final_v4/audits/frozen_evidence/
peak_boundary/<old_code_sha>/ and writes a SHA-256 manifest covering the
raw branch detail files. Fail-closed: refuses to overwrite an existing
archive and never deletes or modifies source evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"E:\RTC_sewer\Project6")
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.runtime import working_code_sha  # noqa: E402

FINAL_V4 = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "final_v4"
PEAK_DIR = FINAL_V4 / "peak_boundary"
AUDITS = FINAL_V4 / "audits"
STAGE_STATUS = AUDITS / "stage_status"

PEAK_STAGE_NAMES = (
    "PlanPeakBoundary",
    "BuildPeakCandidateCatalog",
    "AuditPeakBoundaryPreflight",
    "RunPeakBoundary",
    "BuildPeakBoundaryPartial",
    "AuditPeakBoundaryPartial",
    "BuildPeakBoundaryDataset",
    "AuditPeakBoundary",
)

PEAK_TOP_FILES = (
    "peak_boundary_plan.csv",
    "peak_boundary_anchor_library.csv",
    "peak_boundary_audit.json",
    "sample_manifest.csv",
    "run_manifest.csv",
    "rejected_manifest.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_into(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    code_sha = working_code_sha(PROJECT_ROOT)
    archive_root = (
        AUDITS / "frozen_evidence" / "peak_boundary" / code_sha
    )
    if archive_root.exists():
        print(f"REFUSE: archive already exists: {archive_root}")
        print("Existing frozen evidence must never be overwritten.")
        return 2
    staging = archive_root.with_name(archive_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    copied: list[str] = []

    # 1. Peak Boundary top-level dataset/manifest/audit files.
    for name in PEAK_TOP_FILES:
        source = PEAK_DIR / name
        if not source.exists():
            print(f"FAIL: missing required evidence file: {source}")
            return 3
        copy_into(source, staging / "peak_boundary" / name)
        copied.append(f"peak_boundary/{name}")

    # 2. All Peak stage status stamps + stage completions.
    for stage in PEAK_STAGE_NAMES:
        for suffix in (".json", ".completion.json"):
            source = STAGE_STATUS / f"{stage}{suffix}"
            if not source.exists():
                print(f"FAIL: missing stage status: {source}")
                return 3
            copy_into(
                source, staging / "stage_status" / f"{stage}{suffix}"
            )
            copied.append(f"stage_status/{stage}{suffix}")

    # 3. Preflight report and partial-audit bundles for Peak.
    preflight = AUDITS / "preflight" / "AuditPeakBoundaryPreflight.json"
    if preflight.exists():
        copy_into(
            preflight, staging / "preflight" / preflight.name
        )
        copied.append(f"preflight/{preflight.name}")
    for partial_name in (
        "BuildPeakBoundaryPartial",
        "AuditPeakBoundaryPartial",
    ):
        partial_dir = AUDITS / "partial" / partial_name
        if partial_dir.is_dir():
            for item in sorted(partial_dir.rglob("*")):
                if item.is_file():
                    relative = item.relative_to(AUDITS / "partial")
                    copy_into(item, staging / "partial" / relative)
                    copied.append(f"partial/{relative.as_posix()}")

    # 4. The 240 per-case completion.json files (structure preserved).
    completions = sorted(PEAK_DIR.glob("runs/*/completion.json"))
    if len(completions) != 240:
        print(
            "FAIL: expected exactly 240 completion.json files, found "
            f"{len(completions)}"
        )
        return 3
    for marker in completions:
        case_id = marker.parent.name
        copy_into(marker, staging / "runs" / case_id / "completion.json")
        copied.append(f"runs/{case_id}/completion.json")

    # 5. SHA-256 manifest for every raw branch artifact (not copied:
    #    ~6 GB of case.inp/detail.csv stay in place, hashes freeze them).
    raw_rows: list[dict] = []
    for case_dir in sorted(PEAK_DIR.glob("runs/*")):
        if not case_dir.is_dir():
            continue
        for item in sorted(case_dir.rglob("*")):
            if item.is_file():
                raw_rows.append(
                    {
                        "case_id": case_dir.name,
                        "relative_path": item.relative_to(
                            PEAK_DIR
                        ).as_posix(),
                        "size_bytes": item.stat().st_size,
                        "sha256": sha256_file(item),
                    }
                )
    detail_count = sum(
        1 for row in raw_rows if row["relative_path"].endswith("detail.csv")
    )
    if detail_count != 240:
        print(
            "FAIL: expected 240 raw branch detail.csv files, found "
            f"{detail_count}"
        )
        return 3
    manifest_path = staging / "raw_branch_file_sha256.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    # 6. Archive-level manifest.
    summary = {
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "old_code_sha": code_sha,
        "archive_root": str(archive_root),
        "copied_file_count": len(copied),
        "completion_json_count": len(completions),
        "raw_branch_file_count": len(raw_rows),
        "raw_detail_csv_count": detail_count,
        "copied_files": copied,
    }
    (staging / "archive_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    staging.rename(archive_root)
    print(f"ARCHIVED: {archive_root}")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "copied_files"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
