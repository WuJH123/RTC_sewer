"""Archive the frozen Peak Boundary evidence before any sewerrtc/v4 edit.

Section I of the Pilot subsystem authorization: record and archive, under
``audits/frozen_evidence/peak_boundary/<old_code_sha>/``:

- current working_code_sha;
- every Peak Boundary stage status (+ completion markers);
- sample_manifest.csv / run_manifest.csv / peak_boundary_anchor_library.csv /
  peak_boundary_audit.json (plus plan + rejected manifest for completeness);
- the 240 per-case completion.json markers;
- SHA256 of every raw branch file under peak_boundary/runs/.

The script is copy-only and fail-closed: it never deletes, modifies or
overwrites old evidence, and it refuses to finish if the completion count is
not exactly 240. Lives outside sewerrtc/v4 so it never changes the code SHA.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.runtime import working_code_sha  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4"
PEAK_ROOT = OUTPUT_ROOT / "peak_boundary"
STAGE_STATUS = OUTPUT_ROOT / "audits" / "stage_status"

EXPECTED_COMPLETIONS = 240

PEAK_FILES = (
    "sample_manifest.csv",
    "run_manifest.csv",
    "peak_boundary_anchor_library.csv",
    "peak_boundary_audit.json",
    "peak_boundary_plan.csv",
    "rejected_manifest.csv",
)

PEAK_STAGES = (
    "BuildPeakCandidateCatalog",
    "PlanPeakBoundary",
    "AuditPeakBoundaryPreflight",
    "RunPeakBoundary",
    "BuildPeakBoundaryPartial",
    "AuditPeakBoundaryPartial",
    "BuildPeakBoundaryDataset",
    "AuditPeakBoundary",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    old_code_sha = working_code_sha(PROJECT_ROOT)
    destination = (
        OUTPUT_ROOT
        / "audits"
        / "frozen_evidence"
        / "peak_boundary"
        / old_code_sha
    )
    manifest_path = destination / "archive_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"archive already complete: {manifest_path}")
        print(json.dumps(payload.get("counts", {}), indent=2))
        return 0

    completions = sorted(PEAK_ROOT.glob("runs/*/completion.json"))
    if len(completions) != EXPECTED_COMPLETIONS:
        print(
            f"FAIL CLOSED: expected {EXPECTED_COMPLETIONS} completion.json, "
            f"found {len(completions)}",
            file=sys.stderr,
        )
        return 2

    archived: dict[str, str] = {}

    def copy_into(source: Path, relative: str) -> None:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        shutil.copy2(source, target)
        archived[relative] = _sha256(target)

    destination.mkdir(parents=True, exist_ok=True)

    for name in PEAK_FILES:
        source = PEAK_ROOT / name
        if not source.exists():
            print(f"FAIL CLOSED: missing {source}", file=sys.stderr)
            return 2
        copy_into(source, f"peak_boundary/{name}")

    for stage in PEAK_STAGES:
        for suffix in (".json", ".completion.json"):
            source = STAGE_STATUS / f"{stage}{suffix}"
            if source.exists():
                copy_into(source, f"stage_status/{stage}{suffix}")

    for marker in completions:
        case_id = marker.parent.name
        copy_into(marker, f"completions/{case_id}/completion.json")

    # SHA256 of every raw branch file under runs/ (detail CSVs, SWMM inp/rpt/
    # out/log and completion markers) without copying the heavy payloads.
    raw_rows = []
    for item in sorted(PEAK_ROOT.glob("runs/**/*")):
        if not item.is_file():
            continue
        raw_rows.append(
            {
                "relative_path": item.relative_to(PEAK_ROOT).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
        )
    sha_manifest = destination / "raw_branch_file_sha256.csv"
    with sha_manifest.open("w", encoding="utf-8", newline="") as handle:
        handle.write("relative_path,size_bytes,sha256\n")
        for row in raw_rows:
            handle.write(
                f"{row['relative_path']},{row['size_bytes']},{row['sha256']}\n"
            )
    archived["raw_branch_file_sha256.csv"] = _sha256(sha_manifest)

    detail_files = [
        row for row in raw_rows if row["relative_path"].endswith("detail.csv")
    ]
    counts = {
        "completion_json": len(completions),
        "raw_branch_files_hashed": len(raw_rows),
        "detail_csv_hashed": len(detail_files),
        "archived_files": len(archived),
    }
    payload = {
        "old_code_sha": old_code_sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_output_root": str(OUTPUT_ROOT),
        "counts": counts,
        "archived_sha256": archived,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"archived to: {destination}")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
