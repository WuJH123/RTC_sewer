from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.contracts.prompt3a import config_hash, sha256_file, utc_now
from sewerrtc.io.rainfall_injection import build_rainfall_library


FORMAL_RAIN_IDS = ["T5", "T10", "T20", "T50"]
FORMAL_DURATIONS_MIN = [60, 180, 300]
FORMAL_PATTERNS = ["chicago_early", "chicago_center", "chicago_late"]
DESIGN_DEPTH_MM = {
    "T5": 51.68,
    "T10": 56.39,
    "T20": 61.42,
    "T50": 68.30,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Project6 V3 formal rainfall assets only; no SWMM execution.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "rainfall_library"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        out_dir / f"{rid}_D{duration}_{pattern}.csv"
        for rid in FORMAL_RAIN_IDS
        for duration in FORMAL_DURATIONS_MIN
        for pattern in FORMAL_PATTERNS
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        missing = [path for path in expected if not path.exists()]
        if missing:
            # Generate into a temporary folder first, then copy only missing files.
            tmp = out_dir / "_formal_generation_tmp"
            table = build_rainfall_library(FORMAL_RAIN_IDS, FORMAL_DURATIONS_MIN, DESIGN_DEPTH_MM, FORMAL_PATTERNS, tmp)
            for path in missing:
                generated = tmp / path.name
                path.write_bytes(generated.read_bytes())
            if (tmp / "rainfall_event_table.csv").exists():
                (out_dir / "formal_rainfall_event_table.csv").write_bytes((tmp / "rainfall_event_table.csv").read_bytes())
            for generated in tmp.glob("*.csv"):
                generated.unlink()
            tmp.rmdir()
        else:
            table = None
    else:
        table = build_rainfall_library(FORMAL_RAIN_IDS, FORMAL_DURATIONS_MIN, DESIGN_DEPTH_MM, FORMAL_PATTERNS, out_dir)
        event_table = out_dir / "rainfall_event_table.csv"
        if event_table.exists():
            (out_dir / "formal_rainfall_event_table.csv").write_bytes(event_table.read_bytes())

    manifest_rows = []
    for path in expected:
        event_id = path.stem
        parts = event_id.split("_")
        manifest_rows.append(
            {
                "event_id": event_id,
                "rainfall_path": str(path),
                "rainfall_sha256": sha256_file(path) if path.exists() else "",
                "status": "available" if path.exists() else "missing",
                "return_period_year": parts[0][1:] if parts else "",
                "duration_min": parts[1][1:] if len(parts) > 1 else "",
                "peak_pattern": "_".join(parts[2:]) if len(parts) > 2 else "",
                "generated_by": "scripts/203_build_formal_rainfall_assets.py",
                "created_at": utc_now(),
            }
        )
    import csv

    manifest_path = out_dir / "formal_rainfall_asset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "status": "pass" if all(row["status"] == "available" for row in manifest_rows) else "blocked",
        "formal_core_event_count": len(manifest_rows),
        "available_count": sum(1 for row in manifest_rows if row["status"] == "available"),
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_hash": config_hash(args.config),
        "created_at": utc_now(),
        "swmm_executed": False,
        "formal_results_unlocked": False,
    }
    report_path = out_dir / "formal_rainfall_asset_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
