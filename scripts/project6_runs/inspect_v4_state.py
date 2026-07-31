"""Read-only inspection of Final V4 outputs before the Pilot subsystem work.

Non-v4 helper: lives outside sewerrtc/v4 so it never changes working_code_sha.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.runtime import working_code_sha  # noqa: E402

ROOT = PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4"


def main() -> int:
    print("working_code_sha:", working_code_sha(PROJECT_ROOT))
    peak = ROOT / "peak_boundary"
    print("\n[peak_boundary files]")
    for item in sorted(peak.iterdir()):
        if item.is_file():
            print(f"  {item.name}  {item.stat().st_size}")
    runs = peak / "runs"
    case_dirs = sorted(d for d in runs.iterdir() if d.is_dir())
    completions = list(runs.glob("*/*/completion.json"))
    print(f"[peak runs] case_dirs={len(case_dirs)} completions={len(completions)}")
    print("\n[pilot tree]")
    pilot = ROOT / "pilot"
    for item in sorted(pilot.rglob("*")):
        if item.is_file():
            print(f"  {item.relative_to(pilot)}  {item.stat().st_size}")
    print("\n[stage_status]")
    for item in sorted((ROOT / "audits" / "stage_status").glob("*.json")):
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
        except ValueError:
            payload = {}
        print(
            f"  {item.name}: status={payload.get('status')} "
            f"exit={payload.get('exit_code')} "
            f"scope_complete={payload.get('scope_complete')}"
        )
    catalog = ROOT / "opportunities" / "standard_checkpoint_catalog.csv"
    header = catalog.read_text(encoding="utf-8").splitlines()
    print(f"\n[standard_checkpoint_catalog] rows={len(header) - 1}")
    print("  columns:", header[0])
    anchor = peak / "peak_boundary_anchor_library.csv"
    if anchor.exists():
        print("\n[anchor library] columns:", anchor.read_text(encoding="utf-8").splitlines()[0][:2000])
    plan = ROOT / "pilot" / "planning" / "pilot_candidate_plan.csv"
    if plan.exists():
        lines = plan.read_text(encoding="utf-8").splitlines()
        print(f"\n[pilot_candidate_plan(current role specs)] rows={len(lines) - 1}")
        print("  columns:", lines[0])
    ref_plan = ROOT / "pilot" / "planning" / "pilot_reference_plan.csv"
    if ref_plan.exists():
        lines = ref_plan.read_text(encoding="utf-8").splitlines()
        print(f"\n[pilot_reference_plan] rows={len(lines) - 1}")
        print("  columns:", lines[0])
    peak_plan = ROOT / "peak_boundary" / "peak_boundary_plan.csv"
    if peak_plan.exists():
        lines = peak_plan.read_text(encoding="utf-8").splitlines()
        print(f"\n[peak_boundary_plan] rows={len(lines) - 1}")
        print("  columns:", lines[0][:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
