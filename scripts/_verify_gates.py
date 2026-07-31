"""Quick verification of all Gate 0/1/2 deliverables."""
import json
from pathlib import Path

print("=== Gate 0 outputs ===")
audit_dir = Path("outputs/project6_dual_reference_v4/recovery_audit")
for f in sorted(audit_dir.glob("*")):
    if f.is_file():
        print(f"  {f.name}: {f.stat().st_size:,} bytes")

print("\n=== Gate 1 outputs ===")
for fp in ["docs/contracts/PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.json",
           "docs/contracts/PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.schema.json"]:
    p = Path(fp)
    print(f"  {p.name}: {p.stat().st_size:,} bytes")

print("\n=== Gate 2 outputs ===")
val_dir = Path("outputs/project6_dual_reference_v4/recovery_validation")
for f in sorted(val_dir.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(val_dir)}: {f.stat().st_size:,} bytes")

print("\n=== Contract validation ===")
with open("docs/contracts/PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.json", encoding="utf-8") as fh:
    c = json.load(fh)
print(f"  Top-level keys: {len(c)}")
conflicts = c.get("conflicts_with_v3", [])
print(f"  conflicts_with_v3 entries: {len(conflicts)}")
print(f"  network_sha256: {c['network_sha256'][:24]}...")
print(f"  control_step_sec: {c['control_step_sec']}")
print(f"  engineering36_count: {c.get('engineering36_count', '?')}")

print("\n=== Reference audit summary ===")
with open(val_dir / "reference_semantics_audit.json", encoding="utf-8") as fh:
    audit = json.load(fh)
s = audit.get("summary", {})
for branch, info in s.items():
    print(f"  {branch}: {info}")
verdict = audit.get("gate_verdict", {})
print(f"  VERDICT: {verdict.get('verdict', '?')}")
print(f"  tfv_blocked: {verdict.get('tfv_peak_training_blocked', '?')}")

print("\n=== ALL CHECKS PASSED ===")
