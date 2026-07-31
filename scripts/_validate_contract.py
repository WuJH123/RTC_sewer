"""Minimal JSON-schema validator for the Truth Contract.

Does not depend on the `jsonschema` package (not installed in the venv).
Checks the subset of constraints actually used in our schema:
  - required top-level keys
  - type checks
  - const checks
  - pattern check on network_sha256
  - conflicts_with_v3 item shape
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "docs" / "contracts" / "PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.schema.json"
DATA = ROOT / "docs" / "contracts" / "PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.json"


def _check(data: dict, schema: dict) -> list[str]:
    errors: list[str] = []
    for key in schema.get("required", []):
        if key not in data:
            errors.append(f"missing required key: {key}")
    props = schema.get("properties", {})
    for key, spec in props.items():
        if key not in data:
            continue
        val = data[key]
        t = spec.get("type")
        if t == "string" and not isinstance(val, str):
            errors.append(f"{key}: expected string, got {type(val).__name__}")
        elif t == "integer" and not isinstance(val, int):
            errors.append(f"{key}: expected integer, got {type(val).__name__}")
        elif t == "boolean" and not isinstance(val, bool):
            errors.append(f"{key}: expected boolean, got {type(val).__name__}")
        elif t == "object" and not isinstance(val, dict):
            errors.append(f"{key}: expected object, got {type(val).__name__}")
        elif t == "array" and not isinstance(val, list):
            errors.append(f"{key}: expected array, got {type(val).__name__}")
        if "const" in spec and val != spec["const"]:
            errors.append(f"{key}: expected const {spec['const']!r}, got {val!r}")
        if "pattern" in spec and isinstance(val, str):
            if not re.match(spec["pattern"], val):
                errors.append(f"{key}: value {val!r} does not match pattern {spec['pattern']}")
    return errors


def main() -> int:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    data = json.loads(DATA.read_text(encoding="utf-8"))
    errors = _check(data, schema)
    # conflicts_with_v3 item shape
    cwv = data.get("conflicts_with_v3", [])
    if not isinstance(cwv, list):
        errors.append("conflicts_with_v3: expected array")
    else:
        for i, item in enumerate(cwv):
            for required in ("key", "v3_value", "v4_value", "reason"):
                if required not in item:
                    errors.append(f"conflicts_with_v3[{i}]: missing {required}")
    # facility_semantics sub-required
    fs = data.get("facility_semantics", {})
    for required in ("binary_pumps", "variable_speed_pumps", "all_others"):
        if required not in fs:
            errors.append(f"facility_semantics: missing {required}")
    for i, bp in enumerate(fs.get("binary_pumps", [])):
        for required in ("id", "action_set"):
            if required not in bp:
                errors.append(f"facility_semantics.binary_pumps[{i}]: missing {required}")
    for i, vp in enumerate(fs.get("variable_speed_pumps", [])):
        for required in ("id", "domain"):
            if required not in vp:
                errors.append(f"facility_semantics.variable_speed_pumps[{i}]: missing {required}")
    # kpi_definitions
    kpi = data.get("kpi_definitions", {})
    for required in ("PFV", "TFV", "Peak"):
        if required not in kpi:
            errors.append(f"kpi_definitions: missing {required}")
    # reference_roles
    rr = data.get("reference_roles", {})
    for required in ("PFV", "TFV", "Peak"):
        if required not in rr:
            errors.append(f"reference_roles: missing {required}")
    print(f"validation errors: {len(errors)}")
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
