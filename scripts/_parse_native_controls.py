"""Parse Wuhan INP native [CONTROLS] and analyze vs Engineering36."""
import re
from pathlib import Path

INP = Path("data/wuhan_v8_storage_retrofit.inp")
ENG36_FILE = Path("data/project6_v8_storage_retrofit_control_enabled_ids.txt")

with open(INP, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find [CONTROLS] section
in_controls = False
rules = []
current_rule = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("[CONTROLS]"):
        in_controls = True
        continue
    if in_controls:
        if stripped.startswith("[") and not stripped.startswith("[CONTROLS]"):
            break
        if stripped.startswith("RULE "):
            if current_rule:
                rules.append(current_rule)
            current_rule = [stripped]
        elif current_rule:
            current_rule.append(stripped)
if current_rule:
    rules.append(current_rule)

print(f"Total RULE lines: {len(rules)}")

# Load Engineering36
eng36 = set()
with open(ENG36_FILE) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            eng36.add(line.split(",")[0].strip())
print(f"Engineering36 count: {len(eng36)}")

# Parse each rule
rule_details = []
all_facilities = set()
for rule_lines in rules:
    rule_name = rule_lines[0].replace("RULE ", "").strip()
    facility = None
    setting_val = None
    premise_parts = []
    priority = None
    for part in rule_lines[1:]:
        if part.startswith("IF "):
            premise_parts.append(part[3:])
        elif part.startswith("THEN "):
            tokens = part.split()
            # THEN ORIFICE/PUMP <name> SETTING = <value>
            for j, tok in enumerate(tokens):
                if tok in ("ORIFICE", "PUMP", "WEIR", "OUTLET") and j + 1 < len(tokens):
                    facility = tokens[j + 1]
                    all_facilities.add(facility)
            if "=" in part:
                setting_val = part.split("=")[-1].strip()
        elif part.startswith("PRIORITY"):
            priority = part.split()[-1].strip()
    rule_details.append({
        "name": rule_name,
        "facility": facility,
        "setting": setting_val,
        "premise": "; ".join(premise_parts),
        "priority": priority,
    })

# Analysis
rule_facilities = set(r["facility"] for r in rule_details if r["facility"])
eng36_overlap = sorted(rule_facilities & eng36)
non_eng36 = sorted(rule_facilities - eng36)

print(f"\nRule-controlled facilities: {len(rule_facilities)}")
print(f"Engineering36 overlap: {len(eng36_overlap)}")
print(f"Non-Engineering36 controlled: {len(non_eng36)}")

print("\nEngineering36 facilities controlled by native rules:")
for fac in eng36_overlap:
    matching = [r for r in rule_details if r["facility"] == fac]
    for r in matching:
        print(f"  {fac}: rule={r['name']}, setting={r['setting']}, premise={r['premise']}")

print("\nNon-Engineering36 facilities controlled by native rules:")
for fac in non_eng36:
    matching = [r for r in rule_details if r["facility"] == fac]
    for r in matching:
        print(f"  {fac}: rule={r['name']}, setting={r['setting']}")

# Count PUMPS/ORIFICES in INP
pump_count = 0
orifice_count = 0
weir_count = 0
outlet_count = 0
for line in lines:
    s = line.strip()
    if s.startswith("[PUMPS]"):
        section = "pumps"
        continue
    elif s.startswith("[ORIFICES]"):
        section = "orifices"
        continue
    elif s.startswith("[WEIRS]"):
        section = "weirs"
        continue
    elif s.startswith("[OUTLETS]"):
        section = "outlets"
        continue
    elif s.startswith("["):
        section = ""
        continue
    if section == "pumps" and s and not s.startswith(";;"):
        pump_count += 1
    elif section == "orifices" and s and not s.startswith(";;"):
        orifice_count += 1
    elif section == "weirs" and s and not s.startswith(";;"):
        weir_count += 1
    elif section == "outlets" and s and not s.startswith(";;"):
        outlet_count += 1

print(f"\nINP asset counts:")
print(f"  PUMPS: {pump_count}")
print(f"  ORIFICES: {orifice_count}")
print(f"  WEIRS: {weir_count}")
print(f"  OUTLETS: {outlet_count}")
print(f"  Total controllable: {pump_count + orifice_count + weir_count + outlet_count}")
print(f"  Native RULE count: {len(rule_details)}")
