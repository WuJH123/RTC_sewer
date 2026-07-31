import glob
import json
import os
import shutil
import sys

run_root = sys.argv[1]
keep_prefix = sys.argv[2]  # per-case input_sha prefix to KEEP (current plan)
quarantine = sys.argv[3]

os.makedirs(quarantine, exist_ok=True)
moved = []
kept = []
for marker in glob.glob(os.path.join(run_root, "*", "completion.json")):
    case_dir = os.path.dirname(marker)
    try:
        payload = json.load(open(marker, encoding="utf-8"))
    except (OSError, ValueError):
        continue
    sha = str(payload.get("input_sha", ""))
    if sha.startswith(keep_prefix):
        kept.append(os.path.basename(case_dir))
        continue
    dest = os.path.join(quarantine, os.path.basename(case_dir))
    if os.path.exists(dest):
        dest = dest + "_" + sha[:8]
    shutil.move(case_dir, dest)
    moved.append((os.path.basename(case_dir), sha[:8]))

print("kept_count", len(kept))
print("moved_count", len(moved))
print("moved", moved[:5], "..." if len(moved) > 5 else "")
