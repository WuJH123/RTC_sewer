import collections
import glob
import json
import os
import sys

root = sys.argv[1]
markers = glob.glob(os.path.join(root, "*", "completion.json"))
inp = collections.Counter()
code = collections.Counter()
status = collections.Counter()
for marker in markers:
    try:
        payload = json.load(open(marker, encoding="utf-8"))
    except (OSError, ValueError):
        continue
    inp[str(payload.get("input_sha"))[:8]] += 1
    code[str(payload.get("code_git_sha"))[:8]] += 1
    status[str(payload.get("status"))] += 1

print("case_dirs_with_completion", len(markers))
print("by_input_sha", dict(inp))
print("by_code_sha", dict(code))
print("by_status", dict(status))
