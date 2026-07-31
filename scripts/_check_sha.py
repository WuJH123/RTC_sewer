import hashlib
from pathlib import Path

p = Path("sewerrtc/v4/pipeline.py")
data = p.read_bytes()
print(f"Size: {len(data)} bytes")
print(f"SHA256: {hashlib.sha256(data).hexdigest()}")
crlf = b'\r\n'
print(f"Has CRLF: {crlf in data}")
lf = b'\n'
print(f"Ends with newline: {data[-1:] == lf}")

# Check all v4 py files
v4_dir = Path("sewerrtc/v4")
files = sorted(v4_dir.glob("*.py"))
digest = hashlib.sha256()
# get git head
import subprocess
try:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    git_head = r.stdout.strip()
except:
    git_head = ""
digest.update(git_head.encode())
for f in files:
    digest.update(str(f.relative_to(".")).encode())
    digest.update(f.read_bytes())
print(f"\nComputed working_code_sha: {digest.hexdigest()}")
print(f"Git HEAD: {git_head}")
print(f"Number of v4/*.py files: {len(files)}")
