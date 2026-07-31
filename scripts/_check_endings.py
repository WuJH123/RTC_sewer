"""Check if pipeline.py has consistent line endings."""
from pathlib import Path
p = Path("sewerrtc/v4/pipeline.py")
data = p.read_bytes()
crlf_count = data.count(b'\r\n')
lf_only = data.count(b'\n') - crlf_count
print(f"CRLF count: {crlf_count}")
print(f"LF-only count: {lf_only}")
print(f"File size: {len(data)}")
print(f"First 20 bytes: {data[:20]}")

# Check a few other files for comparison
for name in ["runtime.py", "pipeline_v42.py", "v42_water_balance.py"]:
    p2 = Path(f"sewerrtc/v4/{name}")
    if p2.exists():
        d2 = p2.read_bytes()
        crlf2 = d2.count(b'\r\n')
        lf2 = d2.count(b'\n') - crlf2
        print(f"{name}: CRLF={crlf2}, LF-only={lf2}")
