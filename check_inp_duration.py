from pathlib import Path
import re

p = Path('outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real_v3/work/V31_RP10_D5H_P35_v31_independent_gamma_108__no_controls.inp')
lines = p.read_text()
m = re.search(r'END_DATE\s+(\S+)', lines)
m2 = re.search(r'END_TIME\s+(\S+)', lines)
print(f'END_DATE: {m.group(1) if m else "not found"}')
print(f'END_TIME: {m2.group(1) if m2 else "not found"}')

# Calculate expected end time
from datetime import datetime, timedelta
start = datetime(2022, 8, 11, 0, 0, 0)
end = start + timedelta(minutes=1020)
print(f'Expected END_DATE: {end:%m/%d/%Y}')
print(f'Expected END_TIME: {end:%H:%M:%S}')
