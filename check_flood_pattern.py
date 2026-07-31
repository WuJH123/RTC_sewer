import pandas as pd
import numpy as np

# Load the no_control detail
df = pd.read_csv('outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real_v3/work/v3_cpA_pre_peak__no_control_detail.csv')

# Find flood columns
flood_cols = [c for c in df.columns if c.startswith('flood:')]
print(f"Flood columns: {len(flood_cols)}")

# Calculate total flood at each time step
df['total_flood'] = df[flood_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1)

# Find last flood time
flood_times = df[df['total_flood'] > 0]['elapsed_min']
if len(flood_times) > 0:
    print(f"Last flood time: {flood_times.max()} min")
    print(f"First flood time: {flood_times.min()} min")
    print(f"Total flood duration: {flood_times.max() - flood_times.min()} min")
    
    # Show flood pattern in the tail (after 300min)
    tail = df[df['elapsed_min'] >= 300][['elapsed_min', 'total_flood']]
    print(f"\nTail flood pattern (after 300min):")
    print(f"  Rows with flood: {len(tail[tail['total_flood'] > 0])}")
    print(f"  Rows without flood: {len(tail[tail['total_flood'] == 0])}")
    
    # Show last 20 rows
    print(f"\nLast 20 rows:")
    print(df[['elapsed_min', 'total_flood']].tail(20).to_string(index=False))
    
    # Find when flood becomes continuous zero (if ever)
    no_flood = df[df['total_flood'] == 0]['elapsed_min']
    if len(no_flood) > 0:
        # Find longest continuous no-flood period
        diffs = no_flood.diff()
        breaks = diffs[diffs > 5.0001].index
        print(f"\nLongest continuous no-flood periods:")
        segments = []
        start = no_flood.iloc[0]
        for idx in breaks:
            end = no_flood[no_flood.index < idx].iloc[-1]
            segments.append((start, end, end - start))
            start = no_flood[no_flood.index > idx].iloc[0] if len(no_flood[no_flood.index > idx]) > 0 else None
        if start is not None:
            end = no_flood.iloc[-1]
            segments.append((start, end, end - start))
        segments.sort(key=lambda x: x[2], reverse=True)
        for s in segments[:5]:
            print(f"  {s[0]:.0f} - {s[1]:.0f} min ({s[2]:.0f} min)")
