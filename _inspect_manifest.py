"""Quick manifest inspection."""
import pandas as pd
m = pd.read_parquet("outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet")
print(f"Shape: {m.shape}")
print(f"Columns: {m.columns.tolist()}")
print(f"Has step1_domain_role: {'step1_domain_role' in m.columns}")
print(f"Has formal_target_domain: {'formal_target_domain' in m.columns}")
if "step1_domain_role" in m.columns:
    print(f"Domain role counts:\n{m['step1_domain_role'].value_counts()}")
if "formal_target_domain" in m.columns:
    print(f"Target domain counts:\n{m['formal_target_domain'].value_counts()}")
if "split_group_key" in m.columns:
    print(f"Unique split groups: {m['split_group_key'].nunique()}")
