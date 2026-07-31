"""Diagnose signal strength in V4.2 training data."""
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sewerrtc.v4.v42_trainer import load_v42_training_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "final_v4"

data = load_v42_training_data(str(PROJECT_ROOT), str(OUTPUT_ROOT))

print("=" * 60)
print("TARGET DISTRIBUTION DIAGNOSTICS")
print("=" * 60)

for k in ["pfv_delta", "tfv_delta", "peak_delta"]:
    t = data[k].numpy().astype(np.float64)
    print(f"\n{k}:")
    print(f"  n={len(t)}  mean={t.mean():.4f}  std={t.std():.4f}")
    print(f"  min={t.min():.4f}  max={t.max():.4f}")
    print(f"  positive: {100*np.mean(t > 0):.1f}%")
    print(f"  |t|>1e-3: {100*np.mean(np.abs(t) > 1e-3):.1f}%")
    print(f"  |t|>1:    {100*np.mean(np.abs(t) > 1):.1f}%")
    # Signal-to-noise: std/|mean|
    if abs(t.mean()) > 1e-8:
        print(f"  SNR (|mean|/std): {abs(t.mean())/t.std():.4f}")
    else:
        print(f"  SNR: mean≈0")

# Event structure
eids = data.get("event_id", None)
if eids is not None:
    eids_np = eids.numpy() if isinstance(eids, torch.Tensor) else np.array(eids)
    unique_events = np.unique(eids_np)
    print(f"\nUnique events: {len(unique_events)}")
    samples_per_event = np.bincount(eids_np.astype(int))
    print(f"Samples per event: mean={samples_per_event.mean():.1f} min={samples_per_event.min()} max={samples_per_event.max()}")

# Action difference magnitude
act_c = data["action_candidate"].numpy()
act_r = data["action_reference"].numpy()
diff = act_c - act_r
diff_l1 = np.abs(diff).sum(axis=(1, 2))
diff_l2 = np.sqrt((diff ** 2).sum(axis=(1, 2)))
print(f"\nAction difference (Candidate - Reference):")
print(f"  L1 sum: mean={diff_l1.mean():.4f}  std={diff_l1.std():.4f}")
print(f"  L2 norm: mean={diff_l2.mean():.4f}  std={diff_l2.std():.4f}")
print(f"  zero diff (L1<1e-8): {100*np.mean(diff_l1 < 1e-8):.1f}%")

# Check if rainfall differs between candidate/reference
rain = data["rainfall"].numpy()
print(f"\nRainfall: shape={rain.shape}  sum_mean={rain.sum(axis=tuple(range(1,rain.ndim))).mean():.4f}")

# Check state
state = data["state_history"].numpy()
print(f"State: shape={state.shape}")

# Simple correlation between action diff and target
print("\n" + "=" * 60)
print("SIMPLE LINEAR CORRELATION (action_diff → target)")
print("=" * 60)
act_diff_flat = diff.reshape(len(diff), -1)
for k in ["pfv_delta", "tfv_delta", "peak_delta"]:
    t = data[k].numpy().astype(np.float64)
    # Correlation with each action diff feature
    max_corr = 0
    for j in range(act_diff_flat.shape[1]):
        c = np.corrcoef(act_diff_flat[:, j], t)[0, 1]
        if not np.isnan(c):
            max_corr = max(max_corr, abs(c))
    print(f"{k}: max |corr(action_diff_col, target)| = {max_corr:.6f}")

# Check if flat features have any correlation with target
from sewerrtc.v4.v42_single_head_cv import _extract_flat_features
flat_feats = _extract_flat_features(data)
print(f"\nFlat features: shape={flat_feats.shape}")
for k in ["pfv_delta", "tfv_delta", "peak_delta"]:
    t = data[k].numpy().astype(np.float64)
    corrs = [np.corrcoef(flat_feats[:, j], t)[0, 1] for j in range(flat_feats.shape[1])]
    corrs = np.array([c if not np.isnan(c) else 0 for c in corrs])
    print(f"{k}: max |corr(flat_feat, target)| = {np.max(np.abs(corrs)):.6f}  mean={np.mean(np.abs(corrs)):.6f}")

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
# If max correlation across all features and all targets is < 0.1, signal is very weak
all_max_corr = 0
for k in ["pfv_delta", "tfv_delta", "peak_delta"]:
    t = data[k].numpy().astype(np.float64)
    for j in range(flat_feats.shape[1]):
        c = abs(np.corrcoef(flat_feats[:, j], t)[0, 1])
        if not np.isnan(c):
            all_max_corr = max(all_max_corr, c)
    for j in range(act_diff_flat.shape[1]):
        c = abs(np.corrcoef(act_diff_flat[:, j], t)[0, 1])
        if not np.isnan(c):
            all_max_corr = max(all_max_corr, c)

print(f"Max absolute correlation (any feature, any target): {all_max_corr:.6f}")
if all_max_corr < 0.05:
    print("SIGNAL: NONE (max |r| < 0.05)")
elif all_max_corr < 0.1:
    print("SIGNAL: VERY WEAK (max |r| < 0.1)")
elif all_max_corr < 0.3:
    print("SIGNAL: WEAK (max |r| < 0.3)")
else:
    print(f"SIGNAL: MODERATE+ (max |r| = {all_max_corr:.4f})")
