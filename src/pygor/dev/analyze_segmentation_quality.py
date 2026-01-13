"""Analyze segmentation quality beyond just ROI count.

Compare trained vs pretrained on:
1. Per-image accuracy (not just total count)
2. Over-segmentation vs under-segmentation
3. Visual comparison
"""
from cellpose import models
import pygor.load
import numpy as np
from pathlib import Path
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Load test images
data_dirs = [
    Path(r"D:\Igor analyses\SWN BC main"),
    Path(r"D:\Igor analyses\SWN inj"),
    Path(r"D:\Igor analyses\SWN single colour"),
]

images = []
masks = []
filenames = []

for data_dir in data_dirs:
    if not data_dir.exists():
        continue
    h5_files = sorted(data_dir.glob("**/*.h5"))
    for h5_file in h5_files:
        try:
            d = pygor.load.Core(str(h5_file))
            if d.rois is not None and d.average_stack is not None:
                img = d.average_stack.astype(np.float32)
                binary_mask = (d.rois != 1).astype(np.uint8)
                cellpose_mask, n_rois = label(binary_mask)
                if n_rois >= 3:
                    images.append(img)
                    masks.append(cellpose_mask.astype(np.uint16))
                    filenames.append(h5_file.name)
        except:
            continue

print(f"Loaded {len(images)} images")

# Load models
model_path = Path("src/pygor/dev/.models_simple/models/cellpose_rois")
trained_model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
pretrained_model = models.CellposeModel(gpu=True)

params = {
    "diameter": None,
    "flow_threshold": 0.6,
    "cellprob_threshold": 0.0,
    "min_size": 3,
}

# Analyze each image
print("\nPer-image analysis:")
print("-" * 80)
print(f"{'Image':<40} {'GT':>6} {'Trained':>8} {'Pretrain':>8} {'T-diff':>8} {'P-diff':>8}")
print("-" * 80)

trained_results = []
pretrained_results = []

for i, (img, mask, fname) in enumerate(zip(images, masks, filenames)):
    n_gt = len(np.unique(mask)) - 1

    pred_t, _, _ = trained_model.eval(img, **params)
    n_trained = len(np.unique(pred_t)) - 1

    pred_p, _, _ = pretrained_model.eval(img, **params)
    n_pretrained = len(np.unique(pred_p)) - 1

    diff_t = n_trained - n_gt
    diff_p = n_pretrained - n_gt

    trained_results.append((n_gt, n_trained, diff_t))
    pretrained_results.append((n_gt, n_pretrained, diff_p))

    # Truncate filename for display
    fname_short = fname[:38] if len(fname) > 38 else fname
    print(f"{fname_short:<40} {n_gt:>6} {n_trained:>8} {n_pretrained:>8} {diff_t:>+8} {diff_p:>+8}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

trained_diffs = [r[2] for r in trained_results]
pretrained_diffs = [r[2] for r in pretrained_results]

print(f"\nTrained model:")
print(f"  Mean difference from GT: {np.mean(trained_diffs):+.1f}")
print(f"  Std of difference: {np.std(trained_diffs):.1f}")
print(f"  Under-segmented (merged): {sum(1 for d in trained_diffs if d < 0)} images")
print(f"  Over-segmented (split): {sum(1 for d in trained_diffs if d > 0)} images")
print(f"  Perfect count: {sum(1 for d in trained_diffs if d == 0)} images")

print(f"\nPretrained model:")
print(f"  Mean difference from GT: {np.mean(pretrained_diffs):+.1f}")
print(f"  Std of difference: {np.std(pretrained_diffs):.1f}")
print(f"  Under-segmented (merged): {sum(1 for d in pretrained_diffs if d < 0)} images")
print(f"  Over-segmented (split): {sum(1 for d in pretrained_diffs if d > 0)} images")
print(f"  Perfect count: {sum(1 for d in pretrained_diffs if d == 0)} images")

# Calculate absolute error (better metric)
trained_abs_err = [abs(r[2]) for r in trained_results]
pretrained_abs_err = [abs(r[2]) for r in pretrained_results]

print(f"\nMean Absolute Error (lower is better):")
print(f"  Trained: {np.mean(trained_abs_err):.1f} ROIs")
print(f"  Pretrained: {np.mean(pretrained_abs_err):.1f} ROIs")

# Percentage error
trained_pct_err = [abs(r[2])/r[0]*100 if r[0] > 0 else 0 for r in trained_results]
pretrained_pct_err = [abs(r[2])/r[0]*100 if r[0] > 0 else 0 for r in pretrained_results]

print(f"\nMean Absolute Percentage Error:")
print(f"  Trained: {np.mean(trained_pct_err):.1f}%")
print(f"  Pretrained: {np.mean(pretrained_pct_err):.1f}%")

# Visualize a few examples
print("\n" + "=" * 80)
print("Saving comparison figure...")

# Find images where models differ most
diff_between = [abs(t[1] - p[1]) for t, p in zip(trained_results, pretrained_results)]
interesting_idx = np.argsort(diff_between)[-4:]  # 4 most different

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for row, idx in enumerate(interesting_idx):
    img = images[idx]
    gt_mask = masks[idx]
    n_gt = len(np.unique(gt_mask)) - 1

    pred_t, _, _ = trained_model.eval(img, **params)
    n_t = len(np.unique(pred_t)) - 1

    pred_p, _, _ = pretrained_model.eval(img, **params)
    n_p = len(np.unique(pred_p)) - 1

    # Original
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title(f'Image\n{filenames[idx][:25]}...')
    axes[row, 0].axis('off')

    # Ground truth
    gt_colored = np.ma.masked_where(gt_mask == 0, gt_mask)
    axes[row, 1].imshow(img, cmap='gray')
    axes[row, 1].imshow(gt_colored, cmap='tab20', alpha=0.6)
    axes[row, 1].set_title(f'Ground Truth\n({n_gt} ROIs)')
    axes[row, 1].axis('off')

    # Trained
    t_colored = np.ma.masked_where(pred_t == 0, pred_t)
    axes[row, 2].imshow(img, cmap='gray')
    axes[row, 2].imshow(t_colored, cmap='tab20', alpha=0.6)
    axes[row, 2].set_title(f'Trained\n({n_t} ROIs, {n_t-n_gt:+d})')
    axes[row, 2].axis('off')

    # Pretrained
    p_colored = np.ma.masked_where(pred_p == 0, pred_p)
    axes[row, 3].imshow(img, cmap='gray')
    axes[row, 3].imshow(p_colored, cmap='tab20', alpha=0.6)
    axes[row, 3].set_title(f'Pretrained\n({n_p} ROIs, {n_p-n_gt:+d})')
    axes[row, 3].axis('off')

plt.tight_layout()
plt.savefig('src/pygor/dev/segmentation_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to: src/pygor/dev/segmentation_comparison.png")
plt.close()
