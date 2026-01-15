"""Final test of the trained model vs pretrained baseline."""
from cellpose import models
import pygor.load
import numpy as np
from pathlib import Path
from scipy.ndimage import label

# Load test images
data_dirs = [
    Path(r"D:\Igor analyses\SWN BC main"),
    Path(r"D:\Igor analyses\SWN inj"),
    Path(r"D:\Igor analyses\SWN single colour"),
]

images = []
masks = []

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
        except:
            continue

print(f"Loaded {len(images)} test images")
total_gt = sum(len(np.unique(m)) - 1 for m in masks)
print(f"Total ground truth ROIs: {total_gt}")

# Load trained model
model_path = Path("src/pygor/dev/.models_simple/models/cellpose_rois")
print(f"\nLoading trained model: {model_path}")
trained_model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

# Load pretrained model
print("Loading pretrained model...")
pretrained_model = models.CellposeModel(gpu=True)

# Test both models with optimized parameters
params = {
    "diameter": None,
    "flow_threshold": 0.6,
    "cellprob_threshold": 0.0,
    "min_size": 3,
}

print(f"\nTesting with parameters: {params}")
print("=" * 70)

# Test trained model
print("\n[TRAINED MODEL]")
trained_total = 0
for i, (img, mask) in enumerate(zip(images, masks)):
    pred, _, _ = trained_model.eval(img, **params)
    n_pred = len(np.unique(pred)) - 1
    n_gt = len(np.unique(mask)) - 1
    trained_total += n_pred

print(f"Total detected: {trained_total} / {total_gt} = {100*trained_total/total_gt:.1f}%")

# Test pretrained model
print("\n[PRETRAINED MODEL]")
pretrained_total = 0
for i, (img, mask) in enumerate(zip(images, masks)):
    pred, _, _ = pretrained_model.eval(img, **params)
    n_pred = len(np.unique(pred)) - 1
    n_gt = len(np.unique(mask)) - 1
    pretrained_total += n_pred

print(f"Total detected: {pretrained_total} / {total_gt} = {100*pretrained_total/total_gt:.1f}%")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Ground truth ROIs: {total_gt}")
print(f"Trained model:     {trained_total} ({100*trained_total/total_gt:.1f}%)")
print(f"Pretrained model:  {pretrained_total} ({100*pretrained_total/total_gt:.1f}%)")

if trained_total > pretrained_total:
    print(f"\n✓ Trained model is BETTER by {trained_total - pretrained_total} ROIs")
elif pretrained_total > trained_total:
    print(f"\n✗ Pretrained model is better by {pretrained_total - trained_total} ROIs")
else:
    print(f"\n= Both models perform equally")
