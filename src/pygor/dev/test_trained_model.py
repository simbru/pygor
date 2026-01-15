"""Quick test of the trained model with different diameter settings."""
from cellpose import models
import pygor.load
import numpy as np
from pathlib import Path

# Load a training file
data_dir = Path(r"D:\Igor analyses\SWN BC main")
h5_files = sorted(data_dir.glob("**/*.h5"))

data = None
for h5_file in h5_files[:10]:
    try:
        d = pygor.load.Core(str(h5_file))
        if d.rois is not None and d.average_stack is not None:
            data = d
            print(f"Loaded: {h5_file.name}")
            break
    except:
        continue

if data is None:
    print("No valid data found!")
    exit()

# Count ground truth ROIs
from scipy.ndimage import label
binary_mask = (data.rois != 1).astype(np.uint8)
gt_mask, n_gt = label(binary_mask)
print(f"Ground truth: {n_gt} ROIs")

# Load the trained model
model_path = Path("src/pygor/dev/.models_simple/models/cellpose_rois")
print(f"\nLoading model from: {model_path}")
model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

print("\nTesting with different diameter settings:")
print("-" * 50)

# Test with different diameters
for diam in [None, 4.5, 10, 20, 30]:
    masks, flows, styles = model.eval(
        data.average_stack,
        diameter=diam,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=3,
    )
    n_rois = len(np.unique(masks)) - 1
    auto_diam = styles[0] if len(styles) > 0 else 0
    print(f"  diameter={diam}: detected {n_rois} ROIs, auto_diam={auto_diam:.1f}")

# Test with more permissive cellprob_threshold
print("\nTesting with permissive cellprob_threshold=-2.0:")
print("-" * 50)
for diam in [None, 4.5, 10]:
    masks, flows, styles = model.eval(
        data.average_stack,
        diameter=diam,
        flow_threshold=0.4,
        cellprob_threshold=-2.0,
        min_size=3,
    )
    n_rois = len(np.unique(masks)) - 1
    print(f"  diameter={diam}: detected {n_rois} ROIs")

# Compare with pretrained cyto3
print("\nComparing with pretrained cyto3:")
print("-" * 50)
pretrained = models.CellposeModel(gpu=True, model_type="cyto3")
for diam in [None, 4.5, 10]:
    masks, flows, styles = pretrained.eval(
        data.average_stack,
        diameter=diam,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        min_size=3,
    )
    n_rois = len(np.unique(masks)) - 1
    auto_diam = styles[0] if len(styles) > 0 else 0
    print(f"  diameter={diam}: detected {n_rois} ROIs, auto_diam={auto_diam:.1f}")
