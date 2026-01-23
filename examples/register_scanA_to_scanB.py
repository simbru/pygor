"""
Example: Register one scan to another and transfer ROIs.

This script demonstrates how to:
1. Load two scans from the same field of view
2. Preprocess and segment ROIs on scanA (reference)
3. Register scanB to scanA's coordinate frame
4. Transfer ROIs from scanA to scanB
"""

import numpy as np
import pygor.load


# Load your data - replace these paths with your actual file paths
scanA_path = r"path/to/your/reference_scan.smp"  # Reference scan (will define ROIs)
scanB_path = r"path/to/your/target_scan.smp"     # Target scan (will receive ROIs)

scanA = pygor.load.Core(scanA_path)
scanB = pygor.load.Core(scanB_path)

# Process reference scan (scanA)
scanA.preprocess(detrend=False) #optional detrending
scanA.register(plot=True)
scanA.segment_rois(mode="watershed", plot=True)  # Or use mode="cellpose" if installed

# Register target scan (scanB) to reference
scanB.preprocess(detrend=False)
scanB.register(ref_plane=scanA.average_stack, plot=True)

# Transfer ROIs from scanA to scanB so they are 1:1 in identity + aligned
scanB.transfer_rois_from(scanA, plot=True)

# Verify alignment quality
img_A = scanA.images.mean(axis=0)
img_B = scanB.images.mean(axis=0)
correlation = np.corrcoef(img_A.flatten(), img_B.flatten())[0, 1]
print(f"Pearson correlation between mean images: {correlation:.4f}")