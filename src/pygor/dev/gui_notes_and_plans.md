# Pygor GUI: Notes, Analysis, and Design Considerations

*Document created: 2026-01-09*
*Updated: 2026-01-09 - Separated preprocessing into own document*

---

## Related Documents

- **[preprocessing_plan.md](preprocessing_plan.md)** - Core preprocessing module (SMP/SMH loading, registration, detrending, H5 export)
- **[tandem_workflow.md](tandem_workflow.md)** - Tandem analysis workflow (TandemPair, TandemGroup, TandemBatch)

---

## 0. The Real Goal: Replace IGOR Preprocessing

**Key insight:** The goal isn't "add a GUI to pygor" - it's **consolidate the entire workflow into Python** and eliminate the IGOR dependency.

### Current vs Target Data Flow

```
CURRENT:                          TARGET:
SMP/SMH → IGOR → H5 → Pygor       SMP/SMH → Pygor (GUI) → H5
```

### Desired Tandem Workflow

```
Open Recording A → Register → Draw ROIs → Extract Traces → Save
                                 ↓
                         Auto-apply to Recording B
                         (with shift correction via autocorrelation)
```

---

## 1. Implementation Phases

| Phase | Component | Document | Status |
|-------|-----------|----------|--------|
| **1** | Preprocessing module | [preprocessing_plan.md](preprocessing_plan.md) | Planning |
| **2** | Tandem workflow | [tandem_workflow.md](tandem_workflow.md) | Planning |
| **3** | GUI | This document | Planning |
| **4** | Batch processing | Included in tandem_workflow.md | Planning |

**Important:** Phase 1 (preprocessing) must work standalone before GUI work begins.

---

## 2. Current Pygor Capabilities (Detailed)

### Already Implemented

| Component | Location | Notes |
|-----------|----------|-------|
| **ROI Drawing** | `core.gui.methods.NapariRoiPrompt` | Full Napari GUI, polygon/ellipse, real-time traces |
| **ROI I/O** | `core.gui.methods` | Load/save to H5, format conversion |
| **Trace Extraction** | `core.gui.methods.compute_traces_from_rois()` | Parallel, shared memory, overflow-safe |
| **IPL Depth Selection** | `core.gui.methods.NapariDepthPrompt` | Interactive boundary drawing |
| **H5 Updates** | `core.methods.update_h5_key()` | Append/overwrite existing H5 |
| **STRF Spatial Alignment** | `strf.spatial_alignment` | Correlation, Jaccard, centroid comparison |
| **Multi-color Support** | `classes.strf_data` | Auto-detect, reshape, per-channel analysis |

### Not Implemented (Need to Build)

| Component | Priority | Complexity | Notes |
|-----------|----------|------------|-------|
| **TIFF Loading** | HIGH | Low | Use tifffile or scikit-image |
| **Image Registration** | HIGH | Medium | scikit-image has good tools |
| **ROI Transfer** | HIGH | Medium | Cross-correlation offset + apply |
| **Detrending** | HIGH | Low | Polynomial fit, rolling baseline |
| **H5 Creation** | MEDIUM | Low | Structure template from IGOR |
| **Parameter Table** | MEDIUM | Low | Dataclass or dict |
| **Trigger Detection** | MEDIUM | Medium | From stimulus channel |

---

## 3. Tandem Analysis Architecture

### Core Concept

```
┌─────────────────┐     ┌─────────────────┐
│  Recording A    │     │  Recording B    │
│  (e.g., SWN)    │     │  (e.g., Bars)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
   ┌───────────┐                 │
   │ Register  │                 │
   │ Frames    │                 │
   └─────┬─────┘                 │
         │                       │
         ▼                       │
   ┌───────────┐                 │
   │ Draw ROIs │                 │
   │ (Napari)  │                 │
   └─────┬─────┘                 │
         │                       │
         ▼                       ▼
   ┌───────────┐           ┌───────────┐
   │ Extract   │           │ Auto-     │
   │ Traces A  │           │ Register  │
   └─────┬─────┘           │ to A      │
         │                 └─────┬─────┘
         │                       │
         │                       ▼
         │                 ┌───────────┐
         │                 │ Transfer  │
         │                 │ ROIs      │
         │                 │ (+ shift) │
         │                 └─────┬─────┘
         │                       │
         │                       ▼
         │                 ┌───────────┐
         │                 │ Extract   │
         │                 │ Traces B  │
         │                 └─────┬─────┘
         │                       │
         ▼                       ▼
   ┌─────────────────────────────────┐
   │     Unified Analysis            │
   │  (same cells, different stim)   │
   └─────────────────────────────────┘
```

### ROI Transfer Algorithm

```python
def transfer_rois(roi_mask_a, stack_a, stack_b):
    """
    Transfer ROIs from recording A to recording B with shift correction.

    1. Compute mean projection of both stacks
    2. Find offset via cross-correlation
    3. Shift ROI mask by offset
    4. Return shifted mask for B
    """
    from scipy.ndimage import shift
    from skimage.registration import phase_cross_correlation

    mean_a = stack_a.mean(axis=0)
    mean_b = stack_b.mean(axis=0)

    # Find shift between recordings
    offset, _, _ = phase_cross_correlation(mean_a, mean_b)

    # Apply shift to ROI mask
    roi_mask_b = shift(roi_mask_a, offset, order=0, mode='constant', cval=1)

    return roi_mask_b, offset
```

---

## 4. Proposed GUI Design (Revised)

### Target: Preprocessing + Tandem Workflow

```
┌────────────────────────────────────────────────────────────┐
│  PYGOR PREPROCESSING                                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Step 1: Load Data                                   │  │
│  │  [Browse TIFF] [Browse H5]  Path: ____________      │  │
│  │  [x] Load as tandem pair    Partner: ___________    │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Step 2: Registration                                │  │
│  │  [Register Frames]  Method: [Phase Correlation ▼]   │  │
│  │  Reference: [First frame ▼]  Max shift: [10] px     │  │
│  │  [Preview] [Apply]                                   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Step 3: ROI Placement                               │  │
│  │  [Draw ROIs]  [Auto-detect (SD)]  [Auto (Corr)]     │  │
│  │  ROIs: 42     [View]  [Edit]  [Clear]               │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Step 4: Extract & Save                              │  │
│  │  [Extract Traces]  [████████████░░░░] 75%           │  │
│  │  [x] Apply to tandem   Offset detected: (2, -1) px  │  │
│  │  [Save H5]  [Export to Notebook]                    │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Status: Ready                                       │  │
│  │  [Log window...]                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Tandem-aware from the start** - Load pairs, not singles
2. **Visual feedback** - Preview registration, see ROI count
3. **Progress tracking** - Long operations show progress
4. **Notebook export** - Generate starter notebook with paths filled in

---

## 5. GUI Design

> **Note:** Preprocessing implementation details are in [preprocessing_plan.md](preprocessing_plan.md)

### Target: Preprocessing + Tandem Workflow GUI

The GUI wraps the preprocessing module to provide:
1. File browsing and loading (SMP/SMH or H5)
2. Registration controls with preview
3. ROI tools (launch Napari, view count)
4. Tandem pairing and batch operations
5. Progress tracking for long operations
6. H5 export

### Framework: PySide6 (Native)

Reasons:
- Napari is already Qt-based - seamless integration
- No browser/server overhead
- Better file dialog integration
- Works offline

```toml
[project.optional-dependencies]
gui = ["PySide6>=6.5"]
```

---

## 6. Open Questions

1. **Multi-plane handling** - Defer to later (colleagues handle differently)

2. **Stimulus pairing** - Options:
   - Auto-detect by filename pattern (`0_0_SWN` ↔ `0_0_Bars`)
   - User multi-selects from loaded list
   - **Decision:** User selects (more flexible, less magic)

3. **Backwards compatibility** - H5 output should match IGOR exactly (assumed yes)

---

## 7. Architecture Overview

```
pygor/
├── preprocessing/           # Phase 1 - see preprocessing_plan.md
│   ├── scanm.py            # SMP/SMH loading
│   ├── registration.py     # Frame registration + ROI transfer
│   ├── detrend.py          # Bleach correction
│   ├── triggers.py         # Trigger detection
│   └── export.py           # H5 file creation
│
├── tandem/                  # Phase 2 - see tandem_workflow.md (TODO)
│   ├── pair.py             # TandemPair class
│   └── batch.py            # TandemBatch for multiple pairs
│
├── gui/                     # Phase 3 - this document
│   ├── main_window.py      # PySide6 main window
│   ├── widgets/            # Reusable UI components
│   └── napari_integration.py
│
├── classes/                 # EXISTING
├── core/                    # EXISTING (has gui/methods.py for Napari)
└── ...
```

---

## 8. GUI-Specific Next Steps

1. [ ] Complete preprocessing module (Phase 1)
2. [ ] Design tandem workflow API (Phase 2)
3. [ ] Prototype minimal PySide6 window
4. [ ] Add file browser widget
5. [ ] Integrate Napari ROI tools
6. [ ] Add progress tracking for long operations
7. [ ] Test Qt/Napari integration
