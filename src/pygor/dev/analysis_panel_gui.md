# IGOR-Style Analysis Panel for Pygor

*Created: 2026-03-23*
*Related: [gui_notes_and_plans.md](gui_notes_and_plans.md), [configs_and_settings.md](configs_and_settings.md)*

## Motivation

IGOR Pro provides a step-by-step GUI panel (ImageProc) where users click through
analysis stages sequentially. Each button runs one processing step. Users can
inspect results, tweak parameters, and re-run individual steps. This is intuitive
for interactive exploration of imaging data.

Pygor now has the backend infrastructure to support this:
- `Experiment.run()` and per-recording method calls
- `params.edit()` — IGOR-style editable parameter table (Qt, already built)
- `params["dotted.path"]` — bracket access to all defaults
- `_defaults` as single source of truth (no desync issues)
- Napari integration for ROI drawing and stack viewing

## Design

### Panel Layout

A single `QWidget` window with grouped steps, matching pygor's pipeline order:

```
 AnalysisPanel — {experiment_name}
 ──────────────────────────────────────
 Step 1: Load Data
   [ScanM files...]  [H5 files...]  [.pygor.h5]
   Loaded: 4 recordings (STRF)

 Step 2: Preprocessing
   [Register]  [Preprocess]  [Reset]
   Status: 2/4 registered, 0/4 preprocessed

 Step 3: Parameters
   [Edit All]  [Edit Section...]
   Config: package defaults

 Step 4: ROI Segmentation
   [Blob]  [Watershed]  [Flood Fill]  [Cellpose]  [Draw]
   ROIs: rec0=103, rec1=45, rec2=—, rec3=—

 Step 5: Traces & Triggers
   [Extract Traces]  [Set Trigger Mode...]  [Snippets]

 Step 6: Analysis
   [Calculate STRF]  [Transfer ROIs to...]
   [Compute Tuning]  [Chromatic Overview]

 Step 7: Export
   [Save .pygor.h5]  [View Stack]  [Plot Overview]

 ──────────────────────────────────────
 Log: Registered rec 0_0 (shift: 0.3px, error: 0.001)
```

### Architecture

```
pygor/core/gui/
    param_editor.py      # Already built — params.edit()
    analysis_panel.py    # NEW — main panel widget
    _panel_steps.py      # NEW — per-step widget classes (optional, for organisation)
```

### Key Implementation Details

#### State tracking

Each step group reads state from the recordings themselves:
```python
# Grey out "Extract Traces" until ROIs exist
has_rois = any(rec.rois is not None for rec in exp.recording)
self.btn_extract.setEnabled(has_rois)

# Show status from params
n_registered = sum(1 for r in exp.recording if r.params.registered)
self.lbl_status.setText(f"{n_registered}/{len(exp.recording)} registered")
```

No separate state machine needed — `rec.params.preprocessed`, `rec.params.registered`,
`rec.params.segmented` already track this.

#### Button wiring

Each button calls `Experiment.run()` or per-recording methods in a worker thread:
```python
self.btn_register.clicked.connect(lambda: self._run_step("register", plot=True))
self.btn_preprocess.clicked.connect(lambda: self._run_step("preprocess"))
self.btn_segment_blob.clicked.connect(lambda: self._run_step("segment_rois", mode="blob"))

def _run_step(self, method, **kwargs):
    """Run method on experiment in a QThread to keep GUI responsive."""
    self.worker = ExperimentWorker(self.experiment, method, **kwargs)
    self.worker.finished.connect(self._refresh_status)
    self.worker.error.connect(self._show_error)
    self.worker.start()
```

#### Per-recording controls

For steps that need per-recording params (like segmentation threshold varying
by image quality), the panel should support:
- Click recording name to select it
- "Edit params" opens `rec.params.edit("segmentation")` for that recording
- Run step on selected recording only, or all

#### Integration with params.edit()

"Edit All" button opens `params.edit()` for the first recording.
"Edit Section..." shows a dropdown of sections, then opens `params.edit(section)`.
Changes are live — the panel's next "Run" call picks them up automatically
because methods read from `_defaults`.

#### File loading

```python
def _load_files(self, file_type="smp"):
    paths, _ = QFileDialog.getOpenFileNames(
        self, "Select files",
        filter="ScanM (*.smp *.smh);;HDF5 (*.h5);;Pygor (*.pygor.h5)"
    )
    if paths:
        self.experiment = Experiment.from_files(paths, self.pygor_class)
        self._refresh_all()
```

### Effort Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| Basic panel layout + buttons | 2-3 hours | QGroupBox + QPushButton grid |
| Button → method wiring | 1-2 hours | Straightforward signal/slot |
| State tracking + status labels | 2-3 hours | Read from params, refresh on step completion |
| QThread worker for long ops | 1-2 hours | Prevent GUI freeze |
| Per-recording selection | 2-3 hours | List widget or tabs |
| File loading dialog | 1 hour | QFileDialog |
| Error display + log area | 1 hour | QTextEdit at bottom |
| **Total basic version** | **~1-2 days** | |
| Progress bars | 2-3 hours | Needs callback from methods |
| Plot integration | 3-4 hours | Embed matplotlib or launch separate windows |
| **Total polished version** | **~3-4 days** | |

### Prerequisites

- [x] `params.edit()` — interactive parameter table
- [x] `params["dotted.path"]` — bracket access
- [x] `_defaults` as single source of truth
- [x] `Experiment.run()` — batch method execution
- [x] `transfer_rois_from()` — cross-experiment ROI transfer
- [ ] `%gui qt` auto-activation (done for params.edit, reuse pattern)

### Open Questions

1. **Panel per Experiment or per Recording?** — Per Experiment makes more sense
   (matches IGOR's "one panel, multiple recordings" model). Show a recording
   selector/list on the side.

2. **Standalone window or napari plugin?** — Standalone QWidget is simpler and
   doesn't require napari to be running. Can launch napari for ROI drawing
   on demand.

3. **How to handle paired experiments?** — "Transfer ROIs to..." button opens
   a dialog to select/load the source Experiment. Uses `transfer_rois_from()`.
