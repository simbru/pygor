# Pygor: OS_scirpts (IGOR) style workflows in Python, for Baden-lab members

Welcome to Pygor!

Version: 1.0

A Python toolkit for working alongside Baden-lab's IGOR Pro pipeline. Pygor allows imports of IGOR-processed data via H5 files or direct reading of .smh/.smp files from ScanM. This allows you to use Python for your data analysis.

## Key concepts

- **Data classes**: Load recordings via `pygor.load.Core()`, `pygor.load.STRF()`, etc. Classes inherit from `Core` and include built-in methods for visualization and analysis.
- **Experiment collections**: Group multiple recordings with `pygor.classes.experiment.Experiment` for batch analysis.
- **Extensible**: Add custom analysis objects by inheriting from existing classes.

## How do I install Pygor?

### Sync with uv (preferred)
A new package management framework called uv simplifies installation, but makes 
a local virtual environment that you must use instead of making your own. 
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Run `git clone https://github.com/simbru/pygor` into target directory
3. Change directory into the pygor folder, and run `uv sync`. This should build Pygor and install it into the local uv-managed .venv folder.
4. Use `.venv/scripts/activate` to activate the environment.

More information on [uv by Astral here](https://docs.astral.sh/uv/).

## Using Pygor

### Basic usage
After installing Pygor, you can import classes like so:

```python
import pygor.load

# Load Core data object from H5 file or SMH/SMP file
data = pygor.load.Core("recording.h5") # or use another class name (e.g., STRF, OSDS, or ResponseMapping)

# Access attributes
print(data.frame_hz)
print(data.num_rois)
print(data.traces_znorm.shape)

# Use methods
data.view_stack_rois()
data.get_help() # Get contextual help for the loaded class
```

### Example data and pipelines
Example pipelines and demo data

To avoid issues with Git LFS, demo data can be downloaded from here: 
https://drive.google.com/drive/folders/1LtO1XTIgLIYS6jkh-3tXqc7XdXKJ9OSh


Usage examples can be found in the `/pygor/examples` folder:
- `preprocessing_data.py` - Example of loading data and preprocessing steps including detrending, registration, and ROI segementation (using Cellpose).
- `example_notebook.ipynb` - Walkthrough of basic Pygor functionality in a Jupyter notebook.

Place the strf_demo_data.h5 file into the /pygor/examples folder and you can follow the .ipynb and .py files in there.

### ROI Segmentation
Pygor provides automated ROI segmentation through `data.segment_rois()`. Several methods are available:

**Lightweight methods (no ML required):**
- `watershed` - Local maxima seeded watershed segmentation
- `flood_fill` - IGOR-style region growing from peaks
- `blob` - Difference of Gaussian blob detection

```python
data = pygor.load.Core("recording.h5")
data.segment_rois(mode="watershed")  # or "flood_fill", "blob"
```


**Cellpose:**
Pygor supports Cellpose and custom models for ROI segmentation. 

See "Cellpose (optional, ML-based)" below.

## Optional dependencies

### Jupyter Notebooks (optional)
For interactive analysis in Jupyter notebooks, install the notebook extras:
```bash
uv pip install 'pygor[notebook]'
```

This installs `jupyter`, `notebook`, `ipykernel`, and `ipywidgets`. After installation, register the kernel for use in notebooks:
```bash
python -m ipykernel install --user --name=pygor --display-name="Pygor"
```

Then select "Pygor" as your kernel in Jupyter or VS Code.

### Napari GUI (optional)
For interactive visualization and ROI drawing with napari:
```bash
uv pip install 'pygor[gui]'
```

This enables methods like `data.view_stack()`, `data.view_stack_rois()`, and `data.prompt_ipl_depths()`.

### Cellpose (optional, ML-based):**
Cellpose provides high-quality ML-based segmentation but is a heavyweight dependency (PyTorch, CUDA). It is **not installed by default**.

To install Cellpose support:
```bash
uv pip install 'pygor[cellpose]'
```

Then use with:
```python
data.segment_rois(mode="cellpose")   # Raw Cellpose output
data.segment_rois(mode="cellpose+", model_path="your\path\here")  # Cellpose with custom model and post-processing 

```

Pre-trained models can be found here: https://drive.proton.me/urls/02GT9HWGC0#Ibd9kXWzOwMQ
(So far only RibeyeA)


## Pygor design principles

Inside `pygor/src/pygor/` (Python's way of structuring a package with sub-modules), you will find various files and folders.

- `pygor/classes`: This is where the dataclasses live. Each class gets its own .py file, and classes are automatically identified and loaded by `pygor.load.py` when it is imported.
- `pygor/plotting`: This is where shared plotting-related scripts live
- `pygor/docs`: Documentation will live here
- `pygor/shared`: Other shared scripts
- `pygor/test`: Unittests for Pygor classes
- `pygor/insert_your_class`: Files containing functions related to your packages!

## H5/IGOR Wave to Pygor Attribute Mapping
If you are used to IGOR Pro and OS_scripts, you might have noticed some differences in naming conventions between IGOR Waves and Pygor. This was done to make the code more "Pythonic" and to improve ease of use. Below is a mapping of common H5 keys to their corresponding Pygor attributes, along with brief descriptions.

### Data Arrays

| H5 Key | Pygor Attribute | Description |
|--------|-----------------|-------------|
| wDataCh0_detrended | images | Preprocessed imaging stack (T, Y, X) |
| wDataCh2 | trigger_images | Trigger channel images |
| Traces0_raw | traces_raw | Raw ROI traces |
| Traces0_znorm | traces_znorm | Z-normalized ROI traces |
| ROIs | rois | ROI mask array |
| RoiSizes | roi_sizes | Pixel count per ROI |
| Averages0 | averages | Epoch-averaged traces |
| Snippets0 | snippets | Trial snippets |
| Triggertimes | triggertimes | Stimulus trigger times |
| Triggertimes_Frame | triggertimes_frame | Trigger times in frame units |
| Positions | ipl_depths | IPL depth per ROI |
| QualityCriterion | quality_indices | ROI quality scores |
| correlation_projection | correlation_projection | Pixel correlation map |
| Stack_Ave | average_stack | Time-averaged image |

### OS_Parameters (indexed array with named keys)

| OS_Parameters Key | Pygor Attribute | Description |
|-------------------|-----------------|-------------|
| Trigger_Mode | trigger_mode | Triggers per stimulus epoch |
| nPlanes | n_planes | Number of imaging planes |
| LineDuration | linedur_s | Line scan duration (seconds) |
| Skip_First_Triggers | _Core__skip_first_frames | Triggers to skip at start |
| Skip_Last_Triggers | _Core__skip_last_frames | Triggers to skip at end |

Note: `frame_hz` is calculated from `n_planes` and `linedur_s`, not stored directly.

<!-- ### STRF-specific (strf_data.py)

| H5 Key Pattern | Pygor Attribute | Description |
|----------------|-----------------|-------------|
| STRF{n}_{roi}_{colour} | strfs | Receptive field arrays |
| Noise_FilterLength_s | strf_dur_ms | STRF duration (converted to ms) |  -->

## AI transparency and guidelines
Pygor's core functionality was built before widespread AI coding tools. However, recent development has used ChatGPT, Claude Code, and Github Copilot.

Guidelines:
- Refrain from committing LLM meta files (e.g., .chatgpt, .claude) to the repository. Do not commit AI conversation logs or markdown files the LLM produced for itself.
- Formatting, quality, and style standards must be met.
- Outputs and changes must be removed and assessed by a human.
- Large sections of AI-generated code should be disclosed and included in commit messages. 
- Always test and validate AI-assisted code thoroughly.

## TODO
- Expand documentation and examples
- Add tiff reading support and trigger channel deinterleaving

## Acknowledgements

Pygor builds on excellent open-source tools:

- **[Napari](https://napari.org/)** - Multi-dimensional image viewer for Python. Used for interactive visualization and ROI annotation.
  - Ahlers et al., (2023). napari: a multi-dimensional image viewer for Python. Zenodo. https://doi.org/10.5281/zenodo.8115575
- **[Cellpose](https://www.cellpose.org/)** - Deep learning-based cell segmentation. Used for automated ROI detection.
  - Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.