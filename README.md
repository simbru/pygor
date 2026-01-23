# Pygor: OS_scirpts (IGOR) style workflows in Python, for Baden-lab members

Welcome to Pygor!

A Python toolkit for working alongside Baden-lab's IGOR Pro pipeline, allowing imports of processed data via H5 files and extending analysis capabilities with Python's flexible framework.

NEW: You can now read directly from .smh/.smp files! This functionality is being tested and validated. 

## Here are the main points:

- Pygor consists of data-objects, data-object-specific directories, and an Experiment (name to be decided) class that allows you to collect data-objects and analyse them using the data-directories.
- Pygor classes are built using [dataclasses](https://docs.python.org/3/library/dataclasses.html), which are simple Python classes that store information and methods with minimal configuration and templating.
- The special Pygor class `pygor.classes.experiment.Experiment` provides a handy way to collate Pygor objects, in such a way that analyses can be run on arbitrarily many datasets at a time.
- Pygor classes can be built by inheriting the `pygor.classes.core_data.Core` object, which comes with handy methods like plotting your scan average, ROI positions, and getting contextual help.
- Pygor objects can be called simply by passing `from pygor.load import "class name"`, as the import logic dictated by `pygor.load.py` takes care of the potentially confusing (and mostly just annoying) navigation of the directory structure, which can serve as a barrier of entry to novice users.
- Extending the functionality of Pygor is intended to be *simple*. There are certain design principles the user can follow in order to build out their own analyses pipelines, that can be shared and further improved by other users.

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
data.segment_rois(mode="cellpose+")  # Cellpose with post-processing heuristics
data.segment_rois(mode="cellpose")   # Raw Cellpose output
```

## Pygor design principles

Inside `pygor/src/pygor/` (Python's way of structuring a package with sub-modules), you will find various files and folders.

- `pygor/classes`: This is where the dataclasses live. Each class gets its own .py file, and classes are automatically identified and loaded by `pygor.load.py` when it is imported.
- `pygor/plotting`: This is where shared plotting-related scripts live
- `pygor/docs`: Documentation will live here
- `pygor/shared`: Other shared scripts
- `pygor/test`: Unittests for Pygor classes
- `pygor/insert_your_class`: Files containing functions related to your packages!

## Custom cellpose models
Pre-trained models can be found here: https://drive.proton.me/urls/02GT9HWGC0#Ibd9kXWzOwMQ
(So far only RibeyeA)

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

## AI transparency
Pygor's core functionality was built before widespread AI coding tools. Recent development has used ChatGPT, Claude Code, and Github Copilot.

Contributors should comment AI-generated code sections and include this in commit messages. Always test and validate AI-assisted code thoroughly.  