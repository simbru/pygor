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

### Build with hatch and install with pip (old method)
Currently, the package will very likely require you to do your own development. This sounds scary, but it's pretty easy. For that, we need
to install the latest version of the package in an "editable" state.

1. Install [hatch](https://hatch.pypa.io/latest/) on your computer, either via executable or via pip in your environment.
2. Download Pygor from the Github repo (I prefer using `git clone https://github.com/simbru/pygor` in my target directory).
3. Open up your favorite command line (CMD for example) and change directory to your Pygor folder. You will know  you are in the right spot if you see a file called `pyproject.toml`
4. Simply run `hatch build` from the command line inside the directory. You should see some reference to "building wheels". This means you're on the right track.
5. Once that is done, simply stay in the directory, activate whatever Python environment you want to use, and type `pip install -e .` -> This will allow you to use Pygor, while changing the contents of Pygor's files (editable).

That's it! Activate your environment in your favorite IDE and get going with Pygor! Please flag it if you run across any issues. 

*Alternatively, you should be able to simply download the Git repository, set your working directories correctly, and be on your merry way (but this is completely untested, and you might have to move your Jupyter notebooks around for imports to work properly).*

## Pygor design principles

Inside `pygor/src/pygor/` (Python's way of structuring a package with sub-modules), you will find various files and folders.

- `pygor/classes`: This is where the dataclasses live. Each class gets its own .py file, and classes are automatically identified and loaded by `pygor.load.py` when it is imported.
- `pygor/plotting`: This is where shared plotting-related scripts live
- `pygor/docs`: Documentation will live here
- `pygor/shared`: Other shared scripts
- `pygor/test`: Unittests for Pygor classes
- `pygor/insert_your_class`: Files containing functions related to your packages!

## Custom cellpose models
Models can be found here: https://drive.proton.me/urls/02GT9HWGC0#Ibd9kXWzOwMQ
(So far only RibeyeA)

## H5/IGOR Wave to Pygor Attribute Mapping

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
The core functionality of Pygor was built before the widespread availability of AI coding assistants. However, functions and modules have since been improved and optimised using LLM tools like ChatGPT, Claude Code, and Github Copilot. I do my best to ensure that all AI-assisted code is properly tested and validated. 

I encourage contributors to follow these guidelines:
- Clearly comment any sections of code that were entirely generated or significantly assisted by AI tools. Also include in commit messages where large chunks of AI-generated code were added.
- Snippets or functions that were only lightly edited do not need special comments, unless you feel it is necessary for clarity.
- Always thoroughly test and validate AI-generated code. You have priors about how the code should behave.
- Strive to write clear, maintainable code, regardless of whether AI tools were used. Sometimes AI writes less optimal code that needs human refinement, especially with regards to naming conventions. 
- AI-generated docstring comments should be reviewed and edited for accuracy and clarity.
- When in doubt, err on the side of transparency about AI assistance.