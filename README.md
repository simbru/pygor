# Pygor: Pickup in Python where you left of in IGOR, for Baden-lab members

Welcome to Pygor!

Your one-stop shop for fetching Baden-lab processed IGOR data via H5 files, and transforming them into a flexible yet structured analysis framework in Python.

## Here's the main points:

- Pygor consists of data-objects, data-object-specific directories, and an Experiment (name to be decided) class that allows you to collect data-objects and analyse them using the data-directories.
- Pygor classes are built using [dataclasses](https://docs.python.org/3/library/dataclasses.html), which are simple Python classes which store information and methods with minimal configuration and templating.
- The special Pygor class `pygor.classes.experiment.Experiment` provides a handy way to collate Pygor objects, in such a way that analyses can be ran on arbitrarily many datasets at a time.
- Pygor classes can be built by inheriting the `pygor.classes.core_data.Core` object, which comes with handy methods like plotting your scan average, ROI positions, amd gettomg contextual help with parsing the data and methods availible in your new object.
- Pygor objects can be called simply by passing `from pygor.load import "class name"`, as the import logic dictated by `pygor.load.py` takes care of the potentially confusing (and mostly just annoying) navigation of the directory structure, which can serve as a barrier of entry to novice users.
- Extending the functionality of Pygor is intended to be *simple*. There are certain design princples the user can follow in order to build out their own analyses pipelines, that can be shared and further improved by other users.

## How do I install Pygor

Currently, the package will very likely require you to do your own development. This sounds scary, but its pretty easy

1. Install [hatch](https://hatch.pypa.io/latest/) on your computer, either via executable or via pip in your environment.
2. Download Pygor from the Github repo (I prefer using `git clone https://github.com/simbru/pygo` in my target directory. Moving files around manually should also work (uncertain if Git functionality will work).
3. Open up your favorite command line (CMD for example) and CD to your Pygor directory. You will know  you are in the right spot if you see a file called `pyproject.toml`
4. Simply run `hatch build` from the command line inside the directroy. You should see some reference to "building wheels". This means you're on the rigth track.
5. Once that is done, simply stay in the directroy, activate whatever Python envrionment you want to use, and type `pip install -e .` -> This will allow you to use Pygor, while changing the contents of Pygor's files.

That's it! Activate your environment in your favorite IDE and get going with Pygor! Please flag it if you run across any issues. 

*Alternatively, you should be able to simply download the Git repositroy, set your working directories correctly, and be on your merry way (but this is completely untested, and you might have to move your Jupyter notebooks around for imports to work properly).*

## Pygor design principles

Inside `pygor/src/pygor/` (Python's way of structuring a package with sub-modules), you will find various files and folders.

- `pygor/classes`: This is where the dataclasses live. Each class gets its own .py file, and classes are automatically identified and loaded by `pygor.load.py` when it is imported.
- `pygor/plotting`: This is where shared plotting-related scripts live
- `pygor/docs`: Documentation will live here
- `pygor/shared`: Other shared scripts
- `pygor/test`: Unittests for Pygor classes
- `pygor/insert_your_class`: Files containing functions related to your packages!

## Pygor naming conventions for IGOR waves:

| IGOR Wave          | Pygor core attribute |
| -------------------- | ---------------------- |
| wDataCh0_detrended | images               |
| Traces0_raw          | traces_raw             |
| traces_znorm          | Traces0_znorm             |
| ROIs          | rois             |
| Averages0          | averages             |
| Snippets0          | snippets             |
| OS_Parameters[58]  | frame_hz             |
| OS_Parameters[28]  | trigger_mode             |
| Triggertimes_Frame          | triggerstimes_frame             |
| Triggertimes          | triggertimes             |
| Positions          | ipl_depths             |

Eventually, it would make sense to create a framework for customising this, such that the pipeline can be adapted more broadly to other H5 files with other naming conventions. For now, this will wait until it is a requested feature. 