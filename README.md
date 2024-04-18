# Pygor: Pickup in Python where you left of in IGOR, for Baden-lab members

The intent of this package is to enable Pythonic post-processing, analysis, and plotting of data 2-photon which has been pre-processed using the Baden-lab pipeline written in IGOR Pro 6. 

Many of us (relative) new-comers are already familiar with Python, object-oriented programming, and the expansive number of availible packages for analysis, statistics, data-organisation, and plotting. As such, Pygor is written to extend the IGOR pre-processing pipeline into Python, by tapping into the data-structures (IGOR *waves*) and pre-defined data variables already established within the IGOR. To interface the two, Pygor takes as input .H5 files that are exported from IGOR. These are fairly universal files for storing structured data, with good support on both the IGOR and Python side. Becasue the IGOR pipeline is written with conventions for naming waves, Pygor is designed to pick out certain names (aka *keys*) that correspond to IGOR *waves* from the H5 file. Moreover, customising the export script for H5 files in the current IGOR pipelines is fairly simple, and I plan on writing a guide and how-to on doing this. 

This gets at another principle of Pygor. The intention is for it to be democratic, in the sense that its users (thats us!) contribute what we work on, so that others can use it later on. This not only improves consistency across the lab, but also holds each of us accountable for doing our best when it comes to writing our anaylses. If errors are spotted by others, they can simply be raised as an issue and corrected (or corrected directly). This also means that for each new type of analysis/experiment, the user is expected do develop their own sub-module. Again, I plan on creating guides on how to do this. My intention is to set everything up so that this process is as smooth and simple as possible, potentially allowing even novice Python users to start developing their anaylses.

As an example, my project revolves around anaylsing STRFs. Hence, you will find the following directories/files that correspond to this sub-module:
- `pygor/classes/strf_data.py`: contains the dataclass for instantiating STRF-objects (data-structure)
- `pygor/strf/`: sub-module containing all scripts for analyses, plotting, etc related to STRF-objects
- `pygor/test/test_STRF.py`: a unittesting file that can be ran to check for errors with the internal logic of the STRF-object

In principle, a user who wishes to expand on the functionality of Pygor only has to create a sub-module (such as `pygor/strf/`) to formalise their analysis. However, in some cases, new object methods or attributes may be required, in which case I encourage the user to write their own dataclass (like `pygor/classes/strf_data.py`). More on this to come.

For now, I will keep tinkering away with this library.