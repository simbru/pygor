# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pygor is a Python framework for analyzing neurophysiology data originally processed in IGOR Pro. It provides data-object classes built on dataclasses that handle Baden-lab H5 file formats with structured analysis pipelines for retinal electrophysiology experiments.

**Key Philosophy**: Pygor is designed to work alongside IGOR Pro workflows, providing complementary Python analysis capabilities. Analysis notebooks typically live in separate repositories to avoid GitHub file size issues.

## Development Commands

### Build and Installation
```bash
# Build package
hatch build

# Install in editable mode for development
pip install -e .

# Alternative: Use uv for package management
uv sync  # Builds and installs into local .venv
.venv/scripts/activate  # or source .venv/bin/activate
```

### Testing
```bash
# Run comprehensive test suite (recommended)
python src/pygor/test/run_tests.py

# Alternative: Run all tests with unittest
PYTHONPATH=src python -m unittest discover src/pygor/test -v

# Run specific test file
PYTHONPATH=src python -m unittest src.pygor.test.test_STRF -v
PYTHONPATH=src python -m unittest src.pygor.test.test_Core -v

# Run Windows batch file (from test directory)
cd src/pygor/test && run_test.bat
```

**Test Suite Features:**
- Comprehensive validation of new STRF methods (extrema timing, spatial alignment)
- Edge case testing (empty arrays, invalid inputs, boundary conditions)
- Proper error handling validation (no more masked failures)
- Mock data generation for consistent testing
- Dependency checking and clear error reporting
- Memory efficiency validation for large datasets

### Code Quality
```bash
# Format and lint code (ruff is included in dependencies)
ruff check src/
ruff format src/
```

## Architecture and Design Principles

### Core Architecture Pattern
Pygor follows a **modular organization structure** where:
- **Simple methods** live in the corresponding class file (`src/pygor/classes/`)
- **Advanced functions** live in specialized directory structures
- **Class methods act as simple wrappers** that call complex functions from submodules

This design allows short-hand access to complex functionality while maintaining clean separation of concerns.

### Data Object Structure
- All data classes inherit from `pygor.classes.core_data.Core` 
- Classes are automatically discovered and loaded by `pygor.load.py`
- Classes are built using Python dataclasses with minimal configuration
- The `Experiment` class allows collation of multiple data objects for batch analysis

### Import System
Classes are imported via `from pygor.load import ClassName` which handles automatic discovery of all classes in the `classes/` directory, removing the need to navigate complex directory structures.

### Key Directory Structure
- `src/pygor/classes/`: Data class definitions (STRF, MovingBars, etc.)
- `src/pygor/strf/`: STRF-specific analysis modules (spatial, temporal, clustering, etc.)
- `src/pygor/timeseries/`: Time-series analysis for different stimulus types
- `src/pygor/plotting/`: Shared plotting utilities
- `src/pygor/core/`: Core functionality and GUI components
- `src/pygor/test/`: Unit tests for all classes

## Module-Specific Implementation Details

### STRF Module Critical Information
- **Array axes are always: [cell, time, y, x]** (indices 0, 1, 2, 3)
- **Multicolor data**: Multiple receptive fields belong to the same cell (denoted by `obj.numcolour`)
- **multidimensional_reshape function is crucial** for multicolor operations
- Complex analysis lives in submodules:
  - `strf/spatial.py`: Spatial analysis (balance ratio, opponency, centroids)
  - `strf/temporal.py`: Temporal analysis (time courses, spectral analysis)  
  - `strf/extrema_timing.py`: Pixel-wise timing analysis
  - `strf/spatial_alignment.py`: Color channel alignment analysis
  - `strf/clustering/`: Cell type classification
  - `strf/centsurr/`: Center-surround analysis

### Moving Bars Module
- Located in `timeseries/moving_bars/`
- `tuning_metrics.py`: Directional/orientation selectivity calculations
- Array structure follows standard conventions for directional tuning analysis

### Data Loading and File Conventions
Pygor maps IGOR wave names to Python attributes:

| IGOR Wave | Pygor Attribute |
|-----------|----------------|
| wDataCh0_detrended | images |
| Traces0_raw | traces_raw |
| traces_znorm | Traces0_znorm |
| ROIs | rois |
| Averages0 | averages |
| OS_Parameters[58] | frame_hz |
| Triggertimes | triggertimes |
| Positions | ipl_depths |

## Typical Analysis Workflow

### Standard Pipeline
1. **Data Processing in IGOR Pro**: Run OS_scripts pipeline with desired settings for data extraction and pre-processing
2. **Export to H5**: When satisfied with IGOR processing, export to H5 format
3. **Load in Pygor**: 
   ```python
   import pygor.load  # Shows available classes
   from pygor.load import STRF  # or other class names
   strf_obj = STRF("data_file.h5")
   ```
4. **Analysis**: Use object methods for further processing, statistics, and plotting
5. **No common recipes**: Users typically just load objects at the top of notebooks

### Class Discovery System
When importing `pygor.load`, it prints:
```
Found 7 custom classes in C:\Users\...\classes
Class names: ['CenterSurround', 'Core', 'Experiment', 'FullField', 'MovingBars', 'StaticBars', 'STRF']
Access custom classes using 'from pygor.load import ClassName'
```

## Important Implementation Guidelines

### Development Philosophy
- **Analysis organized by stimulus type**: New experimental stimulus methods should get new analysis objects
- **Extend existing functionality freely**: Users can extend existing classes and reuse functions across modules
- **Cross-module function reuse encouraged**: Functions can be shared between modules when appropriate
- **Methods should accept Pygor objects as inputs**: Complex methods work best when they can take Pygor objects directly

### Adding New Analysis Methods
1. Place complex logic in appropriate submodule (e.g., `strf/new_analysis.py`)
2. Create simple wrapper method in the corresponding class file
3. Follow the pattern: class method calls `submodule.function_name(self, ...)`
4. Maintain consistent parameter naming and return types
5. Use vectorized numpy operations for performance with large datasets

### Performance Considerations
- **Memory usage can be significant**: STRF data (x*y*time*roi*colour) commonly uses 32GB+ RAM
- **Always favor vectorization**: Both for performance and code elegance
- **Large datasets are common**: Design with scalability in mind

### Caching Pattern for Expensive Methods
**Standard Implementation**: For computationally expensive methods that are called multiple times with the same parameters, implement parameter-sensitive caching with manual recomputation option:

```python
def expensive_method(self, param1=None, param2=True, force_recompute=False, **kwargs):
    # Create cache key from parameters
    param1_key = tuple(param1) if param1 is not None and hasattr(param1, '__iter__') and not isinstance(param1, (str, int)) else param1
    cache_key = (param1_key, param2, tuple(sorted(kwargs.items())))
    
    # Check cache unless force_recompute
    if not hasattr(self, '_expensive_method_cache'):
        self._expensive_method_cache = {}
    
    if not force_recompute and cache_key in self._expensive_method_cache:
        return self._expensive_method_cache[cache_key]
    
    # Expensive computation here...
    result = compute_expensive_result()
    
    # Cache result before returning
    self._expensive_method_cache[cache_key] = result
    return result
```

**Key Benefits**:
- Eliminates redundant computation when same method called multiple times
- Parameter-sensitive: different parameter combinations cached separately
- Manual override: `force_recompute=True` bypasses cache when needed
- Especially important for methods called by multiple downstream functions

**Applied to**: `collapse_times()`, and should be applied to other expensive spatial/temporal analysis methods

### Multicolor Data Handling
**Legacy Design Issue**: Many methods were originally designed for ROI-centric analysis (one result per ROI) but multicolor data has multiple STRFs per ROI that need separate handling.

**Method Output Formats**:

**Already Multicolor-shaped** (no reshaping needed):
- `get_polarities()` - Uses multicolor reshape internally
- `collapse_times_chroma()` - Explicitly designed for multicolor
- `get_polarity_category_cell()` - Works with reshaped polarities
- Methods using `cs_seg()` results - Center-surround analysis handles multicolor

**Needs Multicolor Reshaping** (returns flattened ROI results):
- `get_time_amps()` - Returns one amplitude per ROI (flattened across channels)
- `get_space_amps()` - Returns one amplitude per ROI (flattened across channels)  
- `spatial_overlap_index_mean()` - Returns one correlation per ROI
- `spatial_polarity_index()` - Returns one index per ROI
- `collapse_times()` - Returns combined spatial data across all channels

**Reshaping Pattern**: For methods that need multicolor formatting:
```python
result = obj.method_name(**kwargs)
multicolor_result = pygor.utilities.multicolour_reshape(result, obj.numcolour)
# Shape changes from (n_rois,) to (n_colors, n_rois//n_colors)
```

**Auto-Generated `_by_channel` Methods**: For convenience, all methods that need multicolor reshaping automatically get a `_by_channel` variant that applies `multicolour_reshape()`:
```python
# Instead of manually reshaping:
result = obj.get_time_amps()
multicolor_result = pygor.utilities.multicolour_reshape(result, obj.numcolour)

# Just use the auto-generated method:
multicolor_result = obj.get_time_amps_by_channel()
```

**Available `_by_channel` methods** (auto-generated for all methods in `_NEEDS_MULTICOLOR_RESHAPING` list):
- `get_time_amps_by_channel()`
- `get_space_amps_by_channel()` 
- `spatial_overlap_index_mean_by_channel()`
- `spatial_overlap_index_std_by_channel()`
- `spatial_overlap_index_min_by_channel()`
- `spatial_overlap_index_var_by_channel()`
- `spatial_polarity_index_by_channel()`
- `collapse_times_by_channel()`

**Future Consideration**: Default multicolor output for new methods, with legacy compatibility options.

### STRF-Specific Development
- Always respect the [cell, time, y, x] axis convention
- For multicolor analysis, handle color channel indexing via `roi * numcolour + color_channel`
- Use threshold masking (typically ±3 SD) for significance testing
- Return simple arrays when possible, avoid complex nested dictionaries
- Include NaN masking for below-threshold pixels

### IGOR-Python Integration
- **Data validation against IGOR**: Not typically done (analysis extends beyond IGOR capabilities)
- **File format compatibility critical**: IGOR wave naming must be preserved (see core_data.py try_fetch functions)
- **Complementary workflow**: Pygor extends IGOR's OS_scripts functionality with Python's advanced analysis ecosystem

### Collaboration and Branching
- **Active branching**: Multiple branches actively worked on
- **Methods designed for convenience**: Both general-use and experiment-specific methods are acceptable
- **Documentation currently limited**: Small, intimate user base but expanding

### Testing Strategy
- Each class has corresponding test file in `test/` directory
- Tests should verify both single-object and batch processing functionality
- Include edge cases (no signal, multicolor vs single color, invalid indices)
- Use example data from `examples/` directory for testing

### Interactive Analysis Status
- **Currently limited**: Interactive analysis mostly done in IGOR/ImageJ
- **Napari integration exists**: But primarily for future interactive capabilities
- **Current focus**: Batch processing and visualization (plotting)
- **Future goal**: All-in-one platform with Napari as interactive layer

The framework is designed for extensibility while maintaining simple user interfaces for complex neurophysiology analysis pipelines.