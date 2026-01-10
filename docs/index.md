# Pygor Documentation

Welcome to the Pygor documentation. Pygor enables Pythonic post-processing, analysis, and plotting of 2-photon calcium imaging data.

## Quick Start

```python
from pygor.classes.core_data import Core

# Load directly from ScanM files (bypasses IGOR)
data = Core.from_scanm("recording.smp", preprocess=True)

# Or load from IGOR-exported H5
data = Core("recording.h5")

# View the data
data.view_images_interactive()
```

## Documentation

### Core Concepts

- [Purpose & Philosophy](purpose.md) - Why Pygor exists and design principles
- [Getting Started](getting_started.md) - Installation and first steps

### Modules

- **[Preprocessing](modules/preprocessing.md)** - ScanM loading, trigger detection, detrending
- [Core Data](modules/core_data.md) - The Core class and data handling *(coming soon)*
- [STRF Analysis](modules/strf.md) - Spatio-temporal receptive field analysis *(coming soon)*
- [Plotting](modules/plotting.md) - Visualization tools *(coming soon)*

### Configuration

- [Configuration](configuration.md) - User and project settings

### Development

- [Contributing](contributing.md) - How to add new modules *(coming soon)*
- [Testing](testing.md) - Running and writing tests *(coming soon)*

## Module Index

| Module | Description |
|--------|-------------|
| `pygor.classes` | Data classes (Core, STRF, FullField, etc.) |
| `pygor.preproc` | Preprocessing (ScanM loading, detrending) |
| `pygor.strf` | STRF analysis and plotting |
| `pygor.plotting` | General plotting utilities |
| `pygor.timeseries` | Time series analysis |
| `pygor.config` | Configuration management |
