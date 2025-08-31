# CLAUDE.md

Essential architectural patterns for Claude Code when working with pygor.

## Core Architecture

**Import System**: Classes auto-discovered from `classes/` directory
```python
from pygor.load import STRF, MovingBars, Experiment  # etc.
```

**Directory Structure**: 
- `classes/`: Data class definitions 
- `strf/`, `timeseries/`: Analysis modules
- Complex methods live in submodules, class methods are simple wrappers

## Critical Data Conventions

**Array Axes**: Always `[cell, time, y, x]` (indices 0, 1, 2, 3)

**Multicolor Data**: Multiple STRFs per ROI (one per color channel)
- **ROI-centric methods** (need reshaping): `get_time_amps()`, `spatial_overlap_index_mean()`, `collapse_times()`
- **Already multicolor**: `get_polarities()`, `collapse_times_chroma()`, methods using `cs_seg()`

**Auto-Generated `_by_channel` Methods**: All ROI-centric methods automatically get multicolor variants:
```python
obj.get_time_amps_by_channel()        # Instead of manual reshaping
obj.spatial_overlap_index_mean_by_channel()
# etc. - see _NEEDS_MULTICOLOR_RESHAPING list in strf_data.py
```

## Performance Pattern: Method Caching

Standard template for expensive methods:
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

Applied to: `collapse_times()`, `spatial_overlap_index_stats()`, and should be applied to other expensive methods.

## Development Guidelines

- **Always prefer editing existing files** over creating new ones
- **Use vectorized numpy operations** for performance
- **Memory usage can be significant** (32GB+ common for STRF data)
- **Follow existing patterns** in the codebase for consistency