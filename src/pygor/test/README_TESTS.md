# Pygor Test Suite

## Overview

The Pygor test suite has been comprehensively improved to provide robust validation of the codebase functionality. The tests now properly validate outputs, catch real errors, and provide confidence in the code reliability.

## Key Improvements Made

### 1. Fixed Import Issues
- Added proper path configuration to handle module imports
- Tests can now run without requiring pygor to be installed in development mode
- Added dependency checking to provide clear error messages

### 2. Removed Over-Permissive Exception Handling
- Replaced `warnings.warn()` calls with proper test failures
- Removed overly broad try-catch blocks that masked real errors
- Added specific validation for different types of exceptions

### 3. Added Comprehensive Output Validation
- Tests now validate actual function outputs, not just that they run
- Added array shape, type, and value range validation
- Implemented proper assertions for all test cases

### 4. New STRF Method Testing
- Added comprehensive tests for `map_extrema_timing()`
- Added tests for `compute_spatial_overlap_metrics()`
- Added tests for `analyze_multicolor_spatial_alignment()`
- Validates STRF axis convention compliance

### 5. Edge Case and Error Handling Tests
- Tests with empty/zero arrays
- Tests with invalid inputs (NaN, Inf, wrong shapes)
- Boundary condition testing (extreme thresholds)
- Memory efficiency validation with larger arrays

### 6. Test Utilities and Infrastructure
- Created mock data generation functions
- Added reusable assertion helpers
- Implemented temporary file management
- Added STRF axis convention validation

## Test Files

### Core Test Files
- `test_Core.py`: Tests for base Core class functionality
- `test_STRF.py`: Tests for STRF-specific methods and analysis
- `test_analyses_import.py`: Tests for dynamic module import system

### New Test Files
- `test_edge_cases.py`: Edge cases, error handling, and boundary conditions
- `test_utilities.py`: Helper functions and mock data generation
- `run_tests.py`: Comprehensive test runner with dependency checking

## Running Tests

### Using Python Test Runner (Recommended)
```bash
# From project root
python src/pygor/test/run_tests.py
```

### Using Windows Batch File
```cmd
# From test directory
run_test.bat
```

### Using Standard unittest
```bash
# From project root
PYTHONPATH=src python -m unittest discover src/pygor/test -v
```

## Test Organization

### TestCore
- Basic data type validation
- Core data structure validation  
- Metadata and ROI property testing
- Trigger timing consistency checks

### TestSTRF
- STRF-specific method testing
- Bootstrap functionality validation
- New extrema timing analysis tests
- Spatial alignment analysis tests
- Array structure validation

### TestSTRF_plot
- Plotting method validation
- Visual output testing

### TestEdgeCases
- Empty/zero array handling
- Invalid input validation
- NaN and infinite value handling
- Memory efficiency testing
- Parameter boundary validation

## Test Data Requirements

Tests use mock data generation and the example data file:
- `examples/strf_demo_data.h5`: Required for integration testing
- Mock data generators: For unit testing without file dependencies

## Validation Criteria

### Array Structure Validation
- STRF arrays must follow [cell, time, y, x] convention
- Time dimension should be smaller than spatial dimensions
- All arrays must be finite (no NaN/Inf unless expected)

### Output Validation
- Method outputs must match expected types and shapes
- Numerical outputs must be within reasonable ranges
- Error conditions must be handled gracefully

### Performance Validation
- Methods must handle typical dataset sizes (32GB+ memory usage)
- Vectorized operations preferred for efficiency
- Memory usage should be reasonable for given input sizes

## Future Improvements

1. **Integration Testing**: Full workflow testing from data loading to analysis
2. **Performance Benchmarking**: Systematic performance testing with various data sizes
3. **Comparison Testing**: Validation against known good outputs or IGOR results
4. **Continuous Integration**: Automated test running on code changes

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed (numpy, h5py, matplotlib, etc.)
- Check that PYTHONPATH includes the src directory
- Verify example data file exists in examples/

### Test Failures
- Check that test data file is not corrupted
- Verify that recent code changes don't break existing functionality
- Review test output for specific assertion failures

### Memory Issues
- Large test arrays may require substantial RAM
- Consider reducing test data size if memory constrained
- Monitor memory usage during test execution

The improved test suite now provides robust validation and will catch real issues rather than giving false confidence through overly permissive testing.