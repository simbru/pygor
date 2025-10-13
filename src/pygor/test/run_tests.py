#!/usr/bin/env python3
"""
Comprehensive test runner for Pygor test suite
Handles import path setup and provides detailed test reporting
"""

import sys
import pathlib
import unittest
import os
import warnings

# Add src to path for imports
project_root = pathlib.Path(__file__).parents[3]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def run_tests_with_discovery():
    """Run tests using unittest discovery with proper path setup"""
    
    # Change to test directory
    test_dir = pathlib.Path(__file__).parent
    os.chdir(test_dir)
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests
    test_suite = loader.discover('.', pattern='test_*.py')
    
    # Create test runner with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    
    # Run tests
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL RESULT: {'PASS' if success else 'FAIL'}")
    
    return success

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        missing_deps.append("numpy")
        print("✗ numpy")
    
    try:
        import h5py
        print("✓ h5py")
    except ImportError:
        missing_deps.append("h5py")
        print("✗ h5py")
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError:
        missing_deps.append("matplotlib")
        print("✗ matplotlib")
    
    try:
        import pygor.load
        print("✓ pygor")
    except ImportError as e:
        print(f"✗ pygor: {e}")
        print("  Note: This may be expected if pygor is not installed in development mode")
        return False
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running tests")
        return False
    
    print("All dependencies available!")
    return True

def main():
    """Main test runner function"""
    print(f"Pygor Test Suite")
    print(f"================")
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Test directory: {pathlib.Path(__file__).parent}")
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\nSkipping tests due to missing dependencies")
        return 1
    
    # Set up environment
    os.environ['PYTHONPATH'] = str(src_path)
    
    # Filter out expected warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning, module="pygor")
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print(f"\nRunning tests...\n")
    
    # Run the tests
    success = run_tests_with_discovery()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)