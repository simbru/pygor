"""
Parameter Management Playground

Test the unified loading API and config system:
1. Auto-detect format with Core(path)
2. Explicit loaders: Core.from_h5(), Core.from_scanm()
3. Config parameter support
4. Preprocessing options
"""

import pathlib
from pygor.classes.core_data import Core

# Path to example data - update these to your files
EXAMPLE_SMP = pathlib.Path(r"D:\Igor analyses\OSDS\251105 OSDS\1_0_SWN_200_White.smp")
EXAMPLE_H5 = pathlib.Path(r"D:\Igor analyses\OSDS\251105 OSDS\1_0_SWN_200_White.h5")  # Update if different

# Path to a custom config (create one or use the example)
CUSTOM_CONFIG = r"configs\example.toml"


def test_unified_api_scanm():
    """Test 1: Unified API with ScanM file (auto-detect)."""
    print("\n" + "="*60)
    print("TEST 1: Core(smp_path) - Auto-detect ScanM")
    print("="*60)

    # New unified API - auto-detects ScanM from extension
    data = Core(EXAMPLE_SMP)

    print(f"Loaded via: Core(path) with auto-detection")
    print(f"Config source: {data.params._config_source}")
    print(f"Artifact width: {data.params.artifact_width}")
    print(f"Preprocessed: {data.params.preprocessed}")

    return data


def test_unified_api_with_config():
    """Test 2: Unified API with config."""
    print("\n" + "="*60)
    print("TEST 2: Core(smp_path, config=...)")
    print("="*60)

    data = Core(EXAMPLE_SMP, config=CUSTOM_CONFIG)

    print(f"Config source: {data.params._config_source}")
    print(f"Artifact width: {data.params.artifact_width}")

    return data


def test_unified_api_with_preprocess():
    """Test 3: Unified API with preprocessing."""
    print("\n" + "="*60)
    print("TEST 3: Core(smp_path, do_preprocess=True)")
    print("="*60)

    data = Core(EXAMPLE_SMP, do_preprocess=True)

    print(f"Preprocessed: {data.params.preprocessed}")
    if data.params.preprocessing:
        print(f"Preprocessing params: {data.params.preprocessing}")

    return data


def test_explicit_from_scanm():
    """Test 4: Explicit from_scanm (original API still works)."""
    print("\n" + "="*60)
    print("TEST 4: Core.from_scanm() - Explicit loader")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP, config=CUSTOM_CONFIG)

    print(f"Config source: {data.params._config_source}")
    print(f"Artifact width: {data.params.artifact_width}")

    return data


def test_package_defaults():
    """Test 5: Load with package defaults only (backward compat)."""
    print("\n" + "="*60)
    print("TEST 5: Package defaults only")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)

    print(f"Config source: {data.params._config_source}")
    print(f"Artifact width: {data.params.artifact_width}")
    print(f"Preprocessed: {data.params.preprocessed}")

    # Show available defaults
    print("\nAvailable default sections:")
    for section in data.params._defaults.keys():
        print(f"  - {section}")

    return data


def test_custom_config_at_load():
    """Test 6: Load with custom config file."""
    print("\n" + "="*60)
    print("TEST 6: Custom config at load time")
    print("="*60)

    print(f"Using config: {CUSTOM_CONFIG}")
    data = Core.from_scanm(EXAMPLE_SMP, config=CUSTOM_CONFIG)

    print(f"Config source: {data.params._config_source}")
    print(f"Artifact width: {data.params.artifact_width}")

    return data


def test_load_config_after():
    """Test 3: Load config after loading data."""
    print("\n" + "="*60)
    print("TEST 3: Load config after the fact")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)
    print(f"Before: config_source = {data.params._config_source}")
    print(f"Before: artifact_width = {data.params.artifact_width}")

    data.params.load_config(CUSTOM_CONFIG)
    print(f"After:  config_source = {data.params._config_source}")
    print(f"After:  artifact_width = {data.params.artifact_width}")

    return data


def test_direct_modification():
    """Test 4: Modify params directly."""
    print("\n" + "="*60)
    print("TEST 4: Direct param modification")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)
    print(f"Initial artifact_width: {data.params.artifact_width}")

    data.params.artifact_width = 10
    print(f"After setting to 10: {data.params.artifact_width}")

    return data


def test_preprocess_with_params():
    """Test 5: Preprocess uses params.artifact_width."""
    print("\n" + "="*60)
    print("TEST 5: Preprocess respects params.artifact_width")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)

    # Set artifact_width before preprocessing
    data.params.artifact_width = 5
    print(f"Set artifact_width to: {data.params.artifact_width}")

    # Preprocess without specifying artifact_width - should use params value
    data.preprocess(detrend=False)

    print(f"After preprocess:")
    print(f"  params.preprocessed: {data.params.preprocessed}")
    print(f"  params.artifact_width: {data.params.artifact_width}")
    print(f"  preprocessing dict: {data.params.preprocessing}")

    return data


def test_register_uses_artifact_width():
    """Test 6: Registration uses artifact_width from params."""
    print("\n" + "="*60)
    print("TEST 6: Registration uses params.artifact_width")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)
    data.params.artifact_width = 10
    data.preprocess(detrend=False)

    print(f"Artifact width for registration: {data.params.artifact_width}")
    print("Running registration (this excludes artifact region from shift computation)...")

    data.register(
        n_reference_frames=500,
        batch_size=50,
        upsample_factor=2,
        plot=False
    )

    print(f"\nAfter registration:")
    print(f"  params.registered: {data.params.registered}")
    if data.params.registration:
        print(f"  mean_shift: {data.params.registration.get('mean_shift')}")
        print(f"  mean_error: {data.params.registration.get('mean_error')}")

    return data


def test_full_repr():
    """Test 7: Show full params repr."""
    print("\n" + "="*60)
    print("TEST 7: Full params display")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)
    data.params.artifact_width = 10
    data.preprocess(detrend=False)
    data.register(n_reference_frames=500, batch_size=50, upsample_factor=2, plot=False)

    print(data.params)

    return data


def test_summary_view():
    """Test 8: Tree-style summary view for quick browsing."""
    print("\n" + "="*60)
    print("TEST 8: Tree-style summary() view")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)

    print("\n--- Compact view (defaults collapsed) ---")
    print(data.params.summary())

    print("\n--- Full view (all params shown) ---")
    print(data.params.summary(show_all=True))

    return data


def test_summary_after_processing():
    """Test 9: Summary after preprocessing shows modifications."""
    print("\n" + "="*60)
    print("TEST 9: Summary after processing (shows → changes)")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)
    data.params.artifact_width = 5  # Modify from default
    data.preprocess(detrend=False)

    print(data.params.summary(show_all=True))

    return data


def test_html_output():
    """Test 10: Generate HTML output (for Jupyter)."""
    print("\n" + "="*60)
    print("TEST 10: HTML output (Jupyter _repr_html_)")
    print("="*60)

    data = Core.from_scanm(EXAMPLE_SMP)
    data.preprocess(detrend=False)

    html = data.params._repr_html_()
    print(f"HTML output length: {len(html)} characters")
    print("\nFirst 500 chars of HTML:")
    print(html[:500])
    print("...")

    # Save to file for inspection
    output_path = pathlib.Path(__file__).parent / "params_preview.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Params Preview</title></head><body>{html}</body></html>')
    print(f"\nFull HTML saved to: {output_path}")

    return data


def main():
    """Run all tests."""
    print("Parameter Management Playground")
    print("================================")
    print(f"ScanM file: {EXAMPLE_SMP}")
    print(f"Custom config: {CUSTOM_CONFIG}")

    if not EXAMPLE_SMP.exists():
        print(f"\nERROR: Data file not found: {EXAMPLE_SMP}")
        print("Update EXAMPLE_SMP at the top of this file.")
        return

    # === NEW UNIFIED API TESTS ===
    test_unified_api_scanm()          # Core(smp_path) auto-detection
    test_unified_api_with_config()    # Core(smp_path, config=...)
    test_unified_api_with_preprocess()  # Core(smp_path, do_preprocess=True)
    test_explicit_from_scanm()        # Core.from_scanm() still works

    # === ORIGINAL TESTS (backward compat) ===
    test_package_defaults()
    test_custom_config_at_load()
    test_load_config_after()
    test_direct_modification()
    test_preprocess_with_params()
    # test_register_uses_artifact_width()  # Slower - uncomment to test
    # test_full_repr()  # Shows everything - uncomment to test

    # === SUMMARY/BROWSING TESTS ===
    test_summary_view()  # Tree-style quick browsing
    # test_summary_after_processing()  # Shows modifications with arrows
    test_html_output()  # Generate HTML for Jupyter preview

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
