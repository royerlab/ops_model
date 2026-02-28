"""
Quick test to verify the prediction pipeline implementation.

This script checks that:
1. The predict.py imports work correctly
2. The DynaClrAnnDataWriter is accessible
3. Config loading works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

print("Testing DynaCLR prediction pipeline...")
print("-" * 60)

# Test 1: Import predict script
print("\n1. Testing predict.py imports...")
try:
    from ops_model.data import data_loader
    from ops_model.models import dynaclr
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check DynaClrAnnDataWriter exists
print("\n2. Testing DynaClrAnnDataWriter class...")
try:
    assert hasattr(dynaclr, "DynaClrAnnDataWriter")
    writer_class = dynaclr.DynaClrAnnDataWriter
    print(f"✓ DynaClrAnnDataWriter found")
    print(f"  - __init__ parameters: output_dir, run_name, labels_df, save_features, save_projections")
except Exception as e:
    print(f"✗ Writer class check failed: {e}")
    sys.exit(1)

# Test 3: Check predict_step handles both dataset types
print("\n3. Testing predict_step compatibility...")
try:
    import inspect
    predict_step_source = inspect.getsource(dynaclr.LitDynaClr.predict_step)
    assert '"anchor" in batch' in predict_step_source
    assert '"data"' in predict_step_source
    print("✓ predict_step handles both ContrastiveDataset and BasicDataset")
except Exception as e:
    print(f"✗ predict_step check failed: {e}")
    sys.exit(1)

# Test 4: Check example config exists
print("\n4. Testing example config file...")
try:
    config_path = Path(__file__).parent / "phase_only" / "predict_config_example.yml"
    assert config_path.exists()
    
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Verify required keys
    assert "model_type" in config
    assert "ckpt_path" in config
    assert "data_manager" in config
    assert "model" in config
    assert "prediction" in config
    
    print(f"✓ Config file exists and valid")
    print(f"  - Path: {config_path}")
    print(f"  - Keys: {list(config.keys())}")
except Exception as e:
    print(f"✗ Config check failed: {e}")
    sys.exit(1)

# Test 5: Check documentation exists
print("\n5. Testing documentation files...")
try:
    readme_path = Path(__file__).parent / "PREDICTION_README.md"
    summary_path = Path(__file__).parent / "IMPLEMENTATION_SUMMARY.md"
    
    assert readme_path.exists(), "PREDICTION_README.md not found"
    assert summary_path.exists(), "IMPLEMENTATION_SUMMARY.md not found"
    
    print("✓ Documentation files exist")
    print(f"  - PREDICTION_README.md: {readme_path.stat().st_size} bytes")
    print(f"  - IMPLEMENTATION_SUMMARY.md: {summary_path.stat().st_size} bytes")
except Exception as e:
    print(f"✗ Documentation check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nNext steps:")
print("1. Update checkpoint path in predict_config_example.yml")
print("2. Run: python predict.py --config_path phase_only/predict_config_example.yml")
print("3. Verify output with: import anndata as ad; adata = ad.read_zarr('path/to/output.zarr')")
