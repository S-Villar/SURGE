#!/usr/bin/env python3
"""
Demo script to showcase M3DC1 HDF5 loading capabilities.

This script demonstrates SURGE's ability to:
1. Load M3DC1 HDF5 data
2. Convert to DataFrame
3. Auto-detect inputs/outputs
4. Validate data
5. Convert to PyTorch Dataset
6. Integrate with SurrogateEngine

If sdata.h5 doesn't exist, creates a mock dataset for demonstration.
"""

import sys
from pathlib import Path

# Add project root to path (go up from scripts/m3dc1/ to SURGE root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

print("=" * 70)
print("🚀 SURGE M3DC1 Dataset Loading Capabilities Demo")
print("=" * 70)

# Try to import SURGE modules
try:
    from surge import (
        SurrogateDataset,
        SurrogateEngine,
        read_m3dc1_hdf5_structure,
        convert_to_dataframe,
        validate_dataset,
        get_dataset_summary,
        print_validation_report,
        dataframe_to_pytorch_dataset,
        M3DC1_LOADER_AVAILABLE,
        DATASET_UTILS_AVAILABLE,
    )
    print("✅ SURGE modules imported successfully")
    print(f"   M3DC1 loader available: {M3DC1_LOADER_AVAILABLE}")
    print(f"   Dataset utils available: {DATASET_UTILS_AVAILABLE}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Please ensure you're in the surge conda environment:")
    print("   module load conda && conda activate surge")
    sys.exit(1)

print("\n" + "=" * 70)
print("Step 1: Check for M3DC1 HDF5 file")
print("=" * 70)

hdf5_file = Path('sdata.h5')
if hdf5_file.exists():
    print(f"✅ Found M3DC1 HDF5 file: {hdf5_file}")
    use_real_data = True
else:
    print(f"⚠️  M3DC1 HDF5 file not found: {hdf5_file}")
    print("   Creating mock dataset for demonstration...")
    use_real_data = False
    
    # Create a mock DataFrame that mimics M3DC1 structure
    np.random.seed(42)
    n_samples = 100
    
    # Mock input features (similar to M3DC1 inputs)
    mock_data = {
        'run_id': [f'run_{i}' for i in range(n_samples)],
        'input_ntor': np.random.randint(1, 5, n_samples),
        'input_pscale': np.random.uniform(0.5, 2.0, n_samples),
        'input_batemanscale': np.random.uniform(0.8, 1.2, n_samples),
        'input_runid': np.arange(1, n_samples + 1),
        'input_eqid': np.random.randint(1300, 1400, n_samples),
        'eq_a': np.random.uniform(0.3, 0.5, n_samples),
        'eq_R0': np.random.uniform(0.8, 1.2, n_samples),
        'eq_kappa': np.random.uniform(1.5, 2.0, n_samples),
        'eq_delta': np.random.uniform(0.3, 0.6, n_samples),
    }
    
    # Mock output features (gamma and eignemode flattened)
    mock_data['output_gamma'] = (
        0.1 * mock_data['input_pscale'] + 
        0.05 * mock_data['eq_kappa'] + 
        np.random.normal(0, 0.02, n_samples)
    )
    
    # Add flattened eignemode (10 values per sample)
    for i in range(10):
        mock_data[f'output_eignemode_{i}'] = (
            mock_data['output_gamma'] * (1 + i * 0.1) +
            np.random.normal(0, 0.01, n_samples)
        )
    
    df = pd.DataFrame(mock_data)
    print(f"   ✅ Created mock dataset: {df.shape[0]} samples, {df.shape[1]} columns")

print("\n" + "=" * 70)
print("Step 2: Explore HDF5 Structure (if real file exists)")
print("=" * 70)

if use_real_data:
    try:
        structure = read_m3dc1_hdf5_structure(hdf5_file)
        print("✅ Structure exploration complete")
    except Exception as e:
        print(f"⚠️  Error exploring structure: {e}")
        use_real_data = False
        print("   Falling back to mock data...")
else:
    print("⚠️  Skipping - using mock data")

print("\n" + "=" * 70)
print("Step 3: Convert to DataFrame")
print("=" * 70)

if use_real_data:
    try:
        df = convert_to_dataframe(
            hdf5_file,
            include_equilibrium_features=True,
            include_mesh_features=False,
        )
        print(f"✅ Converted HDF5 to DataFrame: {df.shape[0]} samples, {df.shape[1]} columns")
    except Exception as e:
        print(f"⚠️  Error converting HDF5: {e}")
        print("   Using mock data instead...")
        use_real_data = False

if not use_real_data:
    print(f"✅ Using mock DataFrame: {df.shape[0]} samples, {df.shape[1]} columns")

print(f"\n📋 First few columns: {list(df.columns)[:10]}")
print(f"\n📊 First 3 rows:")
print(df.head(3).to_string())

print("\n" + "=" * 70)
print("Step 4: Load with SurrogateDataset and Auto-Detect")
print("=" * 70)

dataset = SurrogateDataset()

if use_real_data:
    try:
        input_cols, output_cols = dataset.load_from_file(
            hdf5_file,
            m3dc1_format=True,
            auto_detect=True,
        )
    except Exception as e:
        print(f"⚠️  Error loading from HDF5: {e}")
        print("   Loading from DataFrame instead...")
        dataset.load_from_dataframe(df, auto_detect=True)
        input_cols, output_cols = dataset.input_columns, dataset.output_columns
else:
    dataset.load_from_dataframe(df, auto_detect=True)
    input_cols, output_cols = dataset.input_columns, dataset.output_columns

# Filter out non-numeric columns (like run_id) from inputs/outputs
numeric_inputs = [col for col in input_cols if col in dataset.df.columns and 
                  pd.api.types.is_numeric_dtype(dataset.df[col])]
numeric_outputs = [col for col in output_cols if col in dataset.df.columns and 
                   pd.api.types.is_numeric_dtype(dataset.df[col])]

print(f"\n✅ Auto-detection complete:")
print(f"   📥 Input columns: {len(input_cols)} (numeric: {len(numeric_inputs)})")
print(f"   📤 Output columns: {len(output_cols)} (numeric: {len(numeric_outputs)})")
print(f"\n   Sample inputs: {numeric_inputs[:5]}")
print(f"   Sample outputs: {numeric_outputs[:5]}")

# Update to use only numeric columns
input_cols = numeric_inputs
output_cols = numeric_outputs

print("\n" + "=" * 70)
print("Step 5: Validate Dataset")
print("=" * 70)

validation_results = validate_dataset(dataset.df)
print_validation_report(validation_results)

summary = get_dataset_summary(dataset.df)
print(f"\n📊 Dataset Summary:")
print(f"   Shape: {summary['shape']}")
print(f"   Memory usage: {summary['memory_usage_mb']:.2f} MB")
print(f"   Numeric columns: {len(summary['numeric_columns'])}")
print(f"   Categorical columns: {len(summary['categorical_columns'])}")

print("\n" + "=" * 70)
print("Step 6: Convert to PyTorch Dataset")
print("=" * 70)

try:
    pytorch_dataset = dataframe_to_pytorch_dataset(
        dataset.df,
        input_columns=input_cols,
        output_columns=output_cols,
        device='cpu',
    )
    
    print(f"✅ Converted to PyTorch TensorDataset")
    print(f"   Dataset size: {len(pytorch_dataset)}")
    print(f"   Input shape: {pytorch_dataset[0][0].shape}")
    print(f"   Output shape: {pytorch_dataset[0][1].shape}")
    
    X_sample, y_sample = pytorch_dataset[0]
    print(f"\n   First sample:")
    print(f"   X (first 5): {X_sample[:5].numpy()}")
    print(f"   y (first 5): {y_sample[:5].numpy() if y_sample.numel() > 5 else y_sample.numpy()}")
except ImportError:
    print("⚠️  PyTorch not available - skipping PyTorch conversion")
    print("   Install with: pip install torch")
except Exception as e:
    print(f"⚠️  Error converting to PyTorch: {e}")

print("\n" + "=" * 70)
print("Step 7: Integration with SurrogateEngine")
print("=" * 70)

try:
    engine = SurrogateEngine()
    # Create a clean dataset with only numeric columns
    clean_df = dataset.df[numeric_inputs + numeric_outputs].copy()
    clean_dataset = SurrogateDataset()
    clean_dataset.load_from_dataframe(
        clean_df,
        input_cols=numeric_inputs,
        output_cols=numeric_outputs,
        auto_detect=False
    )
    engine.load_from_dataset(clean_dataset)
    
    print(f"✅ Dataset loaded into SurrogateEngine")
    print(f"   Samples: {engine.X.shape[0]}")
    print(f"   Features: {engine.F}")
    print(f"   Outputs: {engine.T}")
    
    # Split data
    engine.train_test_split(test_split=0.2, random_state=42)
    print(f"\n✅ Data split:")
    print(f"   Train+Val: {len(engine.X_train_val)} samples")
    print(f"   Test: {len(engine.X_test)} samples")
    
    # Standardize data
    engine.standardize_data()
    print(f"\n✅ Data standardized")
    
    # Initialize and train a model
    engine.init_model('random_forest', n_estimators=50, random_state=42, max_depth=10)
    print(f"\n✅ Model initialized: Random Forest")
    
    engine.train(0)
    print(f"✅ Model trained")
    
    # Make predictions
    engine.predict_output(0)
    
    # Display results
    print(f"\n📊 Model Performance:")
    print(f"   Training R²: {engine.R2_train_val:.4f}")
    print(f"   Test R²: {engine.R2:.4f}")
    print(f"   Training MSE: {engine.MSE_train_val:.6f}")
    print(f"   Test MSE: {engine.MSE:.6f}")
    
except Exception as e:
    print(f"⚠️  Error in SurrogateEngine workflow: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ Demo Complete!")
print("=" * 70)
print("\nSummary of capabilities demonstrated:")
print("  ✅ M3DC1 HDF5 file structure exploration")
print("  ✅ HDF5 to DataFrame conversion")
print("  ✅ Automatic input/output detection")
print("  ✅ Dataset validation")
print("  ✅ PyTorch Dataset conversion")
print("  ✅ SurrogateEngine integration")
print("\n" + "=" * 70)

