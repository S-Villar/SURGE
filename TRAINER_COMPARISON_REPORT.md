# SurrogateTrainer vs MLTrainer Comparison Report

## Executive Summary

**Recommendation:** Merge into a single unified trainer class. `SurrogateTrainer` is a thin wrapper that adds minimal functionality and creates confusion.

---

## Current Architecture

### SurrogateTrainer (Thin Wrapper)
**Lines:** 66-171 (106 lines)
**Purpose:** High-level convenience interface

**Features:**
- ✅ Auto-detects input/output variables from pickle files
- ✅ Uses hardcoded `PwE_` prefix detection for outputs
- ✅ Performs dataset structure analysis (uses `preprocessing` module)
- ✅ Creates an `MLTrainer` instance internally
- ❌ **No direct access to training/prediction** - requires `get_trainer()` call
- ❌ **Limited to pickle files only**

**Methods:**
- `__init__(dir_path=None)`
- `load_dataset_pickle(pickle_path)` → Returns `(input_vars, output_vars)`
- `get_trainer()` → Returns underlying `MLTrainer` instance

**Usage Pattern:**
```python
surge_trainer = SurrogateTrainer()
input_cols, output_cols = surge_trainer.load_dataset_pickle('data.pkl')
trainer = surge_trainer.get_trainer()  # Need to get underlying trainer
trainer.load_df_dataset(surge_trainer.df, input_cols, output_cols)
trainer.train_test_split()
trainer.standardize_data()
# ... rest of workflow
```

---

### MLTrainer (Full-Featured Core)
**Lines:** 174-2451 (2277 lines)
**Purpose:** Complete ML training pipeline

**Features:**
- ✅ Load data from DataFrame (`load_df_dataset`)
- ✅ Data splitting (`train_test_split`)
- ✅ Data standardization (`standardize_data`)
- ✅ Model initialization (`init_model`) - supports registry keys
- ✅ Model training (`train`)
- ✅ Model prediction (`predict_output`)
- ✅ Cross-validation (`cross_validate`)
- ✅ Visualization (`plot_regression_results`, `plot_all_outputs`, `plot_profiles`)
- ✅ Model comparison (`compare_models`, `compare_all_models`)
- ✅ Save/load results (`save_results`)
- ✅ Model registry integration
- ✅ Multiple model support
- ❌ **No auto-detection of inputs/outputs**
- ❌ **No pickle file loading with auto-analysis**

**Methods (Major):**
- `load_df_dataset(df, input_feature_names, output_feature_names)`
- `train_test_split(test_split=0.2, random_state=None)`
- `standardize_data()`
- `init_model(model_type, **kwargs)`
- `train(model_index)`
- `predict_output(model_index)`
- `cross_validate(model_idx, cv_folds=5)`
- `plot_regression_results(...)`
- `save_results(...)`
- ... (30+ methods total)

**Usage Pattern:**
```python
trainer = MLTrainer(n_features=None, n_outputs=None)
trainer.load_df_dataset(df, input_cols, output_cols)  # Manual specification
trainer.train_test_split()
trainer.standardize_data()
trainer.init_model('random_forest')
trainer.train(0)
trainer.predict_output(0)
```

---

## Problem Analysis

### Issues with Current Design

1. **Duplication of Intent**
   - Both classes are "trainers" for surrogate models
   - `SurrogateTrainer` is just a thin convenience layer
   - Users must choose between two APIs for the same task

2. **Awkward API**
   - `SurrogateTrainer` requires `get_trainer()` to access core functionality
   - Forces users to work with two objects (`surge_trainer` and `trainer`)
   - Violates single responsibility principle

3. **Limited Functionality**
   - `SurrogateTrainer` only supports pickle files
   - Hardcoded `PwE_` prefix detection (not generalizable)
   - No direct access to training/prediction methods

4. **Inconsistency**
   - Example scripts use both classes differently
   - `test_hhfw_simple.py` uses `SurrogateTrainer` then `get_trainer()`
   - `test_workflow.py` uses `MLTrainer` directly
   - No clear guidance on which to use

5. **Maintenance Overhead**
   - Two classes to maintain
   - Changes need to be synchronized
   - Documentation must cover both

---

## Proposed Solution

### Option 1: Merge into Unified Class (Recommended)

**New Name:** `SurrogateEngine` (aligns with SURGE project name)

**Add to unified class:**
- `load_dataset_pickle(pickle_path)` method
- `load_dataset_from_file(file_path, **kwargs)` method (CSV, parquet, pickle, etc.)
- `auto_detect_inputs_outputs(df, output_prefix=None)` method
- Keep all existing `MLTrainer` functionality

**Remove:**
- Current `SurrogateTrainer` class
- `get_trainer()` pattern

**Benefits:**
- ✅ Single unified API
- ✅ All functionality in one place
- ✅ No confusion about which class to use
- ✅ Better naming ("Surrogate" is more descriptive than "ML")
- ✅ Flexible: can use auto-detection or manual specification

**Example Usage:**
```python
# Option A: Auto-detection (current SurrogateTrainer style)
engine = SurrogateEngine()
engine.load_dataset_pickle('data.pkl')  # Auto-detects inputs/outputs
engine.train_test_split()
engine.standardize_data()
engine.init_model('random_forest')
engine.train(0)

# Option B: Manual specification (current MLTrainer style)
engine = SurrogateEngine()
engine.load_df_dataset(df, input_cols, output_cols)
engine.train_test_split()
# ... rest same

# Option C: From other file formats
engine = SurrogateEngine()
engine.load_dataset_from_file('data.csv', output_prefix='PwE_')
# ... rest same
```

---

### Option 2: Keep MLTrainer, Enhance It

**Keep:** Current `MLTrainer` as base class

**Add:**
- `load_dataset_pickle(pickle_path, auto_detect=True)` method
- `auto_detect_inputs_outputs(df, output_prefix='PwE_')` method

**Deprecate:** `SurrogateTrainer` (mark as deprecated, remove in future)

**Benefits:**
- ✅ Less disruptive (keep existing name)
- ✅ Backward compatible
- ⚠️ Still have two classes for transition period

---

## Recommended Implementation (Option 1)

### New Unified Class Structure

```python
class SurrogateEngine:
    """
    Unified SURGE surrogate trainer for comprehensive modeling workflows.
    
    Supports both auto-detection and manual specification of inputs/outputs.
    """
    
    def __init__(self, n_features=None, n_outputs=None, dir_path=None):
        """Initialize trainer."""
        # ... existing MLTrainer init ...
    
    # === Data Loading Methods ===
    
    def load_dataset_pickle(self, pickle_path, auto_detect_outputs=True, output_prefix='PwE_'):
        """
        Load dataset from pickle file with optional auto-detection.
        
        Parameters:
        -----------
        pickle_path : str or Path
            Path to pickle file
        auto_detect_outputs : bool, default=True
            If True, automatically detect outputs by prefix
        output_prefix : str, default='PwE_'
            Prefix to use for output detection (if auto_detect_outputs=True)
            
        Returns:
        --------
        tuple : (input_vars, output_vars) if auto_detect, else None
        """
        # Load pickle
        # If auto_detect: use detection logic
        # Otherwise: just load and expect manual specification
        # Call load_df_dataset() internally
    
    def load_df_dataset(self, df, input_feature_names=None, output_feature_names=None):
        """
        Load dataset from DataFrame.
        
        If input_feature_names or output_feature_names are None,
        will attempt auto-detection.
        """
        # ... existing logic ...
    
    def load_dataset_from_file(self, file_path, input_cols=None, output_cols=None, 
                                auto_detect_outputs=False, output_prefix='PwE_', **kwargs):
        """
        Load dataset from file (CSV, parquet, pickle, Excel).
        
        Supports auto-detection if input_cols or output_cols not provided.
        """
        # Load based on extension
        # Auto-detect if needed
        # Call load_df_dataset()
    
    def auto_detect_inputs_outputs(self, df, output_prefix='PwE_'):
        """
        Automatically detect input and output columns.
        
        Returns:
        --------
        tuple : (input_columns, output_columns)
        """
        # Current SurrogateTrainer detection logic
    
    # === All existing MLTrainer methods ===
    # train_test_split, standardize_data, init_model, train, predict_output, etc.
```

---

## Migration Plan

### Step 1: Add Methods to MLTrainer
- Add `load_dataset_pickle()` method
- Add `auto_detect_inputs_outputs()` method
- Test compatibility

### Step 2: Rename MLTrainer → SurrogateEngine
- Update class name
- Update all imports
- Update all references in codebase

### Step 3: Remove Old SurrogateTrainer
- Delete old wrapper class
- Update documentation
- Update examples

### Step 4: Update Tests and Examples
- Update all test scripts
- Update notebooks
- Update documentation

---

## Code Impact Analysis

### Files Using SurrogateTrainer:
- `test_hhfw_simple.py`
- `test_hhfw_dataset.py`
- `surge/__init__.py` (exports)
- `docs/` files
- `examples/` potentially

### Files Using MLTrainer:
- `tests/test_workflow.py`
- `tests/test_model_comparison.py`
- `surge/helpers.py`
- `examples/quick_start_demo.py`
- Many test files

### Estimated Changes:
- ~15-20 files to update
- Mostly simple find/replace
- Tests need to be updated to use unified API

---

## Recommendation

**✅ Implement Option 1: Unified SurrogateEngine**

**Rationale:**
1. Simpler API - one class, one way to do things
2. Better naming - "Surrogate" is domain-specific and clear
3. More flexible - supports both auto-detection and manual specification
4. Less maintenance - single class to maintain
5. Cleaner architecture - no wrapper pattern needed

**Timeline:** Can be done incrementally to maintain backward compatibility.

