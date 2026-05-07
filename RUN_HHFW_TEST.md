# Running the HHFW Dataset Test

## Issue
The base conda environment has NumPy 2.3.4 which is incompatible with TensorFlow/gpflow 
(compiled against NumPy 1.x). 

## Solution Options

### Option 1: Create and use the conda environment (Recommended)
```bash
# Create the environment from environment.yml (has numpy<3.12 compatible versions)
conda env create -f environment.yml

# Activate it
conda activate surge

# Run the test
python test_hhfw_simple.py
```

### Option 2: Fix NumPy version in current environment
```bash
# Downgrade NumPy to <2.0
pip install "numpy<2.0"

# Then run
python test_hhfw_simple.py
```

### Option 3: Use a Python environment with compatible packages
Use a conda environment or venv with:
- numpy<2.0
- scikit-learn
- pandas
- matplotlib
- joblib

Then run:
```bash
python test_hhfw_simple.py
```

The test script will:
1. Load PwE_.pkl dataset
2. Train Random Forest model
3. Run cross-validation
4. Evaluate on test set
5. Save results and visualizations to test_outputs/
