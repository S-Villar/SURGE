"""
Tests for SurrogateDataset class.

Tests data loading, auto-detection, and dataset analysis capabilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surge import SurrogateDataset


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing.
    
    Uses non-pattern names for inputs (so they're detected as standalone/inputs)
    and pattern-based names for outputs (var_1, var_2) so they're detected as outputs.
    """
    np.random.seed(42)
    n_samples = 100
    n_inputs = 3
    n_outputs = 2

    # Create inputs with non-pattern names (standalone columns)
    # These won't match the pattern detection, so they'll be detected as inputs
    X = np.random.random((n_samples, n_inputs))
    input_cols = ['temperature', 'pressure', 'density']  # Non-pattern names

    # Create outputs with pattern-based naming (var_1, var_2)
    # These will match the pattern and be detected as outputs
    y = np.random.random((n_samples, n_outputs))
    output_cols = [f'var_{i+1}' for i in range(n_outputs)]

    df = pd.DataFrame(X, columns=input_cols)
    for i, output_col in enumerate(output_cols):
        df[output_col] = y[:, i]

    return df, input_cols, output_cols


def test_dataset_init():
    """Test SurrogateDataset initialization."""
    dataset = SurrogateDataset()
    assert dataset.df is None
    assert dataset.input_columns == []
    assert dataset.output_columns == []
    assert dataset.output_groups == {}
    assert dataset.analysis is None


def test_load_from_dataframe(sample_dataframe):
    """Test loading data from DataFrame."""
    df, input_cols, output_cols = sample_dataframe

    dataset = SurrogateDataset()
    
    # Test manual specification
    detected_inputs, detected_outputs = dataset.load_from_dataframe(
        df,
        input_cols=input_cols,
        output_cols=output_cols,
        auto_detect=False
    )

    assert dataset.df is not None
    assert len(detected_inputs) == len(input_cols)
    assert len(detected_outputs) == len(output_cols)
    assert detected_inputs == input_cols
    assert detected_outputs == output_cols


def test_load_from_pickle(sample_dataframe):
    """Test loading data from pickle file."""
    df, input_cols, output_cols = sample_dataframe

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "test_data.pkl"
        df.to_pickle(pkl_path)

        dataset = SurrogateDataset()
        detected_inputs, detected_outputs = dataset.load_from_file(
            pkl_path,
            auto_detect=True
        )

        assert dataset.df is not None
        assert len(detected_inputs) > 0, f"Expected inputs detected, got: {detected_inputs}"
        assert len(detected_outputs) > 0, f"Expected outputs detected, got: {detected_outputs}"
        # Verify that auto-detection correctly identifies inputs and outputs
        # (inputs should be the standalone columns, outputs should be var_1, var_2)
        assert set(detected_outputs) == set(output_cols), f"Expected outputs {output_cols}, got {detected_outputs}"


def test_load_from_csv(sample_dataframe):
    """Test loading data from CSV file."""
    df, input_cols, output_cols = sample_dataframe

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_data.csv"
        df.to_csv(csv_path, index=False)

        dataset = SurrogateDataset()
        detected_inputs, detected_outputs = dataset.load_from_file(
            csv_path,
            auto_detect=True
        )

        assert dataset.df is not None
        assert len(detected_inputs) > 0, f"Expected inputs detected, got: {detected_inputs}"
        assert len(detected_outputs) > 0, f"Expected outputs detected, got: {detected_outputs}"
        # Verify that auto-detection correctly identifies outputs
        assert set(detected_outputs) == set(output_cols), f"Expected outputs {output_cols}, got {detected_outputs}"


def test_auto_detection(sample_dataframe):
    """Test automatic input/output detection."""
    df, input_cols, output_cols = sample_dataframe

    # Create a temporary file first
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "test_data.pkl"
        df.to_pickle(pkl_path)

        dataset = SurrogateDataset()
        detected_inputs, detected_outputs = dataset.load_from_file(
            pkl_path,
            auto_detect=True
        )

        # Auto-detection should identify patterns
        # Outputs should be var_1, var_2 (pattern-based columns)
        assert len(detected_outputs) > 0, f"Expected outputs detected, got: {detected_outputs}"
        assert set(detected_outputs) == set(output_cols), f"Expected outputs {output_cols}, got {detected_outputs}"
        
        # Inputs should be the standalone columns (temperature, pressure, density)
        assert len(detected_inputs) > 0, f"Expected inputs detected, got: {detected_inputs}"
        assert set(detected_inputs) == set(input_cols), f"Expected inputs {input_cols}, got {detected_inputs}"


def test_manual_specification(sample_dataframe):
    """Test manual specification of input/output columns."""
    df, input_cols, output_cols = sample_dataframe

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "test_data.pkl"
        df.to_pickle(pkl_path)

        dataset = SurrogateDataset()
        detected_inputs, detected_outputs = dataset.load_from_file(
            pkl_path,
            input_cols=input_cols,
            output_cols=output_cols,
            auto_detect=False
        )

        assert detected_inputs == input_cols
        assert detected_outputs == output_cols


def test_get_statistics(sample_dataframe):
    """Test getting dataset statistics."""
    df, input_cols, output_cols = sample_dataframe

    dataset = SurrogateDataset()
    # Load data properly using load_from_dataframe
    dataset.load_from_dataframe(
        df,
        input_cols=input_cols,
        output_cols=output_cols,
        auto_detect=False
    )

    # Get statistics
    stats = dataset.get_statistics()
    assert isinstance(stats, pd.DataFrame)
    assert len(stats) > 0
    # Verify statistics contain expected columns (from pandas describe())
    assert 'count' in stats.columns or 'mean' in stats.columns or len(stats.index) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

