"""Tests for XGCDataset class."""

from pathlib import Path

import numpy as np
import pytest

from surge.datasets import XGCDataset


@pytest.fixture
def npy_data(tmp_path):
    """Create small .npy data and target files."""
    np.random.seed(42)
    n_samples = 100
    n_inputs = 10
    n_outputs = 2
    data = np.random.randn(n_samples, n_inputs).astype(np.float32)
    target = np.random.randn(n_samples, n_outputs).astype(np.float32)
    data_path = tmp_path / "data.npy"
    target_path = tmp_path / "target.npy"
    np.save(data_path, data)
    np.save(target_path, target)
    return data_path, target_path, n_inputs, n_outputs


def test_xgc_dataset_from_npy(npy_data):
    """from_npy loads data and target into DataFrame."""
    data_path, target_path, n_inputs, n_outputs = npy_data
    dataset = XGCDataset.from_npy(data_path, target_path)
    assert dataset.df is not None
    assert len(dataset.input_columns) == n_inputs
    assert len(dataset.output_columns) == n_outputs
    assert dataset.input_columns[0] == "input_0"
    assert dataset.output_columns[0] == "output_0"
    assert len(dataset.df) == 100


def test_xgc_dataset_from_npy_with_sample(npy_data):
    """from_npy with sample subsamples correctly."""
    data_path, target_path, n_inputs, n_outputs = npy_data
    dataset = XGCDataset.from_npy(data_path, target_path, sample=20)
    assert len(dataset.df) == 20


def test_xgc_dataset_from_npy_custom_names(npy_data):
    """from_npy accepts custom input/output names."""
    data_path, target_path, n_inputs, n_outputs = npy_data
    input_names = [f"x{i}" for i in range(n_inputs)]
    output_names = ["aparh", "apars"]
    dataset = XGCDataset.from_npy(
        data_path, target_path,
        input_names=input_names,
        output_names=output_names,
    )
    assert dataset.input_columns == input_names
    assert dataset.output_columns == output_names


def test_xgc_dataset_from_npy_nonexistent():
    """from_npy raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError, match="Data file"):
        XGCDataset.from_npy("/nonexistent/data.npy", "/nonexistent/target.npy")


@pytest.fixture
def olcf_mock_dir(tmp_path):
    """Create minimal OLCF hackathon mock files."""
    np.random.seed(42)
    n_samples = 50
    n_inputs = 201
    n_outputs = 2
    data = np.random.randn(n_samples, n_inputs).astype(np.float32)
    target = np.random.randn(n_samples, n_outputs).astype(np.float32)
    var_all = np.array(["aparh", "apars", "dBphi", "dBpsi", "dBtheta", "ejpar",
                        "ijpar", "dpot", "epara", "epara2", "epsi", "etheta",
                        "eden", "iden"], dtype=object)
    np.save(tmp_path / "data_nprev5_set1_data.npy", data)
    np.save(tmp_path / "data_nprev5_set1_target.npy", target)
    np.save(tmp_path / "data_nprev5_set1_var_all.npy", var_all)
    return tmp_path


def test_xgc_dataset_from_olcf_hackathon(olcf_mock_dir):
    """from_olcf_hackathon loads OLCF .npy files."""
    dataset = XGCDataset.from_olcf_hackathon(olcf_mock_dir, set_name="set1")
    assert dataset.df is not None
    assert len(dataset.input_columns) == 201
    assert len(dataset.output_columns) == 2
    assert len(dataset.df) == 50


def test_xgc_dataset_from_olcf_hackathon_with_sample(olcf_mock_dir):
    """from_olcf_hackathon with sample subsamples correctly."""
    dataset = XGCDataset.from_olcf_hackathon(
        olcf_mock_dir, set_name="set1", sample=10
    )
    assert len(dataset.df) == 10


def test_xgc_dataset_from_olcf_hackathon_nonexistent_set():
    """from_olcf_hackathon raises FileNotFoundError for missing set."""
    with pytest.raises(FileNotFoundError, match="OLCF hackathon"):
        XGCDataset.from_olcf_hackathon("/tmp", set_name="nonexistent_set")


def test_surrogate_dataset_from_path_xgc_format(olcf_mock_dir):
    """SurrogateDataset.from_path with format=xgc loads OLCF data."""
    from surge.dataset import SurrogateDataset

    dataset = SurrogateDataset.from_path(
        olcf_mock_dir,
        format="xgc",
        analyzer_kwargs={"hints": {"set_name": "set1"}},
    )
    assert dataset.df is not None
    assert len(dataset.input_columns) == 201
    assert len(dataset.output_columns) == 2
    assert len(dataset.df) == 50


def test_surrogate_dataset_from_path_xgc_with_sample(olcf_mock_dir):
    """SurrogateDataset.from_path xgc with sample subsamples."""
    from surge.dataset import SurrogateDataset

    dataset = SurrogateDataset.from_path(
        olcf_mock_dir,
        format="xgc",
        sample=15,
        analyzer_kwargs={"hints": {"set_name": "set1"}},
    )
    assert len(dataset.df) == 15


def test_xgc_dataset_class_inherits_surrogate_dataset():
    """XGCDataset is a subclass of SurrogateDataset."""
    from surge.dataset import SurrogateDataset
    assert issubclass(XGCDataset, SurrogateDataset)


def test_xgc_dataset_iter_batches(npy_data):
    """iter_batches yields numpy batches (framework-agnostic)."""
    data_path, target_path, n_inputs, n_outputs = npy_data
    dataset = XGCDataset.from_npy(data_path, target_path)
    batches = list(dataset.iter_batches(batch_size=16, shuffle=False))
    assert len(batches) > 0
    X_batch, y_batch = batches[0]
    assert X_batch.shape[1] == n_inputs
    assert y_batch.shape[1] == n_outputs
    assert X_batch.shape[0] <= 16
    assert hasattr(X_batch, "shape")  # numpy array


def test_xgc_dataset_to_tf_dataset(npy_data):
    """to_tf_dataset returns tf.data.Dataset with correct batches."""
    pytest.importorskip("tensorflow")
    data_path, target_path, n_inputs, n_outputs = npy_data
    dataset = XGCDataset.from_npy(data_path, target_path)
    ds = dataset.to_tf_dataset(batch_size=16, shuffle=False)
    batches = list(ds.take(3))
    assert len(batches) > 0
    X_batch, y_batch = batches[0]
    assert X_batch.shape[1] == n_inputs
    assert y_batch.shape[1] == n_outputs


def test_xgc_dataset_to_dataloader(npy_data):
    """to_dataloader returns PyTorch DataLoader with correct batches."""
    pytest.importorskip("torch")
    data_path, target_path, n_inputs, n_outputs = npy_data
    dataset = XGCDataset.from_npy(data_path, target_path)
    loader = dataset.to_dataloader(batch_size=16, shuffle=False)
    batches = list(loader)
    assert len(batches) > 0
    X_batch, y_batch = batches[0]
    assert X_batch.shape[1] == n_inputs
    assert y_batch.shape[1] == n_outputs
    assert X_batch.shape[0] <= 16
