"""
Data operations for SURGE visualizer.

Handles dataset loading, statistics, correlations, distributions, and nearest-neighbor queries.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a dataset from file (supports pickle, CSV, parquet, etc.).
    
    Parameters
    ----------
    path : str or Path
        Path to dataset file.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    
    Raises
    ------
    ValueError
        If file format is not supported.
    """
    path = Path(path)
    
    if path.suffix == '.pkl' or path.suffix == '.pickle':
        df = pd.read_pickle(path)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(path)
    elif path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    elif path.suffix == '.h5' or path.suffix == '.hdf5':
        df = pd.read_hdf(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Ensure DataFrame (not Series)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    return df


def basic_stats(df: pd.DataFrame) -> Dict:
    """
    Compute basic statistics for a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    dict
        Dictionary containing:
        - shape: (rows, cols)
        - dtypes: Series of column dtypes
        - null_counts: Series of null counts per column
        - null_percentages: Series of null percentages per column
        - numeric_stats: describe() for numeric columns
        - memory_usage: memory usage in MB
    """
    stats = {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'null_counts': df.isnull().sum(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).round(2),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_stats'] = df[numeric_cols].describe()
    else:
        stats['numeric_stats'] = pd.DataFrame()
    
    return stats


def pearson_corr(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list of str, optional
        Specific columns to compute correlations for. If None, uses all numeric columns.
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    if cols is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[cols].select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    return numeric_df.corr()


def distributions(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare data for distribution plotting (long format).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list of str, optional
        Columns to include. If None, uses all numeric columns.
    
    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns: ['column', 'value']
    """
    if cols is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[cols].select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame(columns=['column', 'value'])
    
    # Convert to long format
    melted = numeric_df.melt(var_name='column', value_name='value')
    return melted.dropna()


def nearest_params(
    df: pd.DataFrame,
    query: Union[Dict, pd.Series, np.ndarray],
    param_cols: List[str],
    k: int = 10,
    metric: str = 'euclidean'
) -> pd.DataFrame:
    """
    Find k-nearest parameter vectors to a query.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to search in.
    query : dict, pd.Series, or np.ndarray
        Query parameters. If dict/Series, keys/indices should match param_cols.
        If array, should be 1D with length matching len(param_cols).
    param_cols : list of str
        Column names to use for distance computation.
    k : int, default=10
        Number of nearest neighbors to return.
    metric : str, default='euclidean'
        Distance metric (euclidean, manhattan, etc.).
    
    Returns
    -------
    pd.DataFrame
        Subset of df containing k-nearest rows, sorted by distance.
        Includes a 'distance' column.
    """
    # Validate columns exist
    missing_cols = set(param_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Extract parameter data
    param_data = df[param_cols].select_dtypes(include=[np.number])
    if param_data.empty:
        return df.iloc[:0].copy()
    
    # Convert query to array
    if isinstance(query, dict):
        query_vec = np.array([query[col] for col in param_cols])
    elif isinstance(query, pd.Series):
        query_vec = np.array([query[col] for col in param_cols])
    elif isinstance(query, np.ndarray):
        query_vec = query.ravel()
        if len(query_vec) != len(param_cols):
            raise ValueError(f"Query length {len(query_vec)} != param_cols length {len(param_cols)}")
    else:
        raise TypeError(f"Unsupported query type: {type(query)}")
    
    # Handle NaN in query
    if np.any(np.isnan(query_vec)):
        raise ValueError("Query contains NaN values")
    
    # Compute distances
    distances = cdist(
        param_data.values,
        query_vec.reshape(1, -1),
        metric=metric
    ).ravel()
    
    # Get k-nearest indices
    k_actual = min(k, len(df))
    nearest_indices = np.argsort(distances)[:k_actual]
    
    # Build result
    result = df.iloc[nearest_indices].copy()
    result['distance'] = distances[nearest_indices]
    result = result.sort_values('distance').reset_index(drop=True)
    
    return result


def prepare_inference_inputs(df: pd.DataFrame, param_cols: List[str]) -> Dict:
    """
    Prepare parameter ranges and defaults for inference controls.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    param_cols : list of str
        Parameter columns.
    
    Returns
    -------
    dict
        Dictionary mapping column names to {'min', 'max', 'mean', 'default'}
    """
    numeric_params = df[param_cols].select_dtypes(include=[np.number])
    
    param_info = {}
    for col in numeric_params.columns:
        param_info[col] = {
            'min': float(numeric_params[col].min()),
            'max': float(numeric_params[col].max()),
            'mean': float(numeric_params[col].mean()),
            'default': float(numeric_params[col].median()),
            'std': float(numeric_params[col].std()),
        }
    
    return param_info

