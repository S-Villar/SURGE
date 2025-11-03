"""
HoloViews visualization functions for SURGE visualizer.

Provides functions for plotting points, profiles, images, correlations, and distributions.
"""

from typing import Dict, List, Optional, Tuple, Union

try:
    import holoviews as hv
    import hvplot.pandas
except ImportError:
    hv = None
    hvplot = None

import numpy as np
import pandas as pd

if hv is not None:
    hv.extension('bokeh')


def plot_point(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    by: Optional[str] = None,
    title: Optional[str] = None
) -> hv.Scatter:
    """
    Plot point predictions (scatter plot).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing points to plot.
    y_col : str
        Column name for y-axis (target).
    x_col : str, optional
        Column name for x-axis. If None, uses index.
    by : str, optional
        Column to color by.
    title : str, optional
        Plot title.
    
    Returns
    -------
    hv.Scatter
        HoloViews Scatter plot.
    """
    if x_col is None:
        x_col = df.index.name or 'index'
        df_plot = df.reset_index()
    else:
        df_plot = df.copy()
    
    plot = df_plot.hvplot.scatter(
        x=x_col,
        y=y_col,
        by=by,
        title=title or f"{y_col} vs {x_col}",
        width=600,
        height=400,
        alpha=0.6,
    )
    
    return plot


def plot_profile(
    x: np.ndarray,
    y_gt: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_pred_std: Optional[np.ndarray] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: Optional[str] = None,
    show_residual: bool = True,
) -> hv.Layout:
    """
    Plot 1D profiles (ground truth vs prediction with optional residuals).
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates (e.g., radial coordinate ρ).
    y_gt : np.ndarray, optional
        Ground truth profile.
    y_pred : np.ndarray, optional
        Predicted profile.
    y_pred_std : np.ndarray, optional
        Uncertainty (standard deviation) for prediction.
    xlabel : str, default='x'
        X-axis label.
    ylabel : str, default='y'
        Y-axis label.
    title : str, optional
        Plot title.
    show_residual : bool, default=True
        Whether to show residual subplot.
    
    Returns
    -------
    hv.Layout
        Layout containing profile plot and optionally residual plot.
    """
    plots = []
    
    # Main profile plot
    profile_df = pd.DataFrame({'x': x})
    
    if y_gt is not None:
        profile_df['Ground Truth'] = y_gt
    if y_pred is not None:
        profile_df['Prediction'] = y_pred
    
    profile_plot = hv.Overlay([])
    if y_gt is not None:
        gt_curve = hv.Curve(
            (x, y_gt),
            label='Ground Truth',
            kdims=[xlabel],
            vdims=[ylabel],
        ).opts(color='black', line_width=2)
        profile_plot *= gt_curve
    
    if y_pred is not None:
        pred_curve = hv.Curve(
            (x, y_pred),
            label='Prediction',
            kdims=[xlabel],
            vdims=[ylabel],
        ).opts(color='red', line_width=1.5, line_dash='dashed')
        profile_plot *= pred_curve
        
        # Add uncertainty band if available
        if y_pred_std is not None:
            upper = y_pred + y_pred_std
            lower = y_pred - y_pred_std
            band = hv.Area(
                (x, lower, upper),
                kdims=[xlabel],
                vdims=['lower', 'upper'],
            ).opts(alpha=0.2, color='red')
            profile_plot *= band
    
    profile_plot = profile_plot.opts(
        width=700,
        height=400,
        title=title or f"{ylabel} Profile",
        legend_position='top_right',
    )
    
    plots.append(profile_plot)
    
    # Residual plot
    if show_residual and y_gt is not None and y_pred is not None:
        residual = y_gt - y_pred
        residual_df = pd.DataFrame({xlabel: x, 'residual': residual})
        residual_plot = residual_df.hvplot.line(
            x=xlabel,
            y='residual',
            title="Residual (GT - Pred)",
            width=700,
            height=200,
            color='blue',
            alpha=0.7,
        ) * hv.HLine(0).opts(color='black', line_dash='dotted')
        
        plots.append(residual_plot)
    
    return hv.Layout(plots).cols(1)


def plot_image(
    img: np.ndarray,
    img_pred: Optional[np.ndarray] = None,
    img_gt: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    show_residual: bool = True,
) -> Union[hv.Image, hv.Layout]:
    """
    Plot 2D images (heatmaps).
    
    Parameters
    ----------
    img : np.ndarray
        Image to plot (if img_pred and img_gt are None).
    img_pred : np.ndarray, optional
        Predicted image.
    img_gt : np.ndarray, optional
        Ground truth image.
    cmap : str, default='viridis'
        Colormap name.
    title : str, optional
        Plot title.
    show_residual : bool, default=True
        Whether to show residual image if both GT and pred are provided.
    
    Returns
    -------
    hv.Image or hv.Layout
        Image plot(s).
    """
    # Determine what to plot
    if img_pred is not None and img_gt is not None:
        plots = []
        
        # Ground truth
        plots.append(
            hv.Image(img_gt, kdims=['x', 'y'], label='Ground Truth')
            .opts(cmap=cmap, width=400, height=400, title='Ground Truth')
        )
        
        # Prediction
        plots.append(
            hv.Image(img_pred, kdims=['x', 'y'], label='Prediction')
            .opts(cmap=cmap, width=400, height=400, title='Prediction')
        )
        
        # Residual
        if show_residual:
            residual = img_gt - img_pred
            plots.append(
                hv.Image(residual, kdims=['x', 'y'], label='Residual')
                .opts(cmap='RdBu_r', width=400, height=400, title='Residual (GT - Pred)')
            )
        
        return hv.Layout(plots)
    elif img is not None:
        return hv.Image(img, kdims=['x', 'y']).opts(
            cmap=cmap,
            width=600,
            height=600,
            title=title or "Image",
        )
    else:
        raise ValueError("Must provide img, or both img_pred and img_gt")


def plot_corr_heatmap(corr_df: pd.DataFrame, title: Optional[str] = None) -> hv.HeatMap:
    """
    Plot correlation heatmap.
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation matrix.
    title : str, optional
        Plot title.
    
    Returns
    -------
    hv.HeatMap
        HoloViews HeatMap.
    """
    # Convert to long format for HoloViews
    corr_long = corr_df.reset_index().melt(
        id_vars='index',
        var_name='y',
        value_name='correlation'
    )
    corr_long = corr_long.rename(columns={'index': 'x'})
    
    heatmap = hv.HeatMap(
        corr_long,
        kdims=['x', 'y'],
        vdims='correlation'
    ).opts(
        cmap='RdBu_r',
        width=600,
        height=600,
        title=title or "Pearson Correlation Matrix",
        colorbar=True,
        symmetric=True,
        clim=(-1, 1),
    )
    
    return heatmap


def plot_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    n_cols: int = 3,
) -> hv.Layout:
    """
    Plot distributions (histograms) for multiple columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list of str, optional
        Columns to plot. If None, uses all numeric columns.
    n_cols : int, default=3
        Number of columns in layout grid.
    
    Returns
    -------
    hv.Layout
        Layout of histogram plots.
    """
    if cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in cols if c in df.columns and df[c].dtype in [np.number, 'float64', 'int64']]
    
    if not numeric_cols:
        # Return empty plot instead of empty Layout
        return hv.Curve([], label="No numeric columns available")
    
    plots = []
    for col in numeric_cols:
        hist = df.hvplot.hist(
            col,
            title=col,
            width=300,
            height=250,
            alpha=0.7,
        )
        plots.append(hist)
    
    return hv.Layout(plots).cols(n_cols)

