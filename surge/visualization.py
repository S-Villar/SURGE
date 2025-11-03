"""
Visualization functions for SURGE model results.

Provides functions to create Ground Truth vs Prediction plots with density heatmaps,
regression score visualizations, and other model performance plots.
"""

import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_gt_vs_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Dataset",
    title: Optional[str] = None,
    ax=None,
    bins: int = 100,
    cmap: str = 'plasma_r',
    show_colorbar: bool = True,
    show_r2: bool = True,
    show_diagonal: bool = True,
    alpha: float = 0.8,
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[str] = None,
    dpi: int = 150,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    units: Optional[str] = None,  # Defaults to 'a.u.' if None
    **kwargs
):
    """
    Plot Ground Truth vs Prediction with density heatmap using 2D histogram.
    
    Creates a density scatter plot where the color intensity (using 'plasma' colormap)
    indicates the number of data points in each bin. This is useful for visualizing
    regression performance when there are many overlapping points.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values. Shape: (n_samples,) or (n_samples, n_outputs).
        For multi-output, will plot the first output or aggregate.
    y_pred : np.ndarray
        Predicted values. Shape should match y_true.
    dataset_name : str, default="Dataset"
        Name of the dataset (e.g., "Training Set", "Test Set")
    title : str, optional
        Plot title. If None, auto-generated from dataset_name.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    bins : int, default=100
        Number of bins for the 2D histogram (bins x bins). Higher values give finer resolution.
    cmap : str, default='plasma_r'
        Colormap for the density heatmap. Defaults to inverse plasma ('plasma_r').
        Use 'plasma' for original direction, or any other matplotlib colormap.
    show_colorbar : bool, default=True
        Whether to show the colorbar.
    show_r2 : bool, default=True
        Whether to display R² score on the plot.
    show_diagonal : bool, default=True
        Whether to show the diagonal line (y=x) for perfect prediction.
    alpha : float, default=0.8
        Transparency of the heatmap.
    figsize : tuple, default=(8, 8)
        Figure size (width, height) in inches.
    save_path : str, optional
        Path to save the figure. If None, figure is displayed.
    dpi : int, default=150
        Resolution for saved figures.
    **kwargs
        Additional arguments passed to imshow or pcolormesh.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    r2_score : float
        The R² score for the predictions.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Flatten if multi-output (use first output or mean)
    if y_true.ndim > 1:
        if y_true.shape[1] == 1:
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
        else:
            # For multi-output, plot the first output by default
            # User can call this function separately for each output
            print(f"⚠️ Multi-output detected ({y_true.shape[1]} outputs). Plotting first output only.")
            y_true = y_true[:, 0]
            y_pred = y_pred[:, 0]
    
    # Ensure 1D arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove NaN and Inf values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        raise ValueError("No valid data points to plot")
    
    # Calculate R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Determine axis limits (with small margin)
    y_min = min(np.min(y_true), np.min(y_pred))
    y_max = max(np.max(y_true), np.max(y_pred))
    margin = (y_max - y_min) * 0.05
    axis_limits = [y_min - margin, y_max + margin]
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create 2D histogram for density
    hist, xedges, yedges = np.histogram2d(
        y_true, y_pred, bins=bins, range=[axis_limits, axis_limits]
    )
    
    # Use LogNorm for better visualization of density range
    hist = hist.T  # Transpose for imshow
    hist_nonzero = hist[hist > 0]
    
    if len(hist_nonzero) > 0:
        vmin = max(1, np.min(hist_nonzero))
        vmax = np.max(hist)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    
    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        hist,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        interpolation='nearest',
        **kwargs
    )
    
    # Add diagonal line for perfect prediction
    if show_diagonal:
        ax.plot(axis_limits, axis_limits, 'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    # Add R² score annotation
    if show_r2:
        ax.text(0.05, 0.95, f'R² = {r2:.2f}', 
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of data points', rotation=270, labelpad=20)
    
    # Set labels and title - use dynamic labels if provided
    if xlabel is None:
        xlabel = 'Ground Truth'
    if ylabel is None:
        ylabel = 'Prediction'
    
    # Add units - default to a.u. (arbitrary units) if not provided
    if units is None:
        units = 'a.u.'  # Default to arbitrary units
    if units:
        xlabel = f'{xlabel} [{units}]'
        ylabel = f'{ylabel} [{units}]'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        title = f'{dataset_name} - Ground Truth vs Prediction'
    ax.set_title(title, fontsize=14, pad=10)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect('equal', adjustable='box')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig, ax, r2


def plot_regression_comparison(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    model_name: str = "Model",
    output_name: str = "",
    bins: int = 100,
    cmap: str = 'plasma_r',
    figsize: Tuple[float, float] = (16, 8),
    save_path: Optional[str] = None,
    dpi: int = 150,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    units: Optional[str] = None,  # Defaults to 'a.u.' if None
):
    """
    Create side-by-side comparison plots for training and test sets.
    
    Creates two density heatmap plots showing Ground Truth vs Prediction
    for both training and test datasets, similar to publication-quality figures.
    
    Parameters
    ----------
    y_true_train : np.ndarray
        Ground truth values for training set.
    y_pred_train : np.ndarray
        Predicted values for training set.
    y_true_test : np.ndarray
        Ground truth values for test set.
    y_pred_test : np.ndarray
        Predicted values for test set.
    model_name : str, default="Model"
        Name of the model (e.g., "RFR", "MLP").
    output_name : str, default=""
        Name of the output variable (e.g., "P_e", "P_D"). Will be used in plot titles.
    bins : int, default=100
        Number of bins for the 2D histogram.
    units : str, optional
        Units for the output variable (e.g., "W/cm³/MWabs"). Defaults to "a.u." (arbitrary units) if None.
        Will be appended to axis labels in brackets.
    cmap : str, default='plasma_r'
        Colormap for the density heatmap. Defaults to inverse plasma ('plasma_r').
        Use 'plasma' for original direction, or any other matplotlib colormap.
    figsize : tuple, default=(16, 8)
        Figure size (width, height) in inches.
    save_path : str, optional
        Path to save the figure.
    dpi : int, default=150
        Resolution for saved figures.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.
    results : dict
        Dictionary with R² scores for train and test sets.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Build title dynamically
    if output_name:
        title_suffix = f" {output_name}" if units else output_name
        title_train = f"(e){title_suffix}" if title_suffix else "(e) Training Set"
        title_test = f"(f){title_suffix}" if title_suffix else "(f) Test Set"
    else:
        title_train = "(e) Training Set"
        title_test = "(f) Test Set"
    
    # Plot training set
    fig_train, ax_train, r2_train = plot_gt_vs_prediction(
        y_true_train, y_pred_train,
        dataset_name=f"Training Set",
        title=title_train,
        ax=axes[0],
        bins=bins,
        cmap=cmap,
        show_colorbar=True,
        show_r2=True,
        show_diagonal=True,
        xlabel=xlabel,
        ylabel=ylabel,
        units=units,
    )
    
    # Plot test set
    fig_test, ax_test, r2_test = plot_gt_vs_prediction(
        y_true_test, y_pred_test,
        dataset_name=f"Test Set",
        title=title_test,
        ax=axes[1],
        bins=bins,
        cmap=cmap,
        show_colorbar=True,
        show_r2=True,
        show_diagonal=True,
        xlabel=xlabel,
        ylabel=ylabel,
        units=units,
    )
    
    # Update colorbar labels - find colorbars from figure
    # The colorbars are stored in fig.colorbar instances
    if len(fig.axes) >= 2:
        # Colorbar axes are typically after the main plot axes
        # For side-by-side plots, we have 2 plot axes + 2 colorbar axes
        for idx, ax_cbar in enumerate(fig.axes[2:]):
            if idx == 0:  # First colorbar (training set)
                ax_cbar.set_ylabel("Number of data points (training set)", rotation=270, labelpad=20)
            elif idx == 1:  # Second colorbar (test set)
                ax_cbar.set_ylabel("Number of data points (test set)", rotation=270, labelpad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {save_path}")
    
    results = {
        'r2_train': r2_train,
        'r2_test': r2_test,
    }
    
    return fig, axes, results


def plot_multi_output_comparison(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    output_names: List[str],
    model_name: str = "Model",
    bins: int = 100,
    cmap: str = 'plasma_r',
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    max_outputs: int = 4,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    units: Optional[str] = None,  # Defaults to 'a.u.' if None
):
    """
    Create comparison plots for multiple outputs.
    
    Creates a grid of plots showing Ground Truth vs Prediction for each output variable.
    
    Parameters
    ----------
    y_true_train : np.ndarray
        Ground truth values for training set. Shape: (n_samples, n_outputs)
    y_pred_train : np.ndarray
        Predicted values for training set. Shape: (n_samples, n_outputs)
    y_true_test : np.ndarray
        Ground truth values for test set. Shape: (n_samples, n_outputs)
    y_pred_test : np.ndarray
        Predicted values for test set. Shape: (n_samples, n_outputs)
    output_names : List[str]
        List of output variable names.
    model_name : str, default="Model"
        Name of the model.
    bins : int, default=100
        Number of bins for the 2D histogram. Higher values give finer resolution but slower computation.
    cmap : str, default='plasma_r'
        Colormap for the density heatmap. Defaults to inverse plasma ('plasma_r').
        Use 'plasma' for original direction, or any other matplotlib colormap.
    units : str, optional
        Units for the output variable. Defaults to "a.u." (arbitrary units) if None.
        Will be appended to axis labels in brackets.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    save_path : str, optional
        Path to save the figure.
    dpi : int, default=150
        Resolution for saved figures.
    max_outputs : int, default=4
        Maximum number of outputs to plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : np.ndarray
        Array of axes objects.
    results : dict
        Dictionary with R² scores for each output.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Determine number of outputs to plot
    n_outputs = min(len(output_names), max_outputs, y_true_train.shape[1] if y_true_train.ndim > 1 else 1)
    
    if n_outputs == 1:
        # Single output: use comparison plot
        return plot_regression_comparison(
            y_true_train, y_pred_train,
            y_true_test, y_pred_test,
            model_name=model_name,
            output_name=output_names[0] if output_names else "",
            bins=bins,
            cmap=cmap,
            save_path=save_path,
            dpi=dpi,
            xlabel=xlabel,
            ylabel=ylabel,
            units=units,
        )
    
    # Create grid: 2 rows (train/test), n_outputs columns
    if figsize is None:
        figsize = (8 * n_outputs, 16)
    
    fig, axes = plt.subplots(2, n_outputs, figsize=figsize)
    if n_outputs == 1:
        axes = axes.reshape(-1, 1)
    
    results = {}
    
    # Plot each output
    for idx in range(n_outputs):
        output_name = output_names[idx] if idx < len(output_names) else f"Output {idx+1}"
        
        # Extract data for this output
        if y_true_train.ndim > 1:
            y_true_train_out = y_true_train[:, idx]
            y_pred_train_out = y_pred_train[:, idx]
            y_true_test_out = y_true_test[:, idx]
            y_pred_test_out = y_pred_test[:, idx]
        else:
            y_true_train_out = y_true_train
            y_pred_train_out = y_pred_train
            y_true_test_out = y_true_test
            y_pred_test_out = y_pred_test
        
        # Build title dynamically based on output name
        title_train = f"(e) {output_name}" if output_name else f"(e) Output {idx+1}"
        title_test = f"(f) {output_name}" if output_name else f"(f) Output {idx+1}"
        
        # Training set plot
        _, _, r2_train = plot_gt_vs_prediction(
            y_true_train_out, y_pred_train_out,
            dataset_name="Training Set",
            title=title_train,
            ax=axes[0, idx],
            bins=bins,
            cmap=cmap,
            show_colorbar=(idx == n_outputs - 1),  # Only show colorbar on last column
            show_r2=True,
            show_diagonal=True,
            xlabel=xlabel,
            ylabel=ylabel,
            units=units,
        )
        
        # Test set plot
        _, _, r2_test = plot_gt_vs_prediction(
            y_true_test_out, y_pred_test_out,
            dataset_name="Test Set",
            title=title_test,
            ax=axes[1, idx],
            bins=bins,
            cmap=cmap,
            show_colorbar=(idx == n_outputs - 1),
            show_r2=True,
            show_diagonal=True,
            xlabel=xlabel,
            ylabel=ylabel,
            units=units,
        )
        
        results[output_name] = {
            'r2_train': r2_train,
            'r2_test': r2_test,
        }
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Multi-output comparison plot saved to: {save_path}")
    
    return fig, axes, results

