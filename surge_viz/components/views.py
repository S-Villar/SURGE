"""
View components for SURGE visualizer.

Provides composable Panel views for different sections.
"""

from typing import Dict, List, Optional

try:
    import holoviews as hv
except ImportError:
    hv = None

import numpy as np
import panel as pn
import pandas as pd

try:
    from ..data_ops import basic_stats, distributions as get_distributions, nearest_params, pearson_corr
    from ..surge_api import infer, load_model
    from ..viz import plot_corr_heatmap, plot_distributions, plot_image, plot_point, plot_profile
    from .controls import CompareControls, DatasetControls, InferenceControls, ModelControls
except ImportError:
    # Handle relative imports when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from surge_viz.data_ops import basic_stats, distributions as get_distributions, nearest_params, pearson_corr
    from surge_viz.surge_api import infer, load_model
    from surge_viz.viz import plot_corr_heatmap, plot_distributions, plot_image, plot_point, plot_profile
    from surge_viz.components.controls import CompareControls, DatasetControls, InferenceControls, ModelControls


class DataView:
    """Data exploration view."""
    
    def __init__(self, controls: DatasetControls):
        self.controls = controls
        self.df: Optional[pd.DataFrame] = None
        self.stats_pane = pn.pane.DataFrame(pd.DataFrame(), width=600)
        
        # Status indicator
        self.status_pane = pn.pane.Str("📋 No dataset loaded. Select a file and click 'Load Dataset' button below.", width=600)
        self.status_pane.css_classes = ['alert', 'alert-info']
        
        # Initialize with empty placeholder instead of empty Layout
        if hv is not None:
            empty_plot = hv.Curve([], label="No data loaded - load a dataset to see statistics")
        else:
            empty_plot = None
        if hv is not None:
            self.corr_plot = pn.pane.HoloViews(empty_plot, width=500, height=400)
            self.dist_plot = pn.pane.HoloViews(empty_plot, width=500, height=300)
        else:
            self.corr_plot = pn.pane.Markdown("**HoloViews not available** - Install with: pip install holoviews")
            self.dist_plot = pn.pane.Markdown("**HoloViews not available** - Install with: pip install holoviews")
    
    def load_data(self, file_path: str):
        """Load dataset from file."""
        try:
            from surge_viz.data_ops import load_dataset
        except ImportError:
            from ..data_ops import load_dataset
        
        try:
            self.df = load_dataset(file_path)
            
            if self.df is None or self.df.empty:
                print("Warning: Dataset is None or empty")
                self.stats_pane.object = pd.DataFrame({"Error": ["Dataset is empty"]})
                return
            
            # Update status
            self.status_pane.object = f"✅ Dataset loaded successfully!\n📊 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n🔢 Numeric columns: {len(self.df.select_dtypes(include=[np.number]).columns)}"
            self.status_pane.css_classes = ['alert', 'alert-success']
            
            # Update controls
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"DEBUG DataView.load_data: Found {len(numeric_cols)} numeric columns")
            if numeric_cols:
                # Update the Param object
                self.controls.numeric_columns = numeric_cols
                # Directly update the widget options
                self.controls.param_selector.options = numeric_cols
                self.controls.output_selector.options = numeric_cols
                # Force a parameter update to trigger reactive updates
                self.controls.param.trigger('numeric_columns')
                print(f"DEBUG: Updated selectors with {len(numeric_cols)} columns")
            else:
                self.stats_pane.object = pd.DataFrame({"Warning": ["No numeric columns found in dataset"]})
                self.status_pane.object = "⚠️ Dataset loaded but contains no numeric columns."
                self.status_pane.css_classes = ['alert', 'alert-warning']
                print("DEBUG: No numeric columns found")
                return
            
            # Update stats
            try:
                stats = basic_stats(self.df)
                numeric_stats = stats.get('numeric_stats', pd.DataFrame())
                if not numeric_stats.empty:
                    self.stats_pane.object = numeric_stats
                else:
                    self.stats_pane.object = pd.DataFrame({"Info": ["No statistics available"]})
            except Exception as e:
                print(f"Error computing stats: {e}")
                self.stats_pane.object = pd.DataFrame({"Error": [f"Could not compute statistics: {str(e)}"]})
            
            # Update correlations
            try:
                corr_df = pearson_corr(self.df)
                if hv is not None and not corr_df.empty:
                    corr_plot_obj = plot_corr_heatmap(corr_df)
                    if corr_plot_obj is not None:
                        self.corr_plot.object = corr_plot_obj
                    else:
                        if hv is not None:
                            self.corr_plot.object = hv.Curve([], label="No correlation plot available")
                        else:
                            self.corr_plot.object = pn.pane.Markdown("HoloViews not available for correlation plot")
                else:
                    if hv is not None:
                        self.corr_plot.object = hv.Curve([], label="No numeric data for correlation")
                    else:
                        self.corr_plot.object = pn.pane.Markdown("No correlation data available")
            except Exception as e:
                print(f"Error plotting correlations: {e}")
                if hv is not None:
                    self.corr_plot.object = hv.Curve([], label=f"Error: {str(e)}")
                else:
                    self.corr_plot.object = pn.pane.Markdown(f"Error plotting correlations: {str(e)}")
            
            # Update distributions
            try:
                if numeric_cols:
                    dist_plot_obj = plot_distributions(self.df, cols=numeric_cols[:6])
                    if hv is not None and dist_plot_obj is not None:
                        self.dist_plot.object = dist_plot_obj
                    else:
                        if hv is not None:
                            self.dist_plot.object = hv.Curve([], label="No distributions available")
                        else:
                            self.dist_plot.object = pn.pane.Markdown("Distributions not available")
                else:
                    if hv is not None:
                        self.dist_plot.object = hv.Curve([], label="No numeric columns selected")
                    else:
                        self.dist_plot.object = pn.pane.Markdown("No numeric columns selected")
            except Exception as e:
                print(f"Error plotting distributions: {e}")
                if hv is not None:
                    self.dist_plot.object = hv.Curve([], label=f"Error: {str(e)}")
                else:
                    self.dist_plot.object = pn.pane.Markdown(f"Error plotting distributions: {str(e)}")
                    
        except Exception as e:
            import traceback
            error_msg = f"❌ Error loading data: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            self.status_pane.object = error_msg
            self.status_pane.css_classes = ['alert', 'alert-danger']
            self.stats_pane.object = pd.DataFrame({"Error": [str(e)]})
            if hv is not None:
                self.corr_plot.object = hv.Curve([], label=error_msg)
                self.dist_plot.object = hv.Curve([], label=error_msg)
            else:
                self.corr_plot.object = pn.pane.Markdown(error_msg)
                self.dist_plot.object = pn.pane.Markdown(error_msg)
    
    def _update_selectors_from_columns(self):
        """Update selector widgets when numeric_columns changes."""
        if hasattr(self, 'controls') and self.controls.numeric_columns:
            # Directly update widget options
            if hasattr(self.controls, 'param_selector'):
                self.controls.param_selector.options = self.controls.numeric_columns
            if hasattr(self.controls, 'output_selector'):
                self.controls.output_selector.options = self.controls.numeric_columns
            print(f"DEBUG _update_selectors_from_columns: Updated selectors with {len(self.controls.numeric_columns)} columns")
    
    def panel(self) -> pn.Column:
        """Return Panel view."""
        # Wire up reactive update
        if hasattr(self.controls, 'param'):
            self.controls.param.watch(lambda event: self._update_selectors_from_columns(), 'numeric_columns')
        
        # Re-wire the load button callback here too
        # This ensures it works when the view is recreated
        try:
            if hasattr(self.controls, 'load_button'):
                # Remove any existing callbacks first
                if hasattr(self.controls.load_button.param, '_callbacks'):
                    try:
                        self.controls.load_button.param._callbacks['clicks'] = []
                    except:
                        pass
                
                # Get the parent app to wire the callback
                # We'll need to pass a callback function that can access the app
                # For now, we'll wire it in the app's panel() method
                pass
        except Exception as e:
            print(f"DEBUG: Could not re-wire load button in DataView: {e}")
        
        try:
            controls_panel = self.controls.panel()
        except Exception as e:
            controls_panel = pn.pane.Markdown(f"Error loading controls: {str(e)}", sizing_mode='stretch_width')
        
        return pn.Column(
            pn.pane.Markdown("## 📊 Data Exploration", sizing_mode='stretch_width'),
            pn.pane.Markdown("**Step 1:** Load a dataset using the file input below.", sizing_mode='stretch_width'),
            controls_panel,
            pn.pane.Markdown("---", sizing_mode='stretch_width'),
            pn.pane.Markdown("### 📋 Status", sizing_mode='stretch_width'),
            self.status_pane,
            pn.pane.Markdown("---", sizing_mode='stretch_width'),
            pn.pane.Markdown("### Statistics", sizing_mode='stretch_width'),
            pn.Row(
                self.stats_pane,
                sizing_mode='stretch_width',
                max_width=800
            ),
            pn.pane.Markdown("### Correlations", sizing_mode='stretch_width'),
            pn.Row(
                self.corr_plot,
                sizing_mode='stretch_width',
                max_width=800
            ),
            pn.pane.Markdown("### Distributions", sizing_mode='stretch_width'),
            pn.Row(
                self.dist_plot,
                sizing_mode='stretch_width',
                max_width=800
            ),
            sizing_mode='stretch_width',
            max_width=1200
        )


class ModelView:
    """Model training and loading view."""
    
    def __init__(self, controls: ModelControls):
        self.controls = controls
        self.model_handle = None
        self.status_pane = pn.pane.Str("No model loaded")
    
    def panel(self) -> pn.Column:
        """Return Panel view."""
        try:
            controls_panel = self.controls.panel()
        except Exception as e:
            controls_panel = pn.pane.Markdown(f"Error loading controls: {str(e)}", sizing_mode='stretch_width')
        
        return pn.Column(
            pn.pane.Markdown("## 🤖 Model Management", sizing_mode='stretch_width'),
            pn.pane.Markdown("**Step 1:** Select parameter and output columns in the Data tab.\n**Step 2:** Choose a model type and click Train.", sizing_mode='stretch_width'),
            controls_panel,
            pn.pane.Markdown("---", sizing_mode='stretch_width'),
            pn.pane.Markdown("### Status", sizing_mode='stretch_width'),
            self.status_pane,
            sizing_mode='stretch_width',
            max_width=1200
        )


class InferenceView:
    """Inference and comparison view."""
    
    def __init__(
        self,
        inference_controls: InferenceControls,
        compare_controls: CompareControls,
        model_handle=None,
        df: Optional[pd.DataFrame] = None
    ):
        self.inference_controls = inference_controls
        self.compare_controls = compare_controls
        self.model_handle = model_handle
        self.df = df
        # Initialize with empty placeholder instead of empty Layout
        if hv is not None:
            empty_plot = hv.Curve([], label="No results yet - train a model and run inference")
        else:
            empty_plot = None
        if hv is not None:
            self.result_plot = pn.pane.HoloViews(empty_plot, width=700, height=500)
        else:
            self.result_plot = pn.pane.Markdown("**HoloViews not available** - Install with: pip install holoviews")
        self.metric_pane = pn.pane.Str("Run inference to see metrics")
    
    def run_inference(self):
        """Run inference and update plots."""
        if self.model_handle is None:
            self.result_plot.object = hv.Curve([], label="No model loaded")
            self.metric_pane.object = "Error: No model loaded"
            return
        
        # Get parameter values
        params = self.inference_controls.param_values
        if not params:
            self.result_plot.object = hv.Curve([], label="No parameters set")
            self.metric_pane.object = "Error: No parameters set"
            return
        
        # Run inference
        try:
            result = infer(self.model_handle, params)
            self._update_plots(result)
        except Exception as e:
            self.result_plot.object = hv.Curve([], label=f"Error: {str(e)}")
            self.metric_pane.object = f"Error: {str(e)}"
    
    def _update_plots(self, result: Dict):
        """Update plots based on inference result."""
        pred_type = result.get('type', 'point')
        y_pred = result.get('yhat', None)
        y_pred_std = result.get('yhat_std', None)
        
        if pred_type == 'point':
            # Point prediction - simple numeric display
            pred_text = f"Prediction: {y_pred[0, 0]:.4f}" if y_pred is not None else "No prediction"
            if y_pred_std is not None:
                pred_text += f" ± {y_pred_std[0, 0]:.4f}"
            self.metric_pane.object = pred_text
            # Use a simple scatter plot for point predictions
            self.result_plot.object = hv.Scatter([(0, y_pred[0, 0])], label=pred_text) if y_pred is not None else hv.Curve([], label="No prediction")
        
        elif pred_type == 'profile':
            # Profile prediction
            x = np.arange(len(y_pred[0])) if y_pred is not None else np.array([])
            profile_plot = plot_profile(
                x=x,
                y_pred=y_pred[0] if y_pred is not None else None,
                y_pred_std=y_pred_std[0] if y_pred_std is not None else None,
            )
            self.result_plot.object = profile_plot
        
        elif pred_type == 'image':
            # Image prediction
            img_shape = result.get('meta', {}).get('image_shape', None)
            if img_shape:
                img = y_pred[0].reshape(img_shape) if y_pred is not None else None
                img_plot = plot_image(img, title="Prediction")
                self.result_plot.object = img_plot
    
    def panel(self) -> pn.Column:
        """Return Panel view."""
        try:
            inference_panel = self.inference_controls.panel()
        except Exception as e:
            inference_panel = pn.pane.Markdown(f"Error loading inference controls: {str(e)}", sizing_mode='stretch_width')
        
        try:
            compare_panel = self.compare_controls.panel()
        except Exception as e:
            compare_panel = pn.pane.Markdown(f"Error loading compare controls: {str(e)}", sizing_mode='stretch_width')
        
        return pn.Column(
            pn.pane.Markdown("## 🔮 Inference & Comparison", sizing_mode='stretch_width'),
            pn.pane.Markdown("**Step 1:** Train a model in the Model tab.\n**Step 2:** Adjust parameter sliders and click Run Inference.", sizing_mode='stretch_width'),
            pn.Row(
                inference_panel,
                compare_panel,
                sizing_mode='stretch_width',
                max_width=1000
            ),
            pn.pane.Markdown("---", sizing_mode='stretch_width'),
            pn.pane.Markdown("### Results", sizing_mode='stretch_width'),
            pn.Row(
                self.result_plot,
                sizing_mode='stretch_width',
                max_width=900
            ),
            pn.pane.Markdown("### Metrics", sizing_mode='stretch_width'),
            self.metric_pane,
            sizing_mode='stretch_width',
            max_width=1200
        )

