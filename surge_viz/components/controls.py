"""
Control components for SURGE visualizer.

Provides Param-based classes for dataset, model, inference, and comparison controls.
"""

from pathlib import Path
from typing import Dict, List, Optional

import panel as pn
import param


class DatasetControls(param.Parameterized):
    """Controls for dataset loading and exploration."""
    
    file_path = param.String(default="", doc="Path to dataset file")
    loaded = param.Boolean(default=False, doc="Whether dataset is loaded")
    numeric_columns = param.List(default=[], doc="Numeric columns in dataset")
    param_columns = param.List(default=[], doc="Selected parameter columns")
    output_columns = param.List(default=[], doc="Selected output columns")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.file_input = pn.widgets.FileInput(
            accept='.pkl,.pickle,.csv,.parquet,.h5,.hdf5',
            name="Choose Dataset File"
        )
        self.file_input.param.watch(self._on_file_select, 'value')
        
        # Load button - user must click this to load
        self.load_button = pn.widgets.Button(
            name="📂 Load",
            button_type="primary",
            disabled=True,
            width=100,
            margin=(5, 0, 0, 0)  # Align with file input
        )
        # Don't wire callback here - let the app do it
        # This avoids multiple wiring when panel is recreated
        
        self.param_selector = pn.widgets.MultiChoice(
            name="Parameter Columns",
            options=[],
            value=[],
        )
        self.param_selector.param.watch(self._on_param_select, 'value')
        
        self.output_selector = pn.widgets.MultiChoice(
            name="Output Columns",
            options=[],
            value=[],
        )
        self.output_selector.param.watch(self._on_output_select, 'value')
    
    def _on_file_select(self, event):
        """Handle file selection."""
        if event.new:
            self.file_path = event.new
            # Enable the load button when a file is selected
            if hasattr(self, 'load_button'):
                self.load_button.disabled = False
                # Get filename for display
                filename = getattr(event.new, 'filename', 'file')
                if filename and len(filename) > 15:
                    filename = filename[:12] + "..."
                self.load_button.name = f"📂 Load {filename}" if filename else "📂 Load"
                print(f"DEBUG _on_file_select: File selected, button enabled. Filename: {filename}")
    
    
    def _on_param_select(self, event):
        """Handle parameter column selection."""
        self.param_columns = list(event.new) if event.new else []
    
    def _on_output_select(self, event):
        """Handle output column selection."""
        self.output_columns = list(event.new) if event.new else []
    
    def panel(self) -> pn.Column:
        """Return Panel widgets."""
        # Put file input and load button in a row
        file_row = pn.Row(
            self.file_input,
            self.load_button,
            sizing_mode='stretch_width',
            margin=(0, 0, 10, 0)
        )
        
        return pn.Column(
            pn.pane.Markdown("### Dataset Controls", sizing_mode='stretch_width'),
            pn.pane.Markdown("**Load Dataset:**", sizing_mode='stretch_width'),
            file_row,
            pn.pane.Markdown("---", sizing_mode='stretch_width'),
            pn.pane.Markdown("**Select Parameter Columns** (inputs for model):", sizing_mode='stretch_width'),
            self.param_selector,
            pn.pane.Markdown("**Select Output Columns** (targets to predict):", sizing_mode='stretch_width'),
            self.output_selector,
            sizing_mode='stretch_width',
            max_width=800
        )


class ModelControls(param.Parameterized):
    """Controls for model training and loading."""
    
    model_type = param.Selector(
        default='random_forest',
        objects=['random_forest', 'mlp', 'pytorch.mlp_model', 'gpflow.gpr'],
        doc="Model type to train"
    )
    model_id = param.String(default="", doc="Model ID to load")
    training = param.Boolean(default=False, doc="Whether model is training")
    trained = param.Boolean(default=False, doc="Whether model is trained")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.model_type_selector = pn.widgets.Select(
            name="Model Type",
            options=self.param.model_type.objects,
            value=self.model_type,
        )
        self.model_type_selector.param.watch(self._on_model_type_select, 'value')
        
        self.train_button = pn.widgets.Button(
            name="Train Model",
            button_type="primary"
        )
        
        self.load_button = pn.widgets.Button(
            name="Load Model",
            button_type="primary"
        )
        
        self.model_id_input = pn.widgets.TextInput(
            name="Model ID",
            placeholder="Enter model ID or path"
        )
    
    def _on_model_type_select(self, event):
        """Handle model type selection."""
        self.model_type = event.new
    
    def panel(self) -> pn.Column:
        """Return Panel widgets."""
        return pn.Column(
            pn.pane.Markdown("### Model Controls", sizing_mode='stretch_width'),
            pn.pane.Markdown("**Model Type:**", sizing_mode='stretch_width'),
            self.model_type_selector,
            self.train_button,
            pn.pane.Markdown("**Or Load Existing Model:**", sizing_mode='stretch_width'),
            self.model_id_input,
            self.load_button,
            sizing_mode='stretch_width',
            max_width=700
        )


class InferenceControls(param.Parameterized):
    """Controls for inference with parameter inputs."""
    
    param_values = param.Dict(default={}, doc="Parameter values for inference")
    run_inference = param.Boolean(default=False, doc="Toggle to run inference")
    
    def __init__(self, param_info: Optional[Dict] = None, **params):
        super().__init__(**params)
        self.param_info = param_info or {}
        self.param_widgets = {}
        self._build_param_widgets()
        
        self.inference_button = pn.widgets.Button(
            name="Run Inference",
            button_type="primary"
        )
    
    def _build_param_widgets(self):
        """Build parameter widgets from param_info."""
        if not self.param_info:
            # No parameters - add a message widget
            msg_widget = pn.pane.Markdown("**No parameters available.** Select parameter columns in the Data tab first.", sizing_mode='stretch_width')
            self.param_widgets['_placeholder'] = msg_widget
            return
        
        for param_name, info in self.param_info.items():
            try:
                min_val = float(info.get('min', 0))
                max_val = float(info.get('max', 1))
                default_val = float(info.get('default', info.get('mean', (min_val + max_val) / 2)))
                step = max((max_val - min_val) / 100, 0.001)  # Ensure step is positive
                
                widget = pn.widgets.FloatSlider(
                    name=param_name,
                    start=min_val,
                    end=max_val,
                    value=default_val,
                    step=step,
                )
                widget.param.watch(lambda event, name=param_name: self._update_param(name, event.new), 'value')
                self.param_widgets[param_name] = widget
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not create widget for {param_name}: {e}")
                continue
    
    def _update_param(self, name: str, value: float):
        """Update parameter value."""
        self.param_values[name] = value
    
    def panel(self) -> pn.Column:
        """Return Panel widgets."""
        widgets = [
            pn.pane.Markdown("### Inference Controls", sizing_mode='stretch_width'),
            pn.pane.Markdown("Adjust parameters and click Run Inference:", sizing_mode='stretch_width')
        ]
        
        if self.param_widgets:
            widgets.extend(self.param_widgets.values())
        else:
            widgets.append(pn.pane.Markdown("**No parameters available.** Select parameter columns in the Data tab and train a model.", sizing_mode='stretch_width'))
        
        widgets.append(self.inference_button)
        return pn.Column(*widgets, sizing_mode='stretch_width', max_width=600)


class CompareControls(param.Parameterized):
    """Controls for comparison mode."""
    
    metric = param.Selector(
        default='RMSE',
        objects=['RMSE', 'MAE', 'R2', 'MSE'],
        doc="Metric for comparison"
    )
    view_mode = param.Selector(
        default='overlay',
        objects=['overlay', 'side_by_side', 'residual'],
        doc="View mode"
    )
    
    def __init__(self, **params):
        super().__init__(**params)
        self.metric_selector = pn.widgets.Select(
            name="Metric",
            options=self.param.metric.objects,
            value=self.metric,
        )
        self.metric_selector.param.watch(lambda e: setattr(self, 'metric', e.new), 'value')
        
        self.view_mode_selector = pn.widgets.Select(
            name="View Mode",
            options=self.param.view_mode.objects,
            value=self.view_mode,
        )
        self.view_mode_selector.param.watch(lambda e: setattr(self, 'view_mode', e.new), 'value')
    
    def panel(self) -> pn.Column:
        """Return Panel widgets."""
        return pn.Column(
            pn.pane.Markdown("### Comparison Controls", sizing_mode='stretch_width'),
            pn.pane.Markdown("**Metric:**", sizing_mode='stretch_width'),
            self.metric_selector,
            pn.pane.Markdown("**View Mode:**", sizing_mode='stretch_width'),
            self.view_mode_selector,
            sizing_mode='stretch_width',
            max_width=400
        )

