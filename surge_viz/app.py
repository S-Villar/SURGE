"""
Main Panel application for SURGE visualizer.

Entry point for the web-based visualization platform.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path to handle imports
__file_path__ = Path(__file__).resolve()
__parent_dir__ = __file_path__.parent.parent
if str(__parent_dir__) not in sys.path:
    sys.path.insert(0, str(__parent_dir__))

import panel as pn

# Use absolute imports
from surge_viz.components.controls import CompareControls, DatasetControls, InferenceControls, ModelControls
from surge_viz.components.views import DataView, InferenceView, ModelView
from surge_viz.data_ops import prepare_inference_inputs

# Set Panel extension with responsive sizing
pn.extension('tabulator', sizing_mode='stretch_width')

# Load HoloViews extension separately
try:
    import holoviews as hv
    hv.extension('bokeh')
except ImportError:
    hv = None
    print("⚠️ HoloViews not available. Install with: pip install holoviews")


class SURGEVisualizerApp:
    """Main SURGE visualizer application."""
    
    def __init__(self):
        # Initialize controls
        self.dataset_controls = DatasetControls()
        self.model_controls = ModelControls()
        self.compare_controls = CompareControls()
        
        # Initialize views
        self.data_view = DataView(self.dataset_controls)
        self.model_view = ModelView(self.model_controls)
        self.inference_view = None
        
        # Current state
        self.df = None
        self.model_handle = None
        
        # Wire up callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Set up event callbacks."""
        # NOTE: Don't wire load button here - Panel recreates widgets
        # We'll wire it in panel() method after widgets are created
        # This ensures the callback works even when panel is rebuilt
        
        print("DEBUG: _setup_callbacks - load button will be wired in panel()")
        
        # Model training
        self.model_controls.train_button.on_click(
            lambda event: self._on_train_model(event)
        )
        
        # Model loading
        self.model_controls.load_button.on_click(
            lambda event: self._on_load_model(event)
        )
        
        # Inference
        # Will be set up when model is loaded
    
    def _handle_load_button_click(self, event):
        """Handle load button click - primary handler."""
        print("=" * 70)
        print("🔵 LOAD BUTTON CLICKED!")
        print("=" * 70)
        if hasattr(event, 'new'):
            print(f"DEBUG: Button clicks: {event.old if hasattr(event, 'old') else '?'} -> {event.new}")
        
        # Check if file was selected
        file_value = self.dataset_controls.file_input.value
        if file_value is None:
            print("❌ DEBUG: No file selected!")
            try:
                pn.state.notifications.warning("Please select a file first.")
            except:
                pass
            return
        
        print(f"✅ DEBUG: File found! Type: {type(file_value)}")
        if isinstance(file_value, bytes):
            print(f"✅ DEBUG: File bytes length: {len(file_value)}")
        
        # Create mock event for _on_dataset_load
        class MockEvent:
            def __init__(self, value):
                self.new = value
        
        mock_event = MockEvent(file_value)
        try:
            self._on_dataset_load(mock_event)
        except Exception as e:
            print(f"❌ ERROR in _on_dataset_load: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_load_button_click_alt(self):
        """Alternative handler via on_click."""
        print("🔵 LOAD BUTTON CLICKED (via on_click backup)!")
        # Create a simple event object
        class SimpleEvent:
            def __init__(self):
                self.new = None
                self.old = None
        
        self._handle_load_button_click(SimpleEvent())
    
    def _on_load_button_clicked(self, event):
        """Handle load button click - user clicked 'Load Dataset' button."""
        print("=" * 70)
        print("🔵 LOAD BUTTON CLICKED!")
        print("=" * 70)
        print(f"DEBUG: Event type: {type(event)}")
        print(f"DEBUG: Event value: {event}")
        print(f"DEBUG: File input exists: {hasattr(self.dataset_controls, 'file_input')}")
        
        # Check if file was selected
        try:
            file_value = self.dataset_controls.file_input.value
            print(f"DEBUG: File input value type: {type(file_value)}")
            print(f"DEBUG: File input value is None: {file_value is None}")
        except Exception as e:
            print(f"DEBUG: Error accessing file_input.value: {e}")
            try:
                pn.state.notifications.error(f"Error accessing file input: {str(e)}")
            except:
                pass
            return
        
        if file_value is None:
            print("❌ DEBUG: No file selected!")
            try:
                pn.state.notifications.warning("Please select a file first using the 'Choose Dataset File' button above.")
            except Exception as e:
                print(f"Could not show notification: {e}")
            return
        
        print(f"✅ DEBUG: File value found! Type: {type(file_value)}")
        if isinstance(file_value, bytes):
            print(f"✅ DEBUG: File bytes length: {len(file_value)}")
        elif isinstance(file_value, str):
            print(f"✅ DEBUG: File string length: {len(file_value)}")
        else:
            print(f"✅ DEBUG: File value: {file_value}")
        
        # Create a mock event object for _on_dataset_load
        class MockEvent:
            def __init__(self, value):
                self.new = value
        
        mock_event = MockEvent(file_value)
        print("DEBUG: Calling _on_dataset_load...")
        try:
            self._on_dataset_load(mock_event)
        except Exception as e:
            print(f"❌ ERROR in _on_dataset_load: {e}")
            import traceback
            traceback.print_exc()
            try:
                pn.state.notifications.error(f"Error loading dataset: {str(e)}")
            except:
                pass
    
    def _on_dataset_load(self, event):
        """Handle dataset loading."""
        if event.new is None:
            print("DEBUG: File input event.new is None")
            pn.state.notifications.error("No file data found. Please select a file again.")
            return
        
        print(f"DEBUG: Starting dataset load...")
        print(f"DEBUG: File bytes type: {type(event.new)}")
        print(f"DEBUG: File bytes length: {len(event.new) if isinstance(event.new, (bytes, str)) else 'unknown'}")
        
        # Update status immediately - show loading
        try:
            self.data_view.status_pane.object = "⏳ Loading dataset... Please wait..."
            self.data_view.status_pane.css_classes = ['alert', 'alert-info']
            print("DEBUG: Status pane updated to loading state")
        except Exception as e:
            print(f"DEBUG: Could not update status pane: {e}")
        
        # Disable load button during loading
        try:
            self.dataset_controls.load_button.disabled = True
            self.dataset_controls.load_button.name = "⏳ Loading..."
        except:
            pass
        
        try:
            # For FileInput, we get bytes
            import tempfile
            import pickle
            
            # Get file bytes
            file_bytes = event.new
            if file_bytes is None:
                print("DEBUG: file_bytes is None")
                return
            
            print(f"DEBUG: Got file bytes, type: {type(file_bytes)}, length: {len(file_bytes) if isinstance(file_bytes, (bytes, str)) else 'unknown'}")
            
            # Write to temp file
            import os
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp:
                    if isinstance(file_bytes, bytes):
                        tmp.write(file_bytes)
                    else:
                        tmp.write(file_bytes.read() if hasattr(file_bytes, 'read') else bytes(file_bytes))
                    tmp_path = tmp.name
                
                print(f"DEBUG: Wrote temp file to {tmp_path}")
                
                # Load dataset
                from surge_viz.data_ops import load_dataset
                self.df = load_dataset(tmp_path)
                
                print(f"DEBUG: Loaded dataset, shape: {self.df.shape if self.df is not None else 'None'}")
                
                if self.df is None or self.df.empty:
                    print("DEBUG: Dataset is None or empty")
                    pn.state.notifications.warning("Dataset loaded but appears to be empty.")
                    return
                
                # Update data view (this also updates status)
                print("DEBUG: Updating data view")
                self.data_view.load_data(tmp_path)
                
                # Force update of selectors by directly setting their options
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                print(f"DEBUG: Found {len(numeric_cols)} numeric columns: {numeric_cols[:5]}...")
                
                # Update the selectors directly (ensure they're updated)
                if hasattr(self.dataset_controls, 'param_selector'):
                    self.dataset_controls.param_selector.options = numeric_cols
                    print(f"DEBUG: Updated param_selector with {len(numeric_cols)} options")
                if hasattr(self.dataset_controls, 'output_selector'):
                    self.dataset_controls.output_selector.options = numeric_cols
                    print(f"DEBUG: Updated output_selector with {len(numeric_cols)} options")
                
                # Update the param object as well
                self.dataset_controls.numeric_columns = numeric_cols
                
                # Force trigger parameter update
                try:
                    self.dataset_controls.param.trigger('numeric_columns')
                except:
                    pass
                
                # Re-enable load button with success message
                try:
                    self.dataset_controls.load_button.disabled = False
                    self.dataset_controls.load_button.name = "✅ Dataset Loaded! (Click to reload)"
                    self.dataset_controls.load_button.button_type = "success"
                except:
                    pass
                
                pn.state.notifications.success(f"Dataset loaded! {self.df.shape[0]} rows, {len(numeric_cols)} numeric columns")
                print("=" * 60)
                print(f"✅ SUCCESS: Dataset loaded! Shape: {self.df.shape}")
                print(f"✅ Numeric columns: {len(numeric_cols)}")
                print("=" * 60)
                
                # Update inference controls if we have parameter info
                if self.df is not None and len(self.dataset_controls.param_columns) > 0:
                    param_info = prepare_inference_inputs(
                        self.df,
                        self.dataset_controls.param_columns
                    )
                    inference_controls = InferenceControls(param_info=param_info)
                    if self.inference_view is not None:
                        self.inference_view.inference_controls = inference_controls
                        self.inference_view.df = self.df
                    else:
                        self.inference_view = InferenceView(
                            inference_controls,
                            self.compare_controls,
                            model_handle=self.model_handle,
                            df=self.df
                        )
                    # Wire up inference button
                    inference_controls.inference_button.on_click(
                        lambda e: self.inference_view.run_inference()
                    )
                
            finally:
                # Clean up temp file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
        except Exception as e:
            import traceback
            error_msg = f"Error loading dataset: {str(e)}"
            print("=" * 60)
            print(f"❌ ERROR: Dataset load error: {error_msg}")
            print(traceback.format_exc())
            print("=" * 60)
            
            # Update status to show error
            try:
                self.data_view.status_pane.object = f"❌ Error loading dataset: {str(e)}"
                self.data_view.status_pane.css_classes = ['alert', 'alert-danger']
            except:
                pass
            
            # Re-enable load button
            try:
                self.dataset_controls.load_button.disabled = False
                self.dataset_controls.load_button.name = "❌ Load Failed - Try Again"
                self.dataset_controls.load_button.button_type = "danger"
            except:
                pass
            
            try:
                pn.state.notifications.error(error_msg)
            except:
                print(f"Could not show notification: {error_msg}")
    
    def _on_train_model(self, event):
        """Handle model training."""
        if self.df is None:
            pn.state.notifications.error("Please load a dataset first")
            return
        
        try:
            # Get input/output columns
            if not self.dataset_controls.param_columns:
                pn.state.notifications.error("Please select parameter columns")
                return
            if not self.dataset_controls.output_columns:
                pn.state.notifications.error("Please select output columns")
                return
            
            # Prepare data
            X = self.df[self.dataset_controls.param_columns]
            y = self.df[self.dataset_controls.output_columns]
            
            # Train model
            from surge_viz.surge_api import train_model
            
            model_id = train_model(
                X, y,
                model_type=self.model_controls.model_type,
                test_split=0.2,
            )
            
            # Load model
            from surge_viz.surge_api import load_model
            self.model_handle = load_model(model_id)
            
            # Update views
            self.model_view.model_handle = self.model_handle
            self.model_view.status_pane.object = f"Model trained and loaded: {model_id}"
            
            if self.inference_view:
                self.inference_view.model_handle = self.model_handle
            
            pn.state.notifications.success(f"Model trained successfully: {model_id}")
            
        except Exception as e:
            pn.state.notifications.error(f"Error training model: {str(e)}")
    
    def _on_load_model(self, event):
        """Handle model loading."""
        model_id = self.model_controls.model_id_input.value
        if not model_id:
            pn.state.notifications.error("Please enter a model ID")
            return
        
        try:
            from surge_viz.surge_api import load_model
            self.model_handle = load_model(model_id)
            
            # Update views
            self.model_view.model_handle = self.model_handle
            self.model_view.status_pane.object = f"Model loaded: {model_id}"
            
            if self.inference_view:
                self.inference_view.model_handle = self.model_handle
            
            pn.state.notifications.success(f"Model loaded: {model_id}")
            
        except Exception as e:
            pn.state.notifications.error(f"Error loading model: {str(e)}")
    
    def panel(self) -> pn.Tabs:
        """Create main application panel."""
        # Build inference view if not already created
        if self.inference_view is None:
            try:
                param_info = {}
                if self.df is not None and len(self.dataset_controls.param_columns) > 0:
                    try:
                        param_info = prepare_inference_inputs(
                            self.df,
                            self.dataset_controls.param_columns
                        )
                    except Exception as e:
                        print(f"Warning: Could not prepare inference inputs: {e}")
                        param_info = {}
                
                inference_controls = InferenceControls(param_info=param_info)
                self.inference_view = InferenceView(
                    inference_controls,
                    self.compare_controls,
                    model_handle=self.model_handle,
                    df=self.df
                )
                # Wire up inference button
                inference_controls.inference_button.on_click(
                    lambda e: self.inference_view.run_inference()
                )
            except Exception as e:
                print(f"Error creating inference view: {e}")
                # Create a simple placeholder inference view
                from surge_viz.components.controls import InferenceControls as IC, CompareControls as CC
                placeholder_controls = IC(param_info={})
                placeholder_compare = CC()
                self.inference_view = InferenceView(
                    placeholder_controls,
                    placeholder_compare,
                    model_handle=None,
                    df=None
                )
        
        # Safely get panels with error handling
        try:
            data_panel = self.data_view.panel()
            
            # CRITICAL: Re-wire the load button callback AFTER panel is created
            # Panel recreates widgets when panel() is called, so we MUST wire here
            try:
                button = self.dataset_controls.load_button
                
                # Clear any existing watchers first to avoid duplicates
                if hasattr(button.param, '_callbacks') and 'clicks' in button.param._callbacks:
                    button.param._callbacks['clicks'] = []
                
                # Wire param.watch - this is the most reliable method
                button.param.watch(
                    self._handle_load_button_click,
                    'clicks',
                    onlychanged=False
                )
                
                # Also wire on_click as backup
                button.on_click(lambda e: self._handle_load_button_click_alt())
                
                print("✅ DEBUG: Load button wired successfully in panel()")
            except Exception as e:
                print(f"❌ DEBUG: Could not wire load button: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            data_panel = pn.pane.Markdown(f"Error loading data view: {str(e)}")
        
        try:
            model_panel = self.model_view.panel()
        except Exception as e:
            model_panel = pn.pane.Markdown(f"Error loading model view: {str(e)}")
        
        try:
            inference_panel = self.inference_view.panel()
        except Exception as e:
            inference_panel = pn.pane.Markdown(f"Error loading inference view: {str(e)}")
        
        tabs = pn.Tabs(
            ("📊 Data", data_panel),
            ("🤖 Model", model_panel),
            ("🔮 Inference & Compare", inference_panel),
            sizing_mode='stretch_width',
            max_width=1200
        )
        
        # Wrap in a container with max width to prevent horizontal scrolling
        container = pn.Column(
            pn.pane.Markdown("# 🚀 SURGE Visualization Platform", sizing_mode='stretch_width'),
            pn.pane.Markdown("Interactive web interface for training and visualizing SURGE surrogate models.", sizing_mode='stretch_width'),
            tabs,
            sizing_mode='stretch_width',
            max_width=1200,
            margin=10,
            css_classes=['surge-app-container']
        )
        
        return container


# Create app instance and panel at module level
try:
    print("=" * 70)
    print("DEBUG: Creating SURGEVisualizerApp...")
    app = SURGEVisualizerApp()
    print("DEBUG: App created, creating panel...")
    main_panel = app.panel()
    print(f"DEBUG: Panel created: {type(main_panel)}")
    
    # Mark as servable - this makes it available when Panel serves the file
    main_panel.servable(title="SURGE Visualizer")
    print("DEBUG: Panel marked as servable")
    print("=" * 70)
except ImportError as e:
    # Missing dependencies - show helpful error message
    missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
    error_msg = f"""
    # ⚠️ SURGE Visualizer - Missing Dependencies
    
    The application cannot start because **{missing_module}** is not installed.
    
    ## Installation Instructions:
    
    ### Option 1: Using Conda (Recommended)
    ```bash
    conda env create -f surge_viz/env.yml
    conda activate surge-viz
    ```
    
    ### Option 2: Using pip
    ```bash
    pip install panel>=1.5.0 holoviews>=1.19.0 hvplot bokeh pandas numpy scipy scikit-learn
    ```
    
    ### Option 3: Install missing module only
    ```bash
    pip install {missing_module}
    ```
    
    After installing, restart the Panel server:
    ```bash
    panel serve surge_viz/app.py --dev --autoreload --port 5007
    ```
    
    **Error Details:**
    ```
    {str(e)}
    ```
    """
    error_panel = pn.pane.Markdown(error_msg, sizing_mode='stretch_width', max_width=1000)
    error_panel.servable(title="SURGE Visualizer - Missing Dependencies")
    main_panel = error_panel
except Exception as e:
    # Other errors - show full traceback
    import traceback
    error_msg = f"""
    # ❌ SURGE Visualizer - Error
    
    An error occurred while initializing the application:
    
    **Error:**
    ```
    {str(e)}
    ```
    
    **Full Traceback:**
    ```
    {traceback.format_exc()}
    ```
    
    Please check the terminal for more details or report this issue.
    """
    error_panel = pn.pane.Markdown(error_msg, sizing_mode='stretch_width', max_width=1000)
    error_panel.servable(title="SURGE Visualizer - Error")
    main_panel = error_panel


# For script execution
def serve(port: int = 5007, dev: bool = True):
    """
    Serve the SURGE visualizer application.
    
    Parameters
    ----------
    port : int, default=5007
        Port to serve on.
    dev : bool, default=True
        Whether to run in development mode with autoreload.
    """
    pn.serve(
        main_panel,
        port=port,
        dev=dev,
        autoreload=dev,
        show=dev,
    )

