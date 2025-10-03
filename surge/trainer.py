import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from .metrics import mean_squared_error, r2_score

# Optional imports for advanced functionality
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import optuna
    from optuna.integration import BoTorchSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import gpflow as gpf
    GPFLOW_AVAILABLE = True
except ImportError:
    GPFLOW_AVAILABLE = False


class SurrogateTrainer:
    """
    SURGE Surrogate Trainer for comprehensive surrogate modeling workflows.
    
    This class provides the main interface for SURGE surrogate modeling,
    integrating data loading, preprocessing, model training, and evaluation
    with automated analysis and visualization capabilities.
    """

    def __init__(self, dir_path=None):
        """
        Initialize the SurrogateTrainer.
        
        Parameters:
        -----------
        dir_path : str, optional
            Directory path for saving outputs
        """
        print("🚀 Initializing SURGE SurrogateTrainer")
        self.dir_path = dir_path or ''
        self.df = None
        self.input_variables = None
        self.output_variables = None
        self.trainer = None

    def load_dataset_pickle(self, pickle_path):
        """
        Load dataset from pickle file and perform comprehensive analysis.
        
        Parameters:
        -----------
        pickle_path : str or Path
            Path to the pickle file containing the dataset
            
        Returns:
        --------
        tuple : (input_variables, output_variables)
            Lists of identified input and output variable names
        """
        import pandas as pd

        from .preprocessing import (
            analyze_dataset_structure,
            get_dataset_statistics,
            print_dataset_analysis,
        )

        print("📥 Loading dataset from pickle file...")

        # Load data
        pickle_path = Path(pickle_path)
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        self.df = pd.read_pickle(pickle_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")

        # Perform comprehensive dataset analysis
        print("\n🔍 Performing dataset structure analysis...")
        dataset_info = analyze_dataset_structure(self.df)

        print("\n📊 Dataset Analysis Results:")
        print_dataset_analysis(dataset_info)

        print("\n📈 Dataset Statistics:")
        stats = get_dataset_statistics(self.df)

        # Identify input and output variables
        print("\n🎯 Identifying input and output variables...")

        # Find output variables (PwE_ prefix)
        output_variables = [col for col in self.df.columns if col.startswith('PwE_')]

        # Find input variables (everything else that's numeric and not an output)
        input_variables = []
        for col in self.df.columns:
            if col not in output_variables:
                # Check if column is numeric and has reasonable variation
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    if self.df[col].nunique() > 1:  # Has variation
                        input_variables.append(col)

        # Store variables
        self.input_variables = input_variables
        self.output_variables = output_variables

        print(f"📥 Input variables ({len(input_variables)}): {input_variables}")
        print(f"📤 Output variables ({len(output_variables)}): {len(output_variables)} targets")
        print(f"   First 5 outputs: {output_variables[:5]}")

        # Initialize MLTrainer with the identified variables
        self.trainer = MLTrainer(len(input_variables), len(output_variables))

        return input_variables, output_variables

    def get_trainer(self):
        """
        Get the underlying MLTrainer instance for advanced operations.
        
        Returns:
        --------
        MLTrainer : The MLTrainer instance
        """
        if self.trainer is None:
            raise ValueError("No trainer available. Call load_dataset_pickle() first.")
        return self.trainer


class MLTrainer:
    """
    Comprehensive ML trainer for SURGE surrogate modeling.
    
    This class provides a unified interface for training various ML models
    including Random Forest, Neural Networks, and Gaussian Processes with
    automated hyperparameter optimization and cross-validation.
    """

    def __init__(self, n_features=None, n_outputs=None, dir_path=None):
        """
        Initialize the MLTrainer.
        
        Parameters:
        -----------
        n_features : int, optional
            Number of input features (can be set later via load_data)
        n_outputs : int, optional
            Number of output targets (can be set later via load_data)
        dir_path : str, optional
            Directory path for saving outputs
        """
        print("🚀 Initializing SURGE MLTrainer")
        self.F = n_features
        self.T = n_outputs
        self.models = []
        self.model_types = []
        self.dir_path = dir_path or ''

        # Model performance tracking
        self.model_performance = []
        
        # Baseline performance tracking
        self.baseline_r2 = None

        # Model type mapping
        self.MODEL_TYPES = {
            0: 'RandomForestRegressor',
            1: 'MLPRegressor',
            2: 'PyTorchMLP',
            3: 'GPRModel'
        }

        if n_features is not None and n_outputs is not None:
            print(f"📊 Configured for {n_features} features → {n_outputs} outputs")
        else:
            print("📊 Features and outputs will be set when loading data")

    # Define PyTorch MLP model
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.0, activation_fn=nn.ReLU):
            super().__init__()
            layers = []
            for i in range(len(hidden_layers)):
                layers.append(nn.Linear(input_size if i == 0 else hidden_layers[i-1], hidden_layers[i]))
                layers.append(activation_fn())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_layers[-1], output_size))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

        def predict(self, x):
            """Custom predict method compatible with sklearn interface"""
            self.eval()
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                prediction = self.forward(x)
            return prediction.cpu().numpy()

    def load_df_dataset(self, df, input_feature_names, output_feature_names):
        """
        Load dataset from pandas DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing features and targets
        input_feature_names : list
            List of input feature column names
        output_feature_names : list
            List of output target column names
        """
        print("📥 Loading dataset...")

        self.df = df
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names

        # Determine main output prefix
        common_prefix = os.path.commonprefix(output_feature_names)
        print(f"📋 Common output prefix: {common_prefix}")
        self.main_output = common_prefix

        # Extract features and targets
        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]

        # Set F and T if not provided in constructor
        if self.F is None:
            self.F = len(input_feature_names)
        if self.T is None:
            self.T = len(output_feature_names)

        print(f"✅ Dataset loaded: {self.X.shape[0]} samples")
        print(f"   📥 Input features: {len(input_feature_names)} (F={self.F})")
        print(f"   📤 Output targets: {len(output_feature_names)} (T={self.T})")

    def train_test_split(self, test_split=0.2):
        """
        Split data into train+validation and test sets.
        
        Parameters:
        -----------
        test_split : float
            Fraction of data to use for testing (default: 0.2)
        """
        print(f"\n🔄 Splitting into test and training data (test_split={test_split})")

        # Split into train+val and test
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            self.X, self.y, test_size=test_split, random_state=42, shuffle=True
        )

        print(f"X train + val shape: {self.X_train_val.shape}")
        print(f"y train + val shape: {self.y_train_val.shape}")
        print(f"X test shape: {self.X_test.shape}")
        print(f"y test shape: {self.y_test.shape}")

    def standardize_data(self):
        """Standardize the dataset using training set statistics."""
        print("\n🔧 Standardizing data")

        # Standardize inputs
        self.x_scaler = SKStandardScaler()
        self.x_train_val_sc = self.x_scaler.fit_transform(self.X_train_val)
        self.x_test_sc = self.x_scaler.transform(self.X_test)

        # Standardize outputs
        self.y_scaler = SKStandardScaler()
        self.y_train_val_sc = self.y_scaler.fit_transform(self.y_train_val)
        self.y_test_sc = self.y_scaler.transform(self.y_test)

        print("✅ x_train_sc, x_test_sc, y_train_sc, y_test_sc generated")
        print("\n--- Statistics for x_scaler [all features] ---")
        print("means: ", self.x_scaler.mean_)
        if self.x_scaler.var_ is not None:
            print("standard deviation: ", np.sqrt(self.x_scaler.var_))

        # Show statistics for most varying output
        if self.y_scaler.mean_ is not None and self.y_scaler.var_ is not None:
            idx_max_ytrain_mean = np.argmax(self.y_scaler.mean_)
            print("\n--- Statistics for y_scaler [on axis, max_value] ---")
            print("means: ", self.y_scaler.mean_[[0, idx_max_ytrain_mean]])
            print("standard deviation: ", np.sqrt(self.y_scaler.var_[[0, idx_max_ytrain_mean]]))

    def init_model(self, model_type):
        """
        Initialize a model of the specified type.
        
        Parameters:
        -----------
        model_type : int
            Model type identifier:
            0: Random Forest Regressor
            1: MLP Regressor (sklearn)
            2: PyTorch MLP
            3: GPR Model
        """
        print(f"\n🎯 Initializing Model {model_type}: {self.MODEL_TYPES.get(model_type, 'Unknown')}")

        if model_type == 0:
            self._init_random_forest()
        elif model_type == 1:
            self._init_mlp_sklearn()
        elif model_type == 2:
            self._init_mlp_pytorch()
        elif model_type == 3:
            self._init_gpr()
        else:
            print("⚠️ Unknown model type, defaulting to Random Forest")
            self._init_random_forest()

    def _init_random_forest(self):
        """Initialize Random Forest Regressor"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        self.models.append(model)
        self.model_types.append(0)

        # Initialize performance tracking for this model
        performance = {
            'model_type': 'RandomForestRegressor',
            'R2_train_val': None,
            'R2': None,
            'MSE_train_val': None,
            'MSE': None,
            'training_time': None,
            'average_inference_time_train': None,
            'average_inference_time_test': None
        }
        self.model_performance.append(performance)
        print("✅ Random Forest Regressor initialized")

    def _init_mlp_sklearn(self):
        """Initialize sklearn MLP Regressor"""
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
        self.models.append(model)
        self.model_types.append(1)

        # Initialize performance tracking for this model
        performance = {
            'model_type': 'MLPRegressor',
            'R2_train_val': None,
            'R2': None,
            'MSE_train_val': None,
            'MSE': None,
            'training_time': None,
            'average_inference_time_train': None,
            'average_inference_time_test': None
        }
        self.model_performance.append(performance)
        print("✅ MLP Regressor (sklearn) initialized")

    def _init_mlp_pytorch(self):
        """Initialize PyTorch MLP"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")

        model = self.MLP(
            input_size=self.F,
            hidden_layers=[128, 64, 32],
            output_size=self.T,
            dropout_rate=0.1
        )
        self.models.append(model)
        self.model_types.append(2)

        # Initialize performance tracking for this model
        performance = {
            'model_type': 'PyTorchMLP',
            'R2_train_val': None,
            'R2': None,
            'MSE_train_val': None,
            'MSE': None,
            'training_time': None,
            'average_inference_time_train': None,
            'average_inference_time_test': None
        }
        self.model_performance.append(performance)
        print("✅ PyTorch MLP initialized")

    def _init_gpr(self):
        """Initialize Gaussian Process Regressor"""
        if not GPFLOW_AVAILABLE:
            raise ImportError("GPflow not available. Please install gpflow.")

        print("🔄 GPR initialization - to be implemented")
        # GPR implementation would go here
        self.models.append(None)  # Placeholder
        self.model_types.append(3)

        # Initialize performance tracking for this model
        performance = {
            'model_type': 'GPR',
            'R2_train_val': None,
            'R2': None,
            'MSE_train_val': None,
            'MSE': None,
            'training_time': None,
            'average_inference_time_train': None,
            'average_inference_time_test': None
        }
        self.model_performance.append(performance)

    def train(self, model_index):
        """
        Train the specified model.
        
        Parameters:
        -----------
        model_index : int
            Index of the model to train
        """
        if model_index >= len(self.models):
            raise ValueError(f"Model index {model_index} not available")

        print(f'\n🎯 Training Model {model_index}: {self.MODEL_TYPES[self.model_types[model_index]]}')

        start_time = time.time()

        if self.model_types[model_index] in [0, 1]:  # sklearn models
            self.models[model_index].fit(self.x_train_val_sc, self.y_train_val_sc)
        elif self.model_types[model_index] == 2:  # PyTorch MLP
            self._train_pytorch_mlp(model_index)
        elif self.model_types[model_index] == 3:  # GPR
            self._train_gpr(model_index)

        training_time = time.time() - start_time

        # Store training time in model performance
        self.model_performance[model_index]['training_time'] = training_time

        print(f"⏱️ Elapsed time: {training_time:.2f} seconds")

    def _train_pytorch_mlp(self, model_index):
        """Train PyTorch MLP with proper training loop"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        model = self.models[model_index]

        # Convert to tensors
        X_train_tensor = torch.tensor(self.x_train_val_sc, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train_val_sc, dtype=torch.float32)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        n_epochs = 100
        batch_size = 32

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.6f}")

    def _train_gpr(self, model_index):
        """Train Gaussian Process Regressor"""
        print("🔄 GPR training - to be implemented")

    def predict_output(self, model_index):
        """
        Generate predictions and calculate performance metrics.
        
        Parameters:
        -----------
        model_index : int
            Index of the model to use for prediction
        """
        print(f'\n🎯 Predicting outputs - Model {model_index}')

        model = self.models[model_index]

        # Training set predictions
        print('\n--- Training Set Results ---')
        start_time = time.time()
        y_pred_train_val_sc = model.predict(self.x_train_val_sc)

        # Ensure 2D shape for sklearn inverse_transform (required for scalers)
        if y_pred_train_val_sc.ndim == 1:
            y_pred_train_val_sc = y_pred_train_val_sc.reshape(-1, 1)

        self.y_pred_train_val = self.y_scaler.inverse_transform(y_pred_train_val_sc)
        t_pred_train_val = time.time() - start_time
        avg_inference_time_train = t_pred_train_val / self.x_train_val_sc.shape[0]
        print(f' t_I(avg) = {avg_inference_time_train:.6f} seconds per sample')

        # Calculate training metrics
        self.MSE_train_val = mean_squared_error(self.y_train_val, self.y_pred_train_val)
        self.R2_train_val = r2_score(self.y_train_val, self.y_pred_train_val)

        print(f' MSE_train_val = {self.MSE_train_val:.6f}')
        print(f' R2_train_val = {self.R2_train_val:.4f}')

        # Test set predictions
        print('\n--- Testing Set Results ---')
        start_time = time.time()
        y_pred_test_sc = model.predict(self.x_test_sc)

        # Ensure 2D shape for sklearn inverse_transform (required for scalers)
        if y_pred_test_sc.ndim == 1:
            y_pred_test_sc = y_pred_test_sc.reshape(-1, 1)

        self.y_pred_test = self.y_scaler.inverse_transform(y_pred_test_sc)
        t_pred_test = time.time() - start_time
        avg_inference_time_test = t_pred_test / self.x_test_sc.shape[0]
        print(f' t_I(avg) = {avg_inference_time_test:.6f} seconds per sample')

        # Calculate test metrics
        self.MSE = mean_squared_error(self.y_test, self.y_pred_test)
        self.R2 = r2_score(self.y_test, self.y_pred_test)

        print(f' MSE = {self.MSE:.6f}')
        print(f' R2 = {self.R2:.4f}')

        # Store all performance metrics
        self.model_performance[model_index].update({
            'R2_train_val': self.R2_train_val,
            'R2': self.R2,
            'MSE_train_val': self.MSE_train_val,
            'MSE': self.MSE,
            'average_inference_time_train': avg_inference_time_train,
            'average_inference_time_test': avg_inference_time_test
        })

    def optimize_solver(self, n_iter=50, model_idx=0, search_spaces=None):
        """
        Optimize hyperparameters using Bayesian Search.
        
        Parameters:
        -----------
        n_iter : int
            Number of optimization iterations
        model_idx : int
            Index of model to optimize
        search_spaces : dict, optional
            Custom search space for hyperparameters
        """
        if not SKOPT_AVAILABLE:
            print("⚠️ scikit-optimize not available. Please install scikit-optimize.")
            return

        model_type = self.model_types[model_idx]

        if model_type == 0:  # Random Forest
            print('🔍 Optimizing Random Forest Regressor Model')
            print('- Method: Bayesian Search Cross Validated')

            model = self.models[model_idx]

            if search_spaces is None:
                search_spaces = {
                    'n_estimators': (50, 500),
                    'max_depth': (5, 50),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': [1.0, 'log2', 'sqrt'],
                }

            opt = BayesSearchCV(
                estimator=model,
                search_spaces=search_spaces,
                n_iter=n_iter,
                scoring='neg_mean_squared_error',
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=2
            )

            print(f'🚀 Running optimization for n_iter = {n_iter}')
            opt.fit(self.x_train_val_sc, self.y_train_val_sc)

            print("🎯 Best hyperparameters:", opt.best_params_)  # type: ignore
            self.models[model_idx] = opt.best_estimator_  # type: ignore

            # Generate predictions with optimized model
            self.predict_output(model_idx)

        else:
            print(f'⚠️ Optimization not implemented for model type {model_type}')

    def cross_validate(self, model_idx=0, cv_folds=5):
        """
        Perform k-fold cross-validation with metrics computed on original scale.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to cross-validate
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Cross-validation results with metrics on original scale
        """
        import time

        from sklearn.base import clone
        from sklearn.metrics import mean_squared_error, r2_score

        print(f'\n🔄 Performing {cv_folds}-fold cross-validation')
        print('   Computing MSE and R² on original (unstandardized) scale...')

        base_model = self.models[model_idx]

        # Initialize KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Storage for results
        fold_mse = []
        fold_r2 = []
        fold_train_time = []
        fold_inference_time = []

        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.x_train_val_sc)):
            print(f'   Fold {fold_idx + 1}/{cv_folds}...', end=' ')

            # Split data for this fold
            X_fold_train = self.x_train_val_sc[train_idx]
            y_fold_train = self.y_train_val_sc[train_idx]
            X_fold_val = self.x_train_val_sc[val_idx]
            y_fold_val = self.y_train_val_sc[val_idx]

            # Clone model for this fold
            fold_model = clone(base_model)

            # Train model and measure training time
            start_time = time.time()
            fold_model.fit(X_fold_train, y_fold_train)
            train_time = time.time() - start_time

            # Make predictions and measure inference time
            start_time = time.time()
            y_pred_sc = fold_model.predict(X_fold_val)
            inference_time = (time.time() - start_time) / len(val_idx)  # Per sample

            # Ensure 2D shapes for sklearn inverse_transform
            if y_pred_sc.ndim == 1:
                y_pred_sc = y_pred_sc.reshape(-1, 1)
            if y_fold_val.ndim == 1:
                y_fold_val = y_fold_val.reshape(-1, 1)

            # Rescale predictions and targets back to original scale
            y_pred_orig = self.y_scaler.inverse_transform(y_pred_sc)
            y_true_orig = self.y_scaler.inverse_transform(y_fold_val)

            # Compute metrics on original scale
            mse = mean_squared_error(y_true_orig, y_pred_orig)
            r2 = r2_score(y_true_orig, y_pred_orig)

            # Store results
            fold_mse.append(mse)
            fold_r2.append(r2)
            fold_train_time.append(train_time)
            fold_inference_time.append(inference_time)

            print(f'MSE={mse:.6f}, R²={r2:.4f}')

        # Convert to numpy arrays
        fold_mse = np.array(fold_mse)
        fold_r2 = np.array(fold_r2)
        fold_train_time = np.array(fold_train_time)
        fold_inference_time = np.array(fold_inference_time)

        # Print summary statistics
        print('\n📊 Cross-validation results (original scale):')
        print(f'   MSE:  {fold_mse.mean():.6f} ± {fold_mse.std():.6f}')
        print(f'   R²:   {fold_r2.mean():.4f} ± {fold_r2.std():.4f}')
        print(f'   Training time:   {fold_train_time.mean():.3f} ± {fold_train_time.std():.3f} sec/fold')
        print(f'   Inference time:  {fold_inference_time.mean()*1000:.2f} ± {fold_inference_time.std()*1000:.2f} ms/sample')

        # Create results dictionary
        results = {
            'mse_folds': fold_mse,
            'r2_folds': fold_r2,
            'train_time_folds': fold_train_time,
            'inference_time_folds': fold_inference_time,
            'mse_mean': fold_mse.mean(),
            'mse_std': fold_mse.std(),
            'r2_mean': fold_r2.mean(),
            'r2_std': fold_r2.std(),
            'train_time_mean': fold_train_time.mean(),
            'train_time_std': fold_train_time.std(),
            'inference_time_mean': fold_inference_time.mean(),
            'inference_time_std': fold_inference_time.std()
        }

        return results

    def save_prediction(self, output_fmt='pickle', output_dir=None, model_idx=0):
        """
        Save predictions to file.
        
        Parameters:
        -----------
        output_fmt : str
            Output format ('pickle')
        output_dir : str
            Output directory path
        model_idx : int
            Model index to save predictions for
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_type_str = {
            0: 'RFR', 1: 'MLP', 2: 'PyTorchMLP', 3: 'GPR'
        }.get(self.model_types[model_idx], 'Model')

        if output_fmt == 'pickle':
            # Create prediction dataframe
            columns_out = [f'{model_type_str}_{col}' for col in self.output_feature_names]

            # Initialize prediction array
            y_pred_full = np.zeros(self.y.shape)

            # Create dataframe with predictions
            df_y_pred = pd.DataFrame(data=y_pred_full, columns=columns_out, dtype='float64')
            df_complete = pd.concat([self.df, df_y_pred], axis=1)

            # Fill in predictions
            df_complete.loc[self.X_train_val.index, columns_out] = self.y_pred_train_val
            df_complete.loc[self.X_test.index, columns_out] = self.y_pred_test

            # Determine output path
            if self.dir_path:
                if output_dir:
                    path_out_folder = os.path.join(self.dir_path, output_dir)
                else:
                    path_out_folder = os.path.join(self.dir_path, 'output_prediction')
            else:
                if output_dir:
                    path_out_folder = output_dir
                else:
                    path_out_folder = os.path.join(os.getenv('HOME', '.'), 'Downloads', 'Model_prediction')

        if not os.path.exists(path_out_folder):
            os.makedirs(path_out_folder)

        path_out_file = os.path.join(path_out_folder, f'{self.main_output}{model_type_str}.pkl')
        print(f"💾 Saving prediction to: {path_out_file}")

        df_complete.to_pickle(path_out_file)
        print('✅ Prediction saved')

    def optimize_with_optuna(self, model_idx=0, n_trials=100, objective_type='mse', sampler_type='default'):
        """
        Optimize hyperparameters using Optuna with different samplers.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to optimize
        n_trials : int
            Number of Optuna trials
        objective_type : str
            Optimization objective ('mse', 'r2')
        sampler_type : str
            Optuna sampler type ('default', 'tpe', 'botorch')
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Please install optuna.")
            return

        import optuna
        print(f"🔍 Optuna imported successfully: {optuna.__version__}")
        model_type = self.model_types[model_idx]
        
        # Initialize sampler based on type
        sampler = None
        if sampler_type == 'tpe':
            try:
                sampler = optuna.samplers.TPESampler(seed=42)
                print(f"🔬 Using TPE Sampler")
            except Exception as e:
                print(f"⚠️ TPE sampler initialization failed: {e}")
                sampler = None
        elif sampler_type == 'botorch':
            try:
                from optuna.integration import BoTorchSampler
                sampler = BoTorchSampler()
                print(f"🔬 Using BoTorch Sampler")
            except ImportError:
                print("⚠️ BoTorch not available. Install with: pip install botorch")
                sampler = None
            except Exception as e:
                print(f"⚠️ BoTorch sampler initialization failed: {e}")
                sampler = None
        else:
            print(f"🔬 Using Default Optuna Sampler")
            sampler = None

        if model_type == 0:  # Random Forest
            print(f'🔍 Optimizing Random Forest with Optuna ({n_trials} trials)')

            def objective_rf(trial):
                # Suggest hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 5, 50)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])

                # Create model with suggested parameters
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )

                # Cross-validation score
                cv_scores = cross_val_score(
                    model, self.x_train_val_sc, self.y_train_val_sc,
                    cv=3, scoring='neg_mean_squared_error', n_jobs=-1
                )

                return -cv_scores.mean()  # Return positive MSE

            # Create study and optimize
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(objective_rf, n_trials=n_trials)

            print("🎯 Best hyperparameters:", study.best_params)
            print(f"🎯 Best MSE: {study.best_value:.6f}")

            # Update model with best parameters
            best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
            self.models[model_idx] = best_model

            # Train with best parameters
            self.train(model_idx)
            self.predict_output(model_idx)

        elif model_type == 2:  # PyTorch MLP
            print(f'🔍 Optimizing PyTorch MLP with Optuna ({n_trials} trials)')

            def objective_mlp(trial):
                # Suggest hyperparameters
                num_layers = trial.suggest_int("num_layers", 1, 5)
                hidden_size = trial.suggest_int("hidden_size", 32, 256)
                dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
                learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
                batch_size = trial.suggest_int("batch_size", 16, 128)
                activation_fn = trial.suggest_categorical("activation_fn", ["ReLU", "Tanh", "LeakyReLU"])

                # Map activation function
                activation_mapping = {
                    "ReLU": nn.ReLU,
                    "Tanh": nn.Tanh,
                    "LeakyReLU": nn.LeakyReLU
                }
                activation_class = activation_mapping[activation_fn]

                # Build model
                hidden_layers = [hidden_size] * num_layers
                model = self.MLP(
                    input_size=self.F,
                    hidden_layers=hidden_layers,
                    output_size=self.T,
                    dropout_rate=dropout_rate,
                    activation_fn=activation_class
                )

                # Setup training
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.MSELoss()

                # Convert data to tensors
                X_tensor = torch.tensor(self.x_train_val_sc, dtype=torch.float32)
                y_tensor = torch.tensor(self.y_train_val_sc, dtype=torch.float32)

                # Create data loader
                dataset = TensorDataset(X_tensor, y_tensor)
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Training loop (reduced epochs for speed)
                model.train()
                for epoch in range(50):  # Reduced for hyperparameter search
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation on validation data (use a portion of training data)
                model.eval()
                with torch.no_grad():
                    predictions = model(X_tensor)
                    mse = criterion(predictions, y_tensor).item()

                return mse

            # Create study and optimize
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(objective_mlp, n_trials=n_trials)

            print("🎯 Best hyperparameters:", study.best_params)
            print(f"🎯 Best MSE: {study.best_value:.6f}")

            # Update model with best parameters
            activation_mapping = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
            activation_class = activation_mapping[study.best_params['activation_fn']]
            hidden_layers = [study.best_params['hidden_size']] * study.best_params['num_layers']

            best_model = self.MLP(
                input_size=self.F,
                hidden_layers=hidden_layers,
                output_size=self.T,
                dropout_rate=study.best_params['dropout_rate'],
                activation_fn=activation_class
            )

            self.models[model_idx] = best_model
            self.best_mlp_params = study.best_params  # Store for final training

            # Train with best parameters
            self._train_pytorch_mlp_with_params(model_idx, study.best_params)
            self.predict_output(model_idx)

        else:
            print(f'⚠️ Optuna optimization not implemented for model type {model_type}')

    def _train_pytorch_mlp_with_params(self, model_index, params):
        """Train PyTorch MLP with specific hyperparameters"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        model = self.models[model_index]

        # Convert to tensors
        X_train_tensor = torch.tensor(self.x_train_val_sc, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train_val_sc, dtype=torch.float32)

        # Setup training with optimized parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))

        # Training parameters
        n_epochs = 200  # More epochs for final training
        batch_size = params.get('batch_size', 32)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop with validation tracking
        model.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    def get_model_summary(self, model_index=None):
        """
        Display comprehensive performance summary for models.
        
        Parameters:
        -----------
        model_index : int, optional
            Index of specific model to summarize. If None, summarizes all models.
        """
        if model_index is not None:
            # Summary for specific model
            if model_index >= len(self.model_performance):
                raise ValueError(f"Model index {model_index} not available")

            perf = self.model_performance[model_index]
            print(f"\n🏆 MODEL {model_index} PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"Model Type: {perf['model_type']}")
            print(f"Training R²: {perf['R2_train_val']:.4f}")
            print(f"Test R²: {perf['R2']:.4f}")
            print(f"Training MSE: {perf['MSE_train_val']:.6f}")
            print(f"Test MSE: {perf['MSE']:.6f}")
            print(f"Training Time: {perf['training_time']:.2f} seconds")
            print(f"Avg Inference Time (Train): {perf['average_inference_time_train']:.6f} s/sample")
            print(f"Avg Inference Time (Test): {perf['average_inference_time_test']:.6f} s/sample")
            print("=" * 60)
        else:
            # Summary for all models
            print("\n🏆 ALL MODELS PERFORMANCE SUMMARY")
            print("=" * 80)

            if not self.model_performance:
                print("No models trained yet.")
                return

            for i, perf in enumerate(self.model_performance):
                print(f"Model {i} ({perf['model_type']}):")
                print(f"  Training R²: {perf['R2_train_val']:.4f} | Test R²: {perf['R2']:.4f}")
                print(f"  Training MSE: {perf['MSE_train_val']:.6f} | Test MSE: {perf['MSE']:.6f}")
                print(f"  Training Time: {perf['training_time']:.2f}s | Inference: {perf['average_inference_time_test']:.6f}s/sample")
                print("-" * 60)

            print("=" * 80)

    def export_model_onnx(self, model_idx=0, output_path='model.onnx'):
        """
        Export PyTorch model to ONNX format for deployment.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to export
        output_path : str
            Path to save ONNX model
        """
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available for ONNX export")
            return

        model_type = self.model_types[model_idx]

        if model_type == 2:  # PyTorch MLP
            try:
                import torch.onnx

                model = self.models[model_idx]
                model.eval()

                # Create dummy input with correct shape
                if self.F is None:
                    raise ValueError("Number of features (F) not set. Call load_data() first.")
                dummy_input = torch.randn(1, self.F, dtype=torch.float32)

                # Export to ONNX
                torch.onnx.export(
                    model,
                    (dummy_input,),
                    output_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )

                print(f"✅ Model exported to ONNX: {output_path}")

            except ImportError:
                print("⚠️ ONNX export requires torch.onnx")
            except Exception as e:
                print(f"❌ ONNX export failed: {e}")
        else:
            print("⚠️ ONNX export only available for PyTorch models (type 2)")

    def run_comprehensive_cv(self, model_idx=0, cv_folds=5, n_repeats=3):
        """
        Run comprehensive cross-validation with multiple repeats.
        
        Parameters:
        -----------
        model_idx : int
            Index of model to cross-validate
        cv_folds : int
            Number of CV folds
        n_repeats : int
            Number of repeated CV runs
        """
        from sklearn.model_selection import RepeatedKFold

        print(f'\n🔄 Comprehensive CV: {n_repeats} × {cv_folds}-fold cross-validation')

        model = self.models[model_idx]

        # Setup repeated k-fold
        rkf = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(
            model, self.x_train_val_sc, self.y_train_val_sc,
            cv=rkf, scoring='neg_mean_squared_error', n_jobs=-1
        )

        # Convert to positive MSE
        cv_mse = -cv_scores

        print(f'📊 Comprehensive CV results ({len(cv_scores)} total folds):')
        print(f'   Mean MSE: {cv_mse.mean():.6f} ± {cv_mse.std():.6f}')
        print(f'   Min MSE: {cv_mse.min():.6f}')
        print(f'   Max MSE: {cv_mse.max():.6f}')

        return {
            'cv_scores': cv_mse,
            'mean_mse': cv_mse.mean(),
            'std_mse': cv_mse.std(),
            'min_mse': cv_mse.min(),
            'max_mse': cv_mse.max()
        }

    def tune(self, model_index=0, method='bayesian', n_trials=20, search_space=None, 
             monitor_resources=True, plot_resources=True, plot_results=True):
        """
        Memory-efficient hyperparameter tuning with various optimization methods and resource monitoring.
        
        Parameters:
        -----------
        model_index : int
            Index of the model to tune
        method : str
            Tuning method: 'random_mem_eff', 'bayesian_skopt', 'optuna_botorch', 'optuna_tpe'
        n_trials : int
            Number of optimization trials
        search_space : dict, optional
            Custom search space for hyperparameters
        monitor_resources : bool, default=True
            Whether to monitor system resources (RAM, CPU) during tuning
        plot_resources : bool, default=True
            Whether to plot resource usage after tuning completes
        plot_results : bool, default=True
            Whether to plot optimization results showing R² evolution
            
        Returns:
        --------
        dict : Tuning results with best parameters, history, and resource monitoring data
        """

        if model_index >= len(self.models):
            raise ValueError(f"Model index {model_index} not available")

        model_type = self.model_types[model_index]
        if model_type != 0:  # Only Random Forest for now
            raise ValueError("Hyperparameter tuning currently only supports Random Forest models")

        print(f"🔍 Hyperparameter Tuning: {method.upper()}")
        print(f"Model: RandomForestRegressor | Trials: {n_trials}")
        if monitor_resources:
            print("📊 Resource monitoring: ENABLED")
        print("=" * 60)

        # Check if baseline performance has already been calculated
        if not hasattr(self, 'baseline_r2') or self.baseline_r2 is None:
            print("📊 Calculating baseline performance...")
            baseline_model = RandomForestRegressor(
                n_estimators=100, max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, random_state=42, n_jobs=-1
            )
            baseline_model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_baseline_sc = baseline_model.predict(self.x_test_sc)
            y_pred_baseline = self.y_scaler.inverse_transform(y_pred_baseline_sc)
            self.baseline_r2 = r2_score(self.y_test, y_pred_baseline)
            print(f"📊 Baseline R²: {self.baseline_r2:.4f}")
        else:
            print(f"📊 Using cached baseline R²: {self.baseline_r2:.4f}")
        print("=" * 60)

        # Initialize resource monitor if requested
        resource_monitor = None
        if monitor_resources:
            try:
                from .utils import ResourceMonitor
                resource_monitor = ResourceMonitor()
                initial_ram, initial_cpu = resource_monitor.update()
                print(f"📊 Initial Resources: RAM: {initial_ram:.1f} MB | CPU: {initial_cpu:.1f}%")
                print("=" * 60)
            except ImportError as e:
                print(f"⚠️ Resource monitoring disabled: {e}")
                monitor_resources = False

        # Default search space for Random Forest
        # max_depth is fixed to None (unlimited) for all optimization methods
        if search_space is None:
            search_space = {
                'n_estimators': (50, 500),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', 1.0]
                # max_depth is excluded from optimization and fixed to None
            }

        # Record start time for total tuning time
        import time
        start_time = time.time()

        # Execute the appropriate tuning method
        if method == 'random_mem_eff':
            result = self._tune_random_memory_efficient(model_index, n_trials, search_space, resource_monitor)
        elif method == 'bayesian_skopt':
            result = self._tune_bayesian_skopt(model_index, n_trials, search_space, resource_monitor)
        elif method == 'optuna_botorch':
            result = self._tune_optuna_botorch(model_index, n_trials, search_space, resource_monitor)
        elif method == 'optuna_tpe':
            result = self._tune_optuna_tpe(model_index, n_trials, search_space, resource_monitor)
        else:
            raise ValueError(f"Unknown tuning method: {method}")

        # Calculate total tuning time
        total_tuning_time = time.time() - start_time
        result['total_tuning_time'] = total_tuning_time
        result['baseline_r2'] = self.baseline_r2

        print(f"\n⏱️ Total tuning time: {total_tuning_time:.2f} seconds")
        print(f"📊 Baseline vs Best: {self.baseline_r2:.4f} → {result['best_r2']:.4f} "
              f"(Δ = {result['best_r2'] - self.baseline_r2:+.4f})")

        # Add resource monitoring results and plotting
        if resource_monitor is not None:
            resource_summary = resource_monitor.get_summary()
            result['resource_summary'] = resource_summary
            result['resource_monitor'] = resource_monitor
            
            print(f"\n📊 Resource Summary:")
            print(f"   Peak RAM Usage: {resource_summary['peak_ram_mb']:.1f} MB")
            print(f"   Average RAM Usage: {resource_summary['avg_ram_mb']:.1f} MB")
            print(f"   Average CPU Usage: {resource_summary['avg_cpu_percent']:.1f}%")
            print(f"   Total Duration: {resource_summary['duration_sec']:.1f} seconds")
            
            if plot_resources:
                resource_monitor.plot_resources(title=f"Resource Usage - {method.upper()} Optimization")

        # Plot optimization results if requested
        if plot_results and 'r2_evolution' in result:
            self._plot_optimization_results(result, self.baseline_r2)

        return result

    def _tune_random_memory_efficient(self, model_index, n_trials, search_space, resource_monitor=None):
        """Random search with memory cleanup and resource monitoring"""
        import gc

        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score

        # Initialize tracking
        results = {
            'method': 'random_mem_eff',
            'iterations': [],
            'r2_scores': [],
            'r2_evolution': [],  # Track R² evolution over iterations
            'best_r2': -np.inf,
            'best_params': None,
            'hyperparams_history': []
        }

        np.random.seed(42)
        for i in range(n_trials):
            # Monitor resources before training
            if resource_monitor:
                pre_ram, pre_cpu = resource_monitor.update()
            
            # Sample random hyperparameters with proper types
            params = {
                'n_estimators': int(np.random.randint(search_space['n_estimators'][0], search_space['n_estimators'][1] + 1)),
                'max_depth': None,  # Fixed to None (unlimited depth)
                'min_samples_split': int(np.random.randint(search_space['min_samples_split'][0], search_space['min_samples_split'][1] + 1)),
                'min_samples_leaf': int(np.random.randint(search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1] + 1)),
                'max_features': search_space['max_features'][np.random.randint(0, len(search_space['max_features']))],
                'random_state': 42,
                'n_jobs': -1
            }

            # Train and evaluate
            model = RandomForestRegressor(**params)
            model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_sc = model.predict(self.x_test_sc)
            y_pred = self.y_scaler.inverse_transform(y_pred_sc)
            r2 = r2_score(self.y_test, y_pred)

            # Monitor resources after training
            if resource_monitor:
                post_ram, post_cpu = resource_monitor.update()

            # Track results
            results['iterations'].append(i + 1)
            results['r2_scores'].append(r2)
            results['r2_evolution'].append(max(results['r2_scores']))  # Best R² so far
            results['hyperparams_history'].append(params.copy())

            if r2 > results['best_r2']:
                results['best_r2'] = r2
                results['best_params'] = params.copy()

            # Enhanced progress report with resource info
            if resource_monitor:
                print(f"Trial {i+1:2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f} | "
                      f"RAM: {pre_ram:.0f}→{post_ram:.0f}MB")
            else:
                print(f"Trial {i+1:2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f}")

            # Memory cleanup
            del model
            gc.collect()
            
            # Monitor resources after cleanup
            if resource_monitor:
                resource_monitor.update()

        return results

    def _tune_bayesian_skopt(self, model_index, n_trials, search_space, resource_monitor=None):
        """Bayesian optimization using scikit-optimize with resource monitoring"""
        try:
            import gc

            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
            from skopt import BayesSearchCV
        except ImportError:
            raise ImportError("scikit-optimize not available. Install with: pip install scikit-optimize")

        # Monitor resources at start
        if resource_monitor:
            resource_monitor.update()

        # Convert search space for skopt (max_depth excluded from optimization)
        skopt_space = {}
        for key, value in search_space.items():
            if key == 'max_features':
                skopt_space[key] = value  # Categorical
            elif isinstance(value, tuple):
                skopt_space[key] = value  # Continuous/Integer ranges
            else:
                skopt_space[key] = value  # Categorical

        # Initialize base model with max_depth=None (fixed, not optimized)
        base_model = RandomForestRegressor(max_depth=None, random_state=42, n_jobs=-1)

        # Bayesian search
        opt = BayesSearchCV(
            estimator=base_model,
            search_spaces=skopt_space,
            n_iter=n_trials,
            scoring='r2',
            cv=1,
            n_jobs=-1,  
            random_state=42,
            verbose=2
        )

        print("🚀 Running Bayesian optimization with scikit-optimize...")
        
        # Monitor during optimization
        if resource_monitor:
            resource_monitor.update()
            
        opt.fit(self.x_train_val_sc, self.y_train_val_sc)
        
        # Monitor after optimization
        if resource_monitor:
            resource_monitor.update()

        # Evaluate best model on test set
        y_pred_sc = opt.predict(self.x_test_sc)
        y_pred = self.y_scaler.inverse_transform(y_pred_sc)
        test_r2 = r2_score(self.y_test, y_pred)

        # Extract iteration data from cv_results_ for consistent format
        cv_results = opt.cv_results_ if hasattr(opt, 'cv_results_') else {}
        iterations = list(range(1, len(cv_results.get('mean_test_score', [])) + 1)) if cv_results else []
        r2_scores = list(cv_results.get('mean_test_score', [])) if cv_results else []
        
        # Calculate r2_evolution (best R² so far at each iteration)
        r2_evolution = []
        if r2_scores:
            best_so_far = -np.inf
            for score in r2_scores:
                if score > best_so_far:
                    best_so_far = score
                r2_evolution.append(best_so_far)

        results = {
            'method': 'bayesian_skopt',
            'iterations': iterations,
            'r2_scores': r2_scores,
            'r2_evolution': r2_evolution,
            'best_r2': test_r2,
            'best_params': dict(opt.best_params_) if hasattr(opt, 'best_params_') else {},
            'cv_best_score': opt.best_score_ if hasattr(opt, 'best_score_') else None,
            'optimization_results': cv_results
        }

        cv_score = opt.best_score_ if hasattr(opt, 'best_score_') else "N/A"
        print(f"✅ Best CV Score: {cv_score}, Test R²: {test_r2:.4f}")

        return results

    def _tune_optuna_botorch(self, model_index, n_trials, search_space, resource_monitor=None):
        """Optuna optimization with BoTorch sampler and resource monitoring"""
        try:
            import gc

            import numpy as np
            import optuna
            from optuna.integration import BoTorchSampler
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
        except ImportError:
            raise ImportError("Optuna/BoTorch not available. Install with: pip install optuna botorch")

        # Monitor resources at start
        if resource_monitor:
            resource_monitor.update()

        # Track results
        results = {
            'method': 'optuna_botorch',
            'iterations': [],
            'r2_scores': [],
            'r2_evolution': [],
            'best_r2': -np.inf,
            'best_params': None
        }

        def objective(trial):
            # Monitor resources before training
            if resource_monitor:
                pre_ram, pre_cpu = resource_monitor.update()
                
            # Suggest hyperparameters (max_depth fixed to None)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', search_space['n_estimators'][0], search_space['n_estimators'][1]),
                'max_depth': None,  # Fixed to None (unlimited depth)
                'min_samples_split': trial.suggest_int('min_samples_split', search_space['min_samples_split'][0], search_space['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1]),
                'max_features': trial.suggest_categorical('max_features', search_space['max_features']),
                'random_state': 42,
                'n_jobs': -1
            }

            # Train and evaluate
            model = RandomForestRegressor(**params)
            model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_sc = model.predict(self.x_test_sc)
            y_pred = self.y_scaler.inverse_transform(y_pred_sc)
            r2 = r2_score(self.y_test, y_pred)

            # Monitor resources after training
            if resource_monitor:
                post_ram, post_cpu = resource_monitor.update()

            # Track progress
            results['iterations'].append(len(results['iterations']) + 1)
            results['r2_scores'].append(r2)
            results['r2_evolution'].append(max(results['r2_scores']))  # Best R² so far
            if r2 > results['best_r2']:
                results['best_r2'] = r2
                results['best_params'] = params.copy()

            # Enhanced progress report with resource info
            if resource_monitor:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f} | "
                      f"RAM: {pre_ram:.0f}→{post_ram:.0f}MB")
            else:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f}")

            # Memory cleanup
            del model
            gc.collect()
            
            # Monitor resources after cleanup
            if resource_monitor:
                resource_monitor.update()

            return r2

        # Create study with BoTorch sampler
        sampler = BoTorchSampler()
        study = optuna.create_study(direction='maximize', sampler=sampler)

        print("🚀 Running Optuna optimization with BoTorch sampler...")
        study.optimize(objective, n_trials=n_trials)

        results['study'] = study
        results['best_r2'] = study.best_value
        results['best_params'] = study.best_params

        print(f"✅ Best R²: {study.best_value:.4f}")

        return results

    def _tune_optuna_tpe(self, model_index, n_trials, search_space, resource_monitor=None):
        """Optuna optimization with TPE sampler and resource monitoring"""
        try:
            import gc

            import numpy as np
            import optuna
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
        except ImportError:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        # Monitor resources at start
        if resource_monitor:
            resource_monitor.update()

        # Track results
        results = {
            'method': 'optuna_tpe',
            'iterations': [],
            'r2_scores': [],
            'r2_evolution': [],
            'best_r2': -np.inf,
            'best_params': None
        }

        def objective(trial):
            # Monitor resources before training
            if resource_monitor:
                pre_ram, pre_cpu = resource_monitor.update()
                
            # Suggest hyperparameters (max_depth fixed to None)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', search_space['n_estimators'][0], search_space['n_estimators'][1]),
                'max_depth': None,  # Fixed to None (unlimited depth)
                'min_samples_split': trial.suggest_int('min_samples_split', search_space['min_samples_split'][0], search_space['min_samples_split'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', search_space['min_samples_leaf'][0], search_space['min_samples_leaf'][1]),
                'max_features': trial.suggest_categorical('max_features', search_space['max_features']),
                'random_state': 42,
                'n_jobs': -1
            }

            # Train and evaluate
            model = RandomForestRegressor(**params)
            model.fit(self.x_train_val_sc, self.y_train_val_sc)
            y_pred_sc = model.predict(self.x_test_sc)
            y_pred = self.y_scaler.inverse_transform(y_pred_sc)
            r2 = r2_score(self.y_test, y_pred)

            # Monitor resources after training
            if resource_monitor:
                post_ram, post_cpu = resource_monitor.update()

            # Track progress
            results['iterations'].append(len(results['iterations']) + 1)
            results['r2_scores'].append(r2)
            results['r2_evolution'].append(max(results['r2_scores']))  # Best R² so far
            if r2 > results['best_r2']:
                results['best_r2'] = r2
                results['best_params'] = params.copy()

            # Enhanced progress report with resource info
            if resource_monitor:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f} | "
                      f"RAM: {pre_ram:.0f}→{post_ram:.0f}MB")
            else:
                print(f"Trial {len(results['iterations']):2d}: R² = {r2:.4f} | Best: {results['best_r2']:.4f}")

            # Memory cleanup
            del model
            gc.collect()
            
            # Monitor resources after cleanup
            if resource_monitor:
                resource_monitor.update()

            return r2

        # Create study with TPE sampler (default)
        study = optuna.create_study(direction='maximize')

        print("🚀 Running Optuna optimization with TPE sampler...")
        study.optimize(objective, n_trials=n_trials)

        results['study'] = study
        results['best_r2'] = study.best_value
        results['best_params'] = study.best_params

        print(f"✅ Best R²: {study.best_value:.4f}")

        return results

    def _plot_optimization_results(self, result, baseline_r2):
        """
        Plot optimization results showing R² evolution over iterations.
        
        Parameters:
        -----------
        result : dict
            Tuning results dictionary with r2_evolution, iterations, etc.
        baseline_r2 : float
            Baseline R² score for reference
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            iterations = result.get('iterations', [])
            r2_evolution = result.get('r2_evolution', [])
            method = result.get('method', 'optimization')
            
            if not iterations or not r2_evolution:
                print("⚠️ No iteration data available for plotting")
                return
                
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, r2_evolution, 'b-', linewidth=2, label=f'{method.upper()} - Best R² Evolution')
            plt.axhline(y=baseline_r2, color='r', linestyle='--', linewidth=2, label=f'Baseline R² = {baseline_r2:.4f}')
            
            # Add final improvement annotation
            final_r2 = r2_evolution[-1] if r2_evolution else baseline_r2
            improvement = final_r2 - baseline_r2
            plt.annotate(f'Final: {final_r2:.4f}\nΔ = {improvement:+.4f}', 
                        xy=(len(iterations), final_r2), 
                        xytext=(len(iterations) * 0.7, final_r2 + 0.01),
                        fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black'))
            
            plt.xlabel('Iteration')
            plt.ylabel('R² Score')
            plt.title(f'Hyperparameter Optimization Results - {method.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            print(f"📊 Optimization plot displayed for {method.upper()}")
            
        except ImportError:
            print("⚠️ Matplotlib not available for plotting. Install with: pip install matplotlib")
        except Exception as e:
            print(f"⚠️ Error creating plot: {e}")
