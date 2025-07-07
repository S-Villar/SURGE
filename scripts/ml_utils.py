import time, copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from itertools import product
import pickle
import os.path

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold, RepeatedKFold, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

import gpflow as gpf
from gpflow.mean_functions import Constant
from gpflow.utilities import positive, print_summary
from gpflow.ci_utils import reduce_in_tests

gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")

from skopt import BayesSearchCV
import optuna
from optuna.integration import BoTorchSampler
import warnings
import tqdm

class MLTrainer():
    def __init__(self, F, T, dir_path = None):
        print(" Initializing the model")
        #self.initial_model = copy.deepcopy(model)
        self.F = F
        self.T = T
        self.m = []
        self.mtype = []
        self.dir_path = dir_path or ''

    # Define the MLP model
    # Define the nested MLP class
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size, dropout_rate, activation_fn):
            #super(MLTrainer.MLP, self).__init__()
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

        def predict(self,x):
            """
            Custom predict method to handle both NumPy arrays and PyTorch tensors.

            Parameters:
             - x: Input data (NumPy array or PyTorch tensor)

            Returns:
            - pred_np: Predictions as a NumPy array
            """
            self.eval()
            if isinstance(x, np.ndarray):  # If input is a NumPy array
                x = torch.tensor(x, dtype=torch.float32)  # Convert to PyTorch tensor

            with torch.no_grad():  # Disable gradient computation
                prediction = self.forward(x)  # Perform forward pass

            pred_np = prediction.cpu().numpy()  # Convert predictions to NumPy
            return pred_np
            
    def set_model(self, model,model_type):
        #n_models = len(self.m)
        self.m.append(model)
        self.mtype.append(model_type)

    def set_dir_path(self, dir_path = None):
        try:
            home_dir = os.getenv('HOME')
            if home_dir is None:
                raise ValueError("Home environment variable is not set.")
            print(f"Home directory: {home_dir}")
            if dir_path == None:
                self.dir_path = os.dir_path(os.path.join(home_dir,'MLTrainerDir'))
            else:
                self.dir_path = os.path.join(home_dir,dir_path)
            print(f"self.dir_path: {self.dir_path}")
           
            return None
        except Exception as e: 
            print(f"Error: {e}")
            return None

    def init_model(self,model_type):
        if model_type == 1:
            print('Model chosen is Random Forest Regressor')
            self.RFR_init()
        elif model_type == 2:
            m = MLPRegressor()
        elif model_type == 3:
            m = gpf.models.GPR(data=train_data, kernel=k)
        else:
            self.RFR_init()
        #self.set_model(m)

    # Objective function for Optuna
    def objective_mlp(self, trial):
        # Sample hyperparameters
        num_layers = trial.suggest_int("num_layers", 1, 10)
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_int("batch_size", 30, 100)
        activation_fn = trial.suggest_categorical("activation_fn", ["ReLU", "Tanh", "LeakyReLU"])
    
        # Map activation function
        activation_mapping = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU
        }
        activation_fn = activation_mapping[activation_fn]
        
        # Build model
        hidden_layers = [hidden_size] * num_layers
        model = self.MLP(input_size=self.x_train_val_sc.shape[1], hidden_layers=hidden_layers,
                         output_size=self.y_train_val_sc.shape[1], dropout_rate=dropout_rate,
                         activation_fn=activation_fn)
        model = model.to(torch.float32)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        x_trsc_tnsr = torch.tensor(self.x_train_val_sc, dtype=torch.float32)
        y_trsc_tnsr = torch.tensor(self.y_train_val_sc, dtype=torch.float32)
        x_tssc_tnsr = torch.tensor(self.x_test_sc, dtype=torch.float32)
        y_tssc_tnsr = torch.tensor(self.x_test_sc, dtype=torch.float32)
        
        # Data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset=list(zip(x_trsc_tnsr, y_trsc_tnsr)), batch_size=batch_size, shuffle=True
        )
    
        # Training loop
        model.train()
        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(x_tssc_tnsr)
            predictions_inverse = self.y_scaler.inverse_transform(predictions.cpu().numpy())            
            mse = mean_squared_error(self.y_test.to_numpy(), predictions_inverse)
       
        return mse

    def train_mlp_best(self, model, optimizer, loss_fn, n_epochs=300, batch_size=40, val_split=0.2):
        """
        Train the MLP model using the best hyperparameters.
        
        Parameters:
        - model: PyTorch model to train.
        - optimizer: PyTorch optimizer (e.g., Adam).
        - loss_fn: Loss function (e.g., nn.MSELoss).
        - n_epochs: Number of epochs (default: 300).
        - batch_size: Batch size (default: 40).
        - val_split: Proportion of train_val data for validation (default: 0.2).

        Returns:
        - Trained model with the best weights.
        - Training and validation history.
        """
        # Convert training-validation data to tensors
        X_train_val_t = torch.tensor(self.x_train_val_sc, dtype=torch.float32)
        y_train_val_t = torch.tensor(self.y_train_val_sc, dtype=torch.float32)

        # Create a TensorDataset
        train_val_dataset = TensorDataset(X_train_val_t, y_train_val_t)

        # Split into train and validation datasets
        train_size = int(len(train_val_dataset) * (1 - val_split))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize tracking variables
        best_mse = np.inf
        best_weights = None
        history = {"train_loss": [], "val_loss": [], "val_r2": []}

        model = model.to(torch.float32)
        # Training loop
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            with tqdm.tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{n_epochs}") as bar:
                for X_batch, y_batch in bar:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    bar.set_postfix(train_loss=loss.item())

            avg_train_loss = running_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation loop
            model.eval()
            val_loss = 0.0
            val_r2 = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    y_pred = model(X_val)
                    loss = loss_fn(y_pred, y_val)
                    val_loss += loss.item()
                    val_r2 += r2_score(y_val.cpu().numpy(), y_pred.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            avg_val_r2 = val_r2 / len(val_loader)
            history["val_loss"].append(avg_val_loss)
            history["val_r2"].append(avg_val_r2)

            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val R2 = {avg_val_r2:.4f}")

            # Save the best model weights
            if avg_val_loss < best_mse:
                best_mse = avg_val_loss
                best_weights = model.state_dict().copy()

        # Load the best weights
        model.load_state_dict(best_weights)
        return model, history
        
    def load_preproc_data(df, input_features, output_targets, testsplit=0.2):
        self.load_df_dataset(df, input_feature_names, output_feature_names)
        self.train_test_split(testsplit)
        self.standardize_data(self.X_train_val,self.y_train_val,self.X_test,self.y_test)
        
    def load_df_dataset(self, df, input_feature_names, output_feature_names):
        
        self.df = df
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names

        common_prefix = os.path.commonprefix(output_feature_names)

        print("Common prefix :", common_prefix)

        self.main_output = common_prefix

        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]

    def train_test_split(self, testsplit): #splits off test data for later
        
        print("\n Splitting into test and training data")
        self.X_train_val = self.X.sample(frac = 1-testsplit, random_state=1)
        self.idx_train_val = self.X_train_val.index 
        self.y_train_val = self.y.loc[self.idx_train_val]      

        self.X_test = self.X.drop(self.X_train_val.index)
        self.idx_test = self.X_test.index
        self.y_test = self.y.loc[self.idx_test]
             
        print('X train + val shape:', self.X_train_val.shape)
        print('y train + val shape:', self.y_train_val.shape)
        
        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)

        self.X_train = self.X_train_val.sample(frac = 1-testsplit, random_state=1)
        self.idx_train = self.X_train.index 
        self.y_train = self.y_train_val.loc[self.idx_train]      

        self.X_val = self.X_train_val.drop(self.X_train.index)
        self.idx_val = self.X_val.index
        self.y_val = self.y_train_val.loc[self.idx_val]

        self.df['Train[0]/Val[1]/Test[2]'] = np.zeros(self.X.shape[0],'int32')
        self.df.loc[self.idx_val,'Train/Val/Test'] = np.ones(self.idx_val.shape[0])
        self.df.loc[self.idx_test,'Train/Val/Test'] = 2*np.ones(self.idx_test.shape[0])

        print('X train shape:', self.X_train.shape)
        print('y train shape:', self.y_train.shape)

        print('X val shape:', self.X_val.shape)
        print('y val shape:', self.y_val.shape)

        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)

        
    def standardize_data(self):
        """Standardize the dataset using the training set"""
        print("\n Standardizing data")
        
        self.x_scaler = StandardScaler()
        self.x_train_val_sc = self.x_scaler.fit_transform(self.X_train_val)
        self.x_test_sc = self.x_scaler.transform(self.X_test)
        
        self.y_scaler = StandardScaler()
        self.y_train_val_sc = self.y_scaler.fit_transform(self.y_train_val)
        self.y_test_sc = self.y_scaler.transform(self.y_test)

        print("x_train_sc, x_test_sc, y_train_sc, y_test_sc generated")
        print("\n--- Statistics for  x_scaler [all features] ---")
        print("means: ",self.x_scaler.mean_)
        print("standard deviation: ",np.sqrt(self.x_scaler.var_))

        idx_max_ytrain_mean = np.argmax(self.y_scaler.mean_)
        
        print("\n--- Statistics for y_scaler [on axis, max_value] ---")
        print("means: ",self.y_scaler.mean_[[0,idx_max_ytrain_mean]])
        print("standard deviation: ",np.sqrt(self.y_scaler.var_[[0,idx_max_ytrain_mean]]))

    def predict_output(self,model_index):
        print('--- Predicting outputs ---')
        i = model_index
        print('--- Model ',i,' ---')  
        model = self.m[i]

        print('\n--- Training Set Results ---')
        start = time.time()
        self.y_pred_train_val = self.y_scaler.inverse_transform(model.predict(self.x_train_val_sc))
        t_pred_train_val = time.time()-start
        print('\n t_I(avg) =', t_pred_train_val/(self.x_train_val_sc.shape[0]))

        self.MSE_train_val = mean_squared_error(self.y_train_val,self.y_pred_train_val)
        self.R2_train_val = r2_score(self.y_train_val,self.y_pred_train_val)

        print(' MSE_train_val = ',self.MSE_train_val)
        print(' R2_train_val = ',self.R2_train_val)

        print('\n--- Testing Set Results ---')            
        start = time.time()
        self.y_pred_test = self.y_scaler.inverse_transform(model.predict(self.x_test_sc))
        t_pred_test = time.time()-start
        print('\n t_I(avg) =', t_pred_test/(self.x_test_sc.shape[0]))

        self.MSE = mean_squared_error(self.y_test,self.y_pred_test)
        self.R2 = r2_score(self.y_test,self.y_pred_test)

        print(' MSE = ',self.MSE)
        print(' R2 = ',self.R2) 
        
    def RFR_init(self,n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2,
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None,
            min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
            verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
        model = RandomForestRegressor(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=2,
            min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
            verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)         
        self.set_model(model,0)

    

    def GPR_init():
        print('TBD: In progress')
        '''The idea is to prototype the whole method to implement the GPR here.
           All models require initialization, hyperparameter tunning, training, and inference.
           For GPR the process is: 
               1) Kernel initialization and model definition. 
               2) Run a kfold for validation only on the training data. That kfold uses Scipy optimizer to
               minimize the training loss using the trainable_variables of the model. Then provides performance stats.
               3) Then now train on the training dataset the same workflow. 
               (2.1/3.1) HT of the model regarding kernels.
               4) 
               3)need to specify an initialization, '''
        
    def train(self,model_index):
        assert model_index<=len(self.m), 'Model selected not in the model list'
        print('\n--- Training Model ',model_index,' ---')
        if self.mtype[model_index] ==0:
            start = time.time()
            self.m[model_index].fit(self.x_train_val_sc,self.y_train_val_sc)
            training_time = time.time()-start
            print(f"Elapsed time: {training_time:.2f} seconds")
        elif self.mtype[model_index] ==1:
            start = time.time()
            
    def optimize_solver(self, n_iter = 50, model_idx=None, search_spaces=None):
        model_type = self.mtype[model_idx]
        if model_type == 0:
            print('Optimizing Random Forest Regressor Model')
            print('- Method: Bayesian Search Cross Validated')
            
            m = self.m[model_idx]
            if search_spaces == None:
                search_spaces = {
                    'n_estimators': (50, 500),                 # Number of trees in the forest
                    'max_depth': (5, 50),                      # Maximum depth of the tree
                    'min_samples_split': (2, 20),              # Minimum number of samples required to split
                    'min_samples_leaf': (1, 10),               # Minimum samples required in a leaf
                    'max_features': [1.0, 'log2', 'sqrt'],     # Number of features to consider for a split
                }
            
            opt = BayesSearchCV(estimator=m, search_spaces=search_spaces,
                                n_iter=30,  # Number of optimization iterations
                                scoring='neg_mean_squared_error',  # Optimization objective
                                cv=3,  # Cross-validation
                                n_jobs=-1,
                                random_state=42,
                                verbose=2
            )

            print(f'Run the optimization for n_iter = {n_iter}')
            opt.fit(self.x_train_val_sc,self.y_train_val_sc)
            print("Best hyperparameters:", opt.best_params_)
            self.m[model_idx] = opt.best_estimator_
            self.predict_output(model_idx)
        elif model_type == 1:
            print('Model optimization for MLPRegressor using PyTorch')
        else:
            print('Optimization not implemented for this model type')

    def save_prediction(self, output_fmt = 'pickle', output_dir = None, model_idx = 0):
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        iflag = 0
        if output_fmt == 'pickle':
            pickle_file_path = os.path.join(output_dir, 'pred_model_',str(self.mtype[model_idx]))
            model_type_str = ''
            #Create columns for
            columns_out = self.output_feature_names
            if self.mtype[model_idx] == 0:
                model_type_str = 'RFR'
            elif self.mtype[model_idx] == 1:
                model_type_str = 'MLP'
            elif self.mtype[model_idx] == 2:
                model_type_str = 'GPR'
            else:
                model_type_str = 'Model'

            # Model adding training and test columns for the mean prediction
            columns_out = [model_type_str+ item for item in columns_out]
            
            y_pred_model = np.zeros(self.y.shape)
                
                #y_pred_model[self.idx_train,:] = self.y_pred_train_val
                #y_pred_model[self.idx_test,:] = self.y_pred_test
                                
            df_y_pred = pd.DataFrame(data=y_pred_model,columns=columns_out,
                                     dtype='float64')
            df_y_pred_total = pd.concat([self.df,df_y_pred],axis=1)
            df_y_pred_total.loc[self.idx_train_val, columns_out] = self.y_pred_train_val
            df_y_pred_total.loc[self.idx_test, columns_out] = self.y_pred_test

            # If model is GPR add std values!
            if model_type_str == 'GPR':
                columns_out_std = [model_type_str+'_std_'+ item for item in columns_out]
                ystd_pred_model = np.zeros(self.y.shape)
                df_ystd_pred = pd.DataFrame(data=ystd_pred_model,columns=columns_out_std,
                                         dtype='float64')
                df_y_pred_total = pd.concat([df_y_pred_total,df_ystd_pred],axis=1)
                df_y_pred_total.loc[self.idx_train_val, columns_out_std] = self.ystd_pred_train_val
                df_y_pred_total.loc[self.idx_test, columns_out_std] = self.ystd_pred_test
            
            #Print to a pickle file
            if self.dir_path == None:
                if output_dir == None:
                    path_out_folder = os.path.join(os.getenv('HOME'),'Downloads','Model_prediction')
                else:
                    path_out_folder = 'Model_prediction'
            else:
                if output_dir == None:
                    path_out_folder = os.path.join(self.dir_path,'output_prediction')
                else:
                    path_out_folder = os.path.join(self.dir_path,output_dir)
                
            print('Output dir is', output_dir)
            print('path_out_folder', path_out_folder)
            print('dir_path', self.dir_path)
            path_out_file = os.path.join(path_out_folder,self.main_output+model_type_str+'.pkl')
            print("Saving prediction to dataframe stored in dir = ",path_out_file,", with name ", path_out_file)
                
            df_y_pred_total.to_pickle(path_out_file)
            iflag = 1

        else:
            raise ValueError("Other output formats than pickle are not supported yet.")

        if iflag:
            print('--- Prediction saved ---')

    def load_prediction(self, path_to_df = None):

        print('To be completed')
        

    def optimize(self, maxiter=2000):
        print(f'Optimizing model {self.m}')
        opt = gpf.optimizers.Scipy()
        gpf.utilities.print_summary(self.m)

        objective_closure = self.m.training_loss_closure()

        print(f'After optimization')
        try:
            opt_logs = opt.minimize(objective_closure,
                                    self.m.trainable_variables,
                                    options=dict(maxiter=maxiter))
            print(opt_logs)
        finally:
            gpf.utilities.print_summary(model)
    
    def load_csv_dataset_with_splits(self, path, input_feature_names, output_feature_names, feat_ranges=None):
        self.df = pd.read_csv(path).dropna()
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names
               
        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]
        temp = self.df['split']
        self.X = self.X.join(temp)
        self.y = self.y.join(temp)      
        
        if feat_ranges:
            self.feat_ranges = feat_ranges
            self.normalize_inputs=True
        else:
            self.normalize_inputs=False        
        
    def load_csv_dataset_for_random_trials(self, path, input_feature_names, output_feature_names):
        self.df = pd.read_csv(path).dropna()
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names
        
        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]
        
    #def load_df_dataset(self,df, input_feature_names, output_feature_names):
    #    self.df = df
    #    self.input_feature_names = input_feature_names
    #    self.output_feature_names = output_feature_names
    #    
    #    self.X = self.df[self.input_feature_names]
    #    self.y = self.df[self.output_feature_names]
        
    def TestSplitByIndex(self, percentage): #splits off test data for later
    
        print("Splitting test and training data")
        self.X_train_val = self.X.sample(frac = 1-percentage, random_state=1)
        self.y_train_val = self.y.loc[self.X_train_val.index]
        
        self.X_test = self.X.drop(self.X_train_val.index)
        self.y_test = self.y.loc[self.X_test.index]
                
        print('X train + val shape:', self.X_train_val.shape)
        print('y train + val shape:', self.y_train_val.shape)
        
        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)
    
    def CustomTestFoldSplit(self):
        
#         if self.normalize_inputs ==True:
            
#             print("Normalizing data")
#             #creating dict of ranges for each input feature
#             #feats = ('B0_eqdsk', 'R0_eqdsk', 'ate0', 'dense0', 'elecfld', 'lh_npara1', 'lh_power', 'zeff', 'ip_scale')
#             #ranges = ((2.5,3.5),(1.8,1.9),(1,5),(1e19,5e19),(-0.0001,0.001),(1.7,2.5),(1e5,3e6),(1.5,2.5),(0.5,1.5))
#             #feat_ranges = {f:r for f,r in zip(feats,ranges)}
#             feat_ranges=self.feat_ranges

#             #normalizing data frame input features based on ranges above
#             for feat in self.input_feature_names: #same thing as feats
#                 min_,max_ = feat_ranges[feat]
#                 self.df[feat]=(self.df[feat]-min_)/(max_-min_) #do norm
                
        
        print("Splitting test and training data based on 'split' column")
        
        self.X_test = self.X[self.X['split']=='TEST']
        self.y_test = self.y[self.y['split']=='TEST']
        
        self.X_test.drop('split', axis=1, inplace=True)
        self.y_test.drop('split', axis=1, inplace=True)
        
        self.folds_X = []
        self.folds_y = []
        for i in range(5):
            fold = self.X[self.X['split']==str(i)]
            self.folds_X.append(fold)
            fold = self.y[self.y['split']==str(i)]
            self.folds_y.append(fold)
        
        self.X_all_train = self.X[self.X['split']!='TEST']
        self.y_all_train = self.y[self.y['split']!='TEST']
        self.X_all_train.drop('split', axis=1, inplace=True)
        self.y_all_train.drop('split', axis=1, inplace=True)
        
        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)
    
    def TestSplitByFeatureValue(self, feat, value):
        
        print("filtering all samples out where ", feat, "==", value)
        train_val = self.df.loc[self.df[feat] != value]
        test = self.df.loc[self.df[feat] == value]
        
        print('train:',len(train_val))
        print('test:',len(test))
        
        self.X_train_val = train_val[self.input_feature_names]
        self.y_train_val = train_val[self.output_feature_names]
        
        self.X_test = test[self.input_feature_names]
        self.y_test = test[self.output_feature_names]
        
        print('X train + val shape:', self.X_train_val.shape)
        print('y train + val shape:', self.y_train_val.shape)
        
        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)

    def transform_data(self, X_train, X_test):
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        #X_val_scaled = scaler.transform(X_val)

        return scaler, X_train_scaled, X_test_scaled#X_val_scaled
    
    def run_random_trials(self, test_set_size, n_trials = 100, restart=False):
        
        if restart:
            with open('loopvars.pkl', 'rb') as file:
                r2_list, mse_list, mae_list, mse_confidence_list, n_test, start = pickle.load(file)
                        
        else:
        
            start = 0

            r2_list = []
            mse_list = []
            mae_list = []
            
            _, _, _, y_test = train_test_split(self.X, self.y, test_size=test_set_size)  # To get test set size
            n_test = len(y_test)
            
            mse_confidence_list = np.zeros((n_trials, n_test))
    
        
        print("Test data size: ",n_test) 
   
        if restart:
            print("Restarting random trials...\n")
        else:
            print("Starting random trials...\n")
    
        for i in range(start, n_trials):
            
            print("Trial "+str(i+1)+"/"+str(n_trials)+"\n")
            
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_set_size, random_state=i)
           
            X_scaler, X_train, X_test = self.transform_data(X_train.to_numpy(), X_test.to_numpy())
            y_scaler, y_train, y_test = self.transform_data(y_train.to_numpy(), y_test.to_numpy())
            
            train_data = (X_train, y_train)
            
            ## Initialize model
            self.init_model(train_data)
            print('Passed Init model')

            ## Training the model           
            
            opt = gpf.optimizers.Scipy()
            print("Starting optimization...")
            start_time = time.time()
            opt.minimize(self.m.training_loss, self.m.trainable_variables)
            end_time = time.time()
            exec_time = end_time-start_time
            print("Training execution time (seconds): {:.5f}".format(end_time-start_time))
            
            # mean and variance GP prediction

            y_pred, y_var = self.m.predict_f(X_test)
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)

            # Compute scores for confidence curve plotting.

            y_var_temp = np.unique(y_var,axis=1)
            ranked_confidence_list = np.argsort(y_var_temp, axis=0).flatten()

            print("Computing scores for confidence plotting...")
            for k in range(len(y_test)):

                # Construct the MSE error for each level of confidence

                conf = ranked_confidence_list[0:k+1]
                mse = mean_squared_error(y_scaler.inverse_transform(y_test[conf]), y_scaler.inverse_transform(y_pred[conf]))
                mse_confidence_list[i, k] = mse

            print("Calculating predictions...")
            y_pred_train, _ = self.m.predict_f(X_train)
            train_mse = mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train))
            print("\nTrain MSE: {:.3f}".format(train_mse))

            score = r2_score(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))
            mse = mean_squared_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))
            mae = mean_absolute_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))

            print("Test results:")
            print("R^2: {:.3f}".format(score))
            print("MSE: {:.3f}".format(mse))
            print("MAE: {:.3f}\n\n".format(mae))

            r2_list.append(score)
            mse_list.append(mse)
            mae_list.append(mae)
            
            with open('loopvars.pkl', 'wb') as file:
                pickle.dump([r2_list, mse_list, mae_list, mse_confidence_list, n_test, i], file)
              
            
        r2_list = np.array(r2_list)
        mse_list = np.array(mse_list)
        mae_list = np.array(mae_list)

        print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
        print("mean MSE: {:.4f} +- {:.4f}".format(np.mean(mse_list), np.std(mse_list)/np.sqrt(len(mse_list))))
        
        return(mse_confidence_list, n_test)
    
    def run_k_fold(self, 
                   k, 
                   random_state = 4, 
                   shuffle = False, 
                   print_log = False):
        
        
        kf = KFold(n_splits=k, random_state=random_state, shuffle=shuffle)
        
        training_results = {}
        
        fold = 0
        
        exec_time_all = []
        trained_models_all = []
        train_mse_all = []
        val_r2_score_all = []
        val_mse_all = []
        val_mae_all = []
        
        X_train_all = []
        X_val_all = []
        y_val_all = []
        y_pred_all = []
        y_train_all = []
        
        best_mse = 1000.0
        best_index = 0
        
        for train_index, val_index in kf.split(self.X_train_val):
        #for train_index, test_index in kf.split(self.X):
            
            if print_log==True:
                print("Fold: ", fold)
            
            #X_train, X_val = self.X_train_val.iloc[train_index],self.X_train_val.iloc[val_index]
            #y_train, y_val = self.y_train_val.iloc[train_index],self.y_train_val.iloc[val_index]
                      
            #X_train, X_test = self.X.iloc[train_index],self.X.iloc[test_index]
            #y_train, y_test = self.y.iloc[train_index],self.y.iloc[test_index]
            
            X_train, X_val = self.X.iloc[train_index],self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index],self.y.iloc[val_index]
            
            print("Training set size: ",X_train.shape)
            print("Validation set size: ", X_val.shape)
            
            #self.X_train_val = X_train
            #self.y_train_val = y_train
            #self.X_test = X_test
            #self.y_test = y_test
            
            #temp = X_train.to_numpy()
            #print(temp[0])
            
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            
            X_scaler, X_train, X_val = self.transform_data(X_train.to_numpy(), X_val.to_numpy())
            y_scaler, y_train, y_val = self.transform_data(y_train.to_numpy(), y_val.to_numpy())
                       
            y_val_all.append(y_scaler.inverse_transform(y_val))
            #print(X_train.shape, y_train.shape)

            train_data = (X_train, y_train)
            
            ## Initialize model
            self.init_model(train_data)

            ## Training the model           
            
            opt = gpf.optimizers.Scipy()
            start_time = time.time()
            opt.minimize(self.m.training_loss, self.m.trainable_variables)
            end_time = time.time()
            exec_time = end_time-start_time
            print("Training execution time (seconds): {:.5f}".format(end_time-start_time))
            #print_summary(m)

            ## Calculating errors over the validation set

            self.y_pred, y_var = self.m.predict_f(X_val)
            y_pred_all.append(y_scaler.inverse_transform(self.y_pred))

            self.y_pred_train, _ = self.m.predict_f(X_train)
            train_mse = mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(self.y_pred_train))
            print("Train MSE: {:.5f}".format(train_mse))


            score = r2_score(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mse = mean_squared_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mae = mean_absolute_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))

            print("\nVal R^2: {:.5f}".format(score))
            print("Val MSE: {:.5f} ".format(mse))
            print("Val MAE: {:.5f} \n\n".format(mae))
            
            # Keep best prediction
            if mse<best_mse:
                best_mse = mse
                best_index = fold
            
            exec_time_all.append(exec_time)
            trained_models_all.append(self.m)
            train_mse_all.append(train_mse)
            val_r2_score_all.append(score)
            val_mse_all.append(mse)
            val_mae_all.append(mae)
            
            X_val_all.append(X_scaler.inverse_transform(X_val))
            
            print(self.m)
            
            fold+=1
            
        
        training_results['execution_times']=exec_time_all
        training_results['trained_models']=trained_models_all
        training_results['training_mse']=train_mse_all
        training_results['validation_mse']=val_mse_all
        training_results['validation_mae']=val_mae_all
        training_results['validation_r2']=val_r2_score_all

        training_results['train_index']=train_index
        training_results['val_index']=val_index
        
        training_results['X_train_all']=X_train_all
        training_results['X_val_all']=X_val_all
        training_results['best_index']=best_index
        training_results['y_val_all']=y_val_all
        training_results['y_pred_all']=y_pred_all
        
        training_results['y_train_all']=y_train_all
        
        return training_results
            
        
    def run_k_fold_with_splits(self, inner_cv=False):
        
        training_results = {}
        
       
        exec_time_all = []
        trained_models_all = []
        train_mse_all = []
        val_r2_score_all = []
        val_mse_all = []
        val_mae_all = []
        
        X_train_all = []
        X_val_all = []
        y_val_all = []
        y_pred_all = []
        y_train_all = []
        
        best_models_dict = dict()
        best_models_dict["Kernel_Index"] = list()
        best_models_dict["Name"] = list()
        best_models_dict["Parameters"] = list()
        best_models_dict["MSE"] = list()
        
        best_mse = 1000.0
        best_index = 0
        
        if inner_cv:
            all_inner_results = []
        
            outer_results_mean = np.zeros(shape=(len(self.all_kernels),5))
            ## Each row is a kernel; columns: mean inner fold 0, mean inner fold 1, ...
            outer_results_std = np.zeros(shape=(len(self.all_kernels),5))
            ## Each row is a kernel; columns: std inner fold 0, std inner fold 1, ...

            if os.path.exists('outer_results_power.pkl'):
                print("File exists!")
                with open('outer_results_power.pkl', 'rb') as file:
                    outer_results_mean, outer_results_std, i_temp, fold_temp = pickle.load(file)
        
        for fold in range(5):
            
            print("Fold: ", fold)
            
            X_val = self.folds_X[fold].drop('split', axis=1)
            y_val = self.folds_y[fold].drop('split', axis=1)
                       
            X_train = pd.DataFrame()
            y_train = pd.DataFrame()
            for i in range(5):
                if i != fold:
                    X_train = pd.concat([X_train,self.folds_X[i].drop('split', axis=1)])
                    y_train = pd.concat([y_train,self.folds_y[i].drop('split', axis=1)])

                  
            print("Training set size: ",X_train.shape)
            print("Validation set size: ", X_val.shape)
            
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            
            X_scaler, X_train, X_val = self.transform_data(X_train.to_numpy(), X_val.to_numpy())
            y_scaler, y_train, y_val = self.transform_data(y_train.to_numpy(), y_val.to_numpy())
                       
            y_val_all.append(y_scaler.inverse_transform(y_val))

            train_data = (X_train, y_train)
            
            ##################### Inner cross validation for hyperparameters tunning ##
            
            if inner_cv:
            
                cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

                inner_results = np.zeros(shape=(len(self.all_kernels),5))
                # Each row is one kernel, each column: mse inner fold 0, mse inner fold 1, msr inner dold 2, avg mse, sd mse

                best_inner_mse = 1000.0
                best_model_index = 0

                for i in range(0, len(self.all_kernels)):
                    start_time = time.time()
                    print("Kernel: ",i)
                    
                    f = 0
                    mse_temp = list()
                    for train_index, val_index in cv_inner.split(X_train_all[-1]):

                        print("Fold inner: ", f)

                        X_inner_train, X_inner_val = X_train_all[-1].iloc[train_index],X_train_all[-1].iloc[val_index]
                        y_inner_train, y_inner_val = y_train_all[-1].iloc[train_index],y_train_all[-1].iloc[val_index]

                        X_inner_scaler, X_inner_train, X_inner_val = self.transform_data(X_inner_train.to_numpy(), X_inner_val.to_numpy())
                        y_inner_scaler, y_inner_train, y_inner_val = self.transform_data(y_inner_train.to_numpy(), y_inner_val.to_numpy())

                        train_inner_data = (X_inner_train, y_inner_train)

                        #print("Training inner set size: ",X_inner_train.shape, y_inner_train.shape)
                        #print("Validation inner set size: ", X_inner_val.shape, y_inner_val.shape)

                        k = self.all_kernels[i]
                        m = gpf.models.GPR(data=train_inner_data, kernel=k)
                        opt = gpf.optimizers.Scipy()
                        #start_time = time.time()
                        opt.minimize(m.training_loss, m.trainable_variables)
                        #end_time = time.time()
                        #exec_time = end_time-start_time
                        #print("Training execution time (seconds): {:.5f}".format(end_time-start_time))

                        y_inner_pred, y_inner_var = m.predict_f(X_inner_val)
                        y_inner_pred_train, _ = m.predict_f(X_inner_train)
                        inner_train_mse = mean_squared_error(y_inner_scaler.inverse_transform(y_inner_train), y_inner_scaler.inverse_transform(y_inner_pred_train))
                        inner_val_mse = mean_squared_error(y_scaler.inverse_transform(y_inner_val), y_inner_scaler.inverse_transform(y_inner_pred))

                        inner_results[i][f] = inner_val_mse
                        mse_temp.append(inner_val_mse)
                        print("Inner validation mse: ",inner_val_mse)
                        f = f + 1

                    inner_results[i][3] = np.mean(mse_temp)
                    inner_results[i][4] = np.std(mse_temp)

                    self.all_kernels_dict["MSE"].append(inner_results[i][3])

                    if inner_results[i][3]<best_inner_mse:
                        best_inner_mse = inner_results[i][3]
                        best_model_index = i
                        
                    outer_results_mean[i][fold] = np.mean(mse_temp)
                    outer_results_std[i][fold] = np.std(mse_temp)
                    
                    with open('outer_results_power.pkl', 'wb') as file:
                        pickle.dump([outer_results_mean, outer_results_std, i, fold], file)
                        
                    end_time = time.time()
                    exec_time = end_time-start_time
                    print("Inner cross-validation execution time (seconds): {:.5f}".format(end_time-start_time))
                    

                all_inner_results.append(inner_results)   
                ##########################################################


                ## Get best model and train in the whole data

                best_model_name = self.all_kernels_dict["Name"][best_model_index]
                best_model_parameters = self.all_kernels_dict["Parameters"][best_model_index]
                #best_model_mse = self.all_kernels_dict["MSE"][best_model_index]

                best_models_dict["Kernel_Index"].append(best_model_index)
                best_models_dict["Name"].append(best_model_name)
                best_models_dict["Parameters"].append(best_model_parameters)
                best_models_dict["MSE"].append(best_model_mse)
            
            ### end if inner_cv loop
            
            else:
                best_model_index = 0
                best_model_name = self.all_kernels_dict["Name"][best_model_index]
                best_model_parameters = self.all_kernels_dict["Parameters"][best_model_index]
                #best_model_mse = self.all_kernels_dict["MSE"][best_model_index]

                best_models_dict["Kernel_Index"].append(best_model_index)
                best_models_dict["Name"].append(best_model_name)
                best_models_dict["Parameters"].append(best_model_parameters)
                #best_models_dict["MSE"].append(best_model_mse)
            
            if inner_cv:
                print("Best model for this fold: ",best_models_dict["Kernel_Index"][fold], best_models_dict["Name"][fold], best_models_dict["Parameters"][fold], best_models_dict["MSE"][fold])
            
            k = self.all_kernels[best_model_index]
            m = gpf.models.GPR(data=train_data, kernel=k)
            opt = gpf.optimizers.Scipy()
            start_time = time.time()
            opt.minimize(m.training_loss, m.trainable_variables)
            end_time = time.time()
            exec_time = end_time-start_time
            print("Training execution time (seconds): {:.5f}".format(end_time-start_time))

            self.y_pred, y_var = m.predict_f(X_val)
            y_pred_all.append(y_scaler.inverse_transform(self.y_pred))

            self.y_pred_train, _ = m.predict_f(X_train)
            train_mse = mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(self.y_pred_train))
            print("Train MSE: {:.5f}".format(train_mse))

            score = r2_score(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mse = mean_squared_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mae = mean_absolute_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))

            print("\nVal R^2: {:.5f}".format(score))
            print("Val MSE: {:.5f} ".format(mse))
            print("Val MAE: {:.5f} \n\n".format(mae))
            
            # Keep best prediction
            if mse<best_mse:
                best_mse = mse
                best_index = fold
            
            exec_time_all.append(exec_time)
            trained_models_all.append(m)
            train_mse_all.append(train_mse)
            val_r2_score_all.append(score)
            val_mse_all.append(mse)
            val_mae_all.append(mae)
            
            X_val_all.append(X_scaler.inverse_transform(X_val))
            
            #print_summary(self.m)
            
            fold+=1
            
        
        training_results['execution_times']=exec_time_all
        training_results['trained_models']=trained_models_all
        training_results['training_mse']=train_mse_all
        training_results['validation_mse']=val_mse_all
        training_results['validation_mae']=val_mae_all
        training_results['validation_r2']=val_r2_score_all

        #training_results['train_index']=train_index
        #training_results['val_index']=val_index
        
        training_results['X_train_all']=X_train_all
        training_results['X_val_all']=X_val_all
        training_results['best_index']=best_index
        training_results['y_val_all']=y_val_all
        training_results['y_pred_all']=y_pred_all
        
        training_results['y_train_all']=y_train_all
        
        training_results['best_models_dict'] = best_models_dict
        
        if inner_cv:
            training_results['all_inner_results'] = all_inner_results
        
        return training_results
        
    def get_device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
          
   


class MLPModel():
    def __init__(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.n_epochs = n_epochs

    def data_train_test(self, X_train, y_train, X_test, y_test, X_scaler=StandardScaler(), Y_Scaler=StandardScaler()):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler

    def build_model(self, params):
        n_layers = int(params['n_layers'])
        hidden_sizes = [int(params[f'hidden_size_{i}']) for i in range(n_layers)]
        layers = []
        input_size = self.input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.output_size))  # Output layer
        
        return nn.Sequential(*layers) 

    def objective(self,params):
        n_layers = int(params['n_layers'])
        hidden_sizes = [int(params[f'hidden_size_{i}']) for i in range(n_layers)]
        learning_rate = params['lr']
        batch_size = int(params['batch_size'])

        # Use synthetic data for this example
        #torch.manual_seed(42)
        #X_train = torch.rand(1000, self.input_size)
        #y_train = torch.rand(1000, self.output_size)
        #X_test = torch.rand(200, self.input_size)
        #y_test = torch.rand(200, self.output_size)

        # Create DataLoader
        train_dataset = TensorDataset(self.X_train,self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Build model
        model = self.build_model(params)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(self.n_epochs):  # Train for 20 epochs
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        return test_loss.item()


    def optimize(self, max_evals=5, span_lr=[-2,-1], span_bs=[100,150]):#span_n_layers=[1,6],span_lr=[-4,-1],span_bs=[20,50]):
        """
        Run the hyperparameter optimization.
        :param max_evals: Number of optimization trials.
        :return: Best hyperparameters.
        """
        # Define the hyperparameter search space
        #max_layers = self.max_layers
        space = {
            #'n_layers': hp.quniform('n_layers', span_n_layers[0], span_n_layers[1], span_n_layers[1]-span_n_layers[0]),
            'lr': hp.loguniform('lr', span_lr[0], span_lr[1]),
            'batch_size': hp.quniform('batch_size', span_bs[0], span_bs[1], ),
        }
        space = {
            'hidden_layer_sizes': hp.choice(
                'hidden_layer_sizes',
                [
                    [hp.randint(f'hidden_size_{i}', 16, 257) for i in range(n)]
                    for n in range(1, 6)  # Allow between 1 and 5 layers
                ]
            ), # Architechture 
            'lr': hp.loguniform('lr', -4, -1),  # Learning rate
            'batch_size': hp.quniform('batch_size', 16, 128, 1),  # Batch size
        }

        # Add hidden sizes dynamically
        for i in range(max_layers):
            space[f'hidden_size_{i}'] = hp.randint(f'hidden_size_{i}', 16, 257)

        # Run optimization
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        return best



class GPRTrainer():
    
    def __init__(self, D, P):
        #print("obtained model:")
        #print(model)
        #self.initial_model = copy.deepcopy(model)
        self.D = D
        self.P = P
        self.all_kernels = list()
        self.all_kernels_dict = dict()
        
    def set_model(self, model):
        self.m = model
    
    def set_allkernels(self, kernels):
        self.all_kernels = kernels
        
    def set_allkernels_dict(self, kernels_dict):
        self.all_kernels_dict = kernels_dict

    def create_kernels(self,hyper=False):
        
        ## Creating all kernels
        all_kernels = list()
        all_kernels_dict = dict()
        all_kernels_dict["Name"] = list()
        all_kernels_dict["Parameters"] = list()
        all_kernels_dict["MSE"] = list()
        
        if hyper:
        
            ## Preparing parameter space
            lengs = list()
            for i in np.arange(0.25,2.1,0.25):
                lengs.append(np.tile(i,9))
            lengs = np.asarray(lengs)

            parameters_kernels1 = list(product(lengs, np.arange(0.1,2.2,0.5),np.arange(0.1,2.2,0.5)))
            #parameters_kernels2 = list(product(lengs, np.arange(0.1,2.2,0.5),np.arange(0.1,2.2,0.5),np.arange(0.1,2.2,0.5)))
            #parameters_kernels1 = list(product(lengs, np.arange(2,2.1,1),np.arange(2,2.1,1)))
            #parameters_kernels2 = list(product(lengs, np.arange(2,2.1,1),np.arange(2,2.1,1),np.arange(1,1.1,1)))
            #print(len(parameters_kernels1))
            #print(len(parameters_kernels2))

            print("Exponential")
            name = "Exponential + Linear"
            for i in range(len(parameters_kernels1)):
                k1 = gpf.kernels.Exponential(lengthscales=parameters_kernels1[i][0],variance=parameters_kernels1[i][1])
                k2 = gpf.kernels.Linear(variance=parameters_kernels1[i][2])
                k = k1+k2
                all_kernels.append(k)
                all_kernels_dict["Name"].append(name)
                all_kernels_dict["Parameters"].append((parameters_kernels1[i][0],parameters_kernels1[i][1],parameters_kernels1[i][2]))

            name = "Matern12 + Linear"
            print("Matern12")
            for i in range(len(parameters_kernels1)):
                k1 = gpf.kernels.Matern12(lengthscales=parameters_kernels1[i][0],variance=parameters_kernels1[i][1])
                k2 = gpf.kernels.Linear(variance=parameters_kernels1[i][2])
                k = k1+k2
                all_kernels.append(k)
                all_kernels_dict["Name"].append(name)
                all_kernels_dict["Parameters"].append((parameters_kernels1[i][0],parameters_kernels1[i][1],parameters_kernels1[i][2]))

            name = "Matern32 + Linear"
            print("Matern32")
            for i in range(len(parameters_kernels1)):
                k1 = gpf.kernels.Matern32(lengthscales=parameters_kernels1[i][0],variance=parameters_kernels1[i][1])
                k2 = gpf.kernels.Linear(variance=parameters_kernels1[i][2])
                k = k1+k2
                all_kernels.append(k)
                all_kernels_dict["Name"].append(name)
                all_kernels_dict["Parameters"].append((parameters_kernels1[i][0],parameters_kernels1[i][1],parameters_kernels1[i][2]))

            name = "Matern52 + Linear"
            print("Matern52")
            for i in range(len(parameters_kernels1)):
                k1 = gpf.kernels.Matern52(lengthscales=parameters_kernels1[i][0],variance=parameters_kernels1[i][1])
                k2 = gpf.kernels.Linear(variance=parameters_kernels1[i][2])
                k = k1+k2
                all_kernels.append(k)
                all_kernels_dict["Name"].append(name)
                all_kernels_dict["Parameters"].append((parameters_kernels1[i][0],parameters_kernels1[i][1],parameters_kernels1[i][2]))

            name = "SquaredExponential + Linear"
            print("SquaredExponential")
            for i in range(len(parameters_kernels1)):
                k1 = gpf.kernels.SquaredExponential(lengthscales=parameters_kernels1[i][0],variance=parameters_kernels1[i][1])
                k2 = gpf.kernels.Linear(variance=parameters_kernels1[i][2])
                k = k1+k2
                all_kernels.append(k)
                all_kernels_dict["Name"].append(name)
                all_kernels_dict["Parameters"].append((parameters_kernels1[i][0],parameters_kernels1[i][1],parameters_kernels1[i][2]))

            name = "RationalQuadratic + Linear"
            print("RationalQuadratic")
            for i in range(len(parameters_kernels1)):
                k1 = gpf.kernels.RationalQuadratic(lengthscales=parameters_kernels1[i][0],variance=parameters_kernels1[i][1],alpha=1)
                k2 = gpf.kernels.Linear(variance=parameters_kernels1[i][2])
                k = k1+k2
                if len(all_kernels)>1104:
                    k
                all_kernels.append(k)
                all_kernels_dict["Name"].append(name)
                all_kernels_dict["Parameters"].append((parameters_kernels1[i][0],parameters_kernels1[i][1],parameters_kernels1[i][2]))
                
        else:
#             k1 = gpf.kernels.Exponential(lengthscales=[.75,.75,.75,.75,.75,.75,.75,.75,.75],variance=0.1)
#             k2 = gpf.kernels.Linear(0.6)
#             k = k1+k2
#             all_kernels.append(k)
#             all_kernels_dict["Name"].append("Exponential")
#             all_kernels_dict["Parameters"].append(([.75,.75,.75,.75,.75,.75,.75,.75,.75],0.1,0.6))
            
#             k1 = gpf.kernels.Matern12(lengthscales=[.75,.75,.75,.75,.75,.75,.75,.75,.75],variance=0.1)
#             k2 = gpf.kernels.Linear(0.6)
#             k = k1+k2
#             all_kernels.append(k)
#             all_kernels_dict["Name"].append("Matern12")
#             all_kernels_dict["Parameters"].append(([.75,.75,.75,.75,.75,.75,.75,.75,.75],0.1,0.6))
            
            k1 = gpf.kernels.Matern32(lengthscales=[.75,.75,.75,.75,.75,.75,.75,.75,.75],variance=0.1)
            k2 = gpf.kernels.Linear(0.6)
            k = k1+k2
            all_kernels.append(k)
            all_kernels_dict["Name"].append("Matern32")
            all_kernels_dict["Parameters"].append(([.75,.75,.75,.75,.75,.75,.75,.75,.75],0.1,0.6))
            
#             k1 = gpf.kernels.Matern52(lengthscales=[.75,.75,.75,.75,.75,.75,.75,.75,.75],variance=0.1)
#             k2 = gpf.kernels.Linear(0.6)
#             k = k1+k2
#             all_kernels.append(k)
#             all_kernels_dict["Name"].append("Matern52")
#             all_kernels_dict["Parameters"].append(([.75,.75,.75,.75,.75,.75,.75,.75,.75],0.1,0.6))
            
#             k1 = gpf.kernels.SquaredExponential(lengthscales=[.75,.75,.75,.75,.75,.75,.75,.75,.75],variance=0.1)
#             k2 = gpf.kernels.Linear(0.6)
#             k = k1+k2
#             all_kernels.append(k)
#             all_kernels_dict["Name"].append("SquaredExponential")
#             all_kernels_dict["Parameters"].append(([.75,.75,.75,.75,.75,.75,.75,.75,.75],0.1,0.6))
            
#             k1 = gpf.kernels.RationalQuadratic(lengthscales=[.75,.75,.75,.75,.75,.75,.75,.75,.75],variance=0.1,alpha=1)
#             k2 = gpf.kernels.Linear(0.6)
#             k = k1+k2
#             all_kernels.append(k)
#             all_kernels_dict["Name"].append("Rational Quadratic")
#             all_kernels_dict["Parameters"].append(([.75,.75,.75,.75,.75,.75,.75,.75,.75],0.1,0.6,0.5))

        print("Created total " + str(len(all_kernels)) + " kernels")
        self.set_allkernels(all_kernels)
        self.set_allkernels_dict(all_kernels_dict)
    
    def init_model(self, train_data):
        print("Initializing model...")
        #print("Changes are being made!!")
        ## Creating kernel
        ## Using the sum of two kernels: Matern52 and Linear
        ## Also using "Automatic Relevance Determination (ARD)" from gpflow, i.e., we can define lengthscales for all the input dimensions

        #print(train_data.shape)
        #print(type(train_data))
        length_inputs = train_data[0].shape[1]
        inp_l_scales = list(np.ones(length_inputs))
        k1 = gpf.kernels.Matern32(lengthscales=inp_l_scales,variance=2)
        k2 = gpf.kernels.Linear(variance=2)
        
        ### Used so far    
        
        #k1 = gpf.kernels.Matern32(lengthscales=[1,1,1,1,1,1,1,1,1],variance=2)
        #k2 = gpf.kernels.Linear(variance=2)
        
        #### Used for Input to XRAY
        # k1 = gpf.kernels.Matern52(lengthscales=[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],variance=1)
        # k2 = gpf.kernels.Linear(variance=1)
        
        #k1 = gpf.kernels.Matern52(lengthscales=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],variance=2)
        #k1 = gpf.kernels.SquaredExponential(lengthscales=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],variance=2)
        #k2 = gpf.kernels.Linear(variance=2)
        
        
        k = k1+k2
        
        # k = gpf.kernels.SquaredExponential(lengthscales=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],variance=1)
        # m = gpf.models.VGP(data=train_data, kernel=k, likelihood=gpf.likelihoods.Gaussian())

        ## Create Gaussian Process Regression model using kernel defined above

        m = gpf.models.GPR(data=train_data, kernel=k)
        #m.likelihood.variance.assign(2)
        self.set_model(m)
    
    def optimize(self, maxiter=2000):
        print(f'Optimizing model {self.m}')
        opt = gpf.optimizers.Scipy()
        gpf.utilities.print_summary(self.m)

        objective_closure = self.m.training_loss_closure()

        print(f'After optimization')
        try:
            opt_logs = opt.minimize(objective_closure,
                                    self.m.trainable_variables,
                                    options=dict(maxiter=maxiter))
            print(opt_logs)
        finally:
            gpf.utilities.print_summary(model)
    
    def load_csv_dataset_with_splits(self, path, input_feature_names, output_feature_names, feat_ranges=None):
        self.df = pd.read_csv(path).dropna()
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names
               
        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]
        temp = self.df['split']
        self.X = self.X.join(temp)
        self.y = self.y.join(temp)      
        
        if feat_ranges:
            self.feat_ranges = feat_ranges
            self.normalize_inputs=True
        else:
            self.normalize_inputs=False
        
        
        
    def load_csv_dataset_for_random_trials(self, path, input_feature_names, output_feature_names):
        self.df = pd.read_csv(path).dropna()
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names
        
        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]
        
    def load_df_dataset(self,df, input_feature_names, output_feature_names):
        self.df = df
        self.input_feature_names = input_feature_names
        self.output_feature_names = output_feature_names
        
        self.X = self.df[self.input_feature_names]
        self.y = self.df[self.output_feature_names]
        
    def TestSplitByIndex(self, percentage): #splits off test data for later
    
        print("Splitting test and training data")
        self.X_train_val = self.X.sample(frac = 1-percentage, random_state=1)
        self.y_train_val = self.y.loc[self.X_train_val.index]
        
        self.X_test = self.X.drop(self.X_train_val.index)
        self.y_test = self.y.loc[self.X_test.index]
                
        print('X train + val shape:', self.X_train_val.shape)
        print('y train + val shape:', self.y_train_val.shape)
        
        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)
    
    def CustomTestFoldSplit(self):
        
#         if self.normalize_inputs ==True:
            
#             print("Normalizing data")
#             #creating dict of ranges for each input feature
#             #feats = ('B0_eqdsk', 'R0_eqdsk', 'ate0', 'dense0', 'elecfld', 'lh_npara1', 'lh_power', 'zeff', 'ip_scale')
#             #ranges = ((2.5,3.5),(1.8,1.9),(1,5),(1e19,5e19),(-0.0001,0.001),(1.7,2.5),(1e5,3e6),(1.5,2.5),(0.5,1.5))
#             #feat_ranges = {f:r for f,r in zip(feats,ranges)}
#             feat_ranges=self.feat_ranges

#             #normalizing data frame input features based on ranges above
#             for feat in self.input_feature_names: #same thing as feats
#                 min_,max_ = feat_ranges[feat]
#                 self.df[feat]=(self.df[feat]-min_)/(max_-min_) #do norm
                
        
        print("Splitting test and training data based on 'split' column")
        
        self.X_test = self.X[self.X['split']=='TEST']
        self.y_test = self.y[self.y['split']=='TEST']
        
        self.X_test.drop('split', axis=1, inplace=True)
        self.y_test.drop('split', axis=1, inplace=True)
        
        self.folds_X = []
        self.folds_y = []
        for i in range(5):
            fold = self.X[self.X['split']==str(i)]
            self.folds_X.append(fold)
            fold = self.y[self.y['split']==str(i)]
            self.folds_y.append(fold)
        
        self.X_all_train = self.X[self.X['split']!='TEST']
        self.y_all_train = self.y[self.y['split']!='TEST']
        self.X_all_train.drop('split', axis=1, inplace=True)
        self.y_all_train.drop('split', axis=1, inplace=True)
        
        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)
    
    def TestSplitByFeatureValue(self, feat, value):
        
        print("filtering all samples out where ", feat, "==", value)
        train_val = self.df.loc[self.df[feat] != value]
        test = self.df.loc[self.df[feat] == value]
        
        print('train:',len(train_val))
        print('test:',len(test))
        
        self.X_train_val = train_val[self.input_feature_names]
        self.y_train_val = train_val[self.output_feature_names]
        
        self.X_test = test[self.input_feature_names]
        self.y_test = test[self.output_feature_names]
        
        print('X train + val shape:', self.X_train_val.shape)
        print('y train + val shape:', self.y_train_val.shape)
        
        print('X test shape:', self.X_test.shape)
        print('y test shape:', self.y_test.shape)

    def transform_data(self, X_train, X_test):
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        #X_val_scaled = scaler.transform(X_val)

        return scaler, X_train_scaled, X_test_scaled#X_val_scaled
    
    def run_random_trials(self, test_set_size, n_trials = 100, restart=False):
        
        if restart:
            with open('loopvars.pkl', 'rb') as file:
                r2_list, mse_list, mae_list, mse_confidence_list, n_test, start = pickle.load(file)
                        
        else:
        
            start = 0

            r2_list = []
            mse_list = []
            mae_list = []
            
            _, _, _, y_test = train_test_split(self.X, self.y, test_size=test_set_size)  # To get test set size
            n_test = len(y_test)
            
            mse_confidence_list = np.zeros((n_trials, n_test))
    
        
        print("Test data size: ",n_test) 
   
        if restart:
            print("Restarting random trials...\n")
        else:
            print("Starting random trials...\n")
    
        for i in range(start, n_trials):
            
            print("Trial "+str(i+1)+"/"+str(n_trials)+"\n")
            
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_set_size, random_state=i)
           
            X_scaler, X_train, X_test = self.transform_data(X_train.to_numpy(), X_test.to_numpy())
            y_scaler, y_train, y_test = self.transform_data(y_train.to_numpy(), y_test.to_numpy())
            
            train_data = (X_train, y_train)
            
            ## Initialize model
            self.init_model(train_data)
            print('Passed Init model')

            ## Training the model           
            
            opt = gpf.optimizers.Scipy()
            print("Starting optimization...")
            start_time = time.time()
            opt.minimize(self.m.training_loss, self.m.trainable_variables)
            end_time = time.time()
            exec_time = end_time-start_time
            print("Training execution time (seconds): {:.5f}".format(end_time-start_time))
            
            # mean and variance GP prediction

            y_pred, y_var = self.m.predict_f(X_test)
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)

            # Compute scores for confidence curve plotting.

            y_var_temp = np.unique(y_var,axis=1)
            ranked_confidence_list = np.argsort(y_var_temp, axis=0).flatten()

            print("Computing scores for confidence plotting...")
            for k in range(len(y_test)):

                # Construct the MSE error for each level of confidence

                conf = ranked_confidence_list[0:k+1]
                mse = mean_squared_error(y_scaler.inverse_transform(y_test[conf]), y_scaler.inverse_transform(y_pred[conf]))
                mse_confidence_list[i, k] = mse

            print("Calculating predictions...")
            y_pred_train, _ = self.m.predict_f(X_train)
            train_mse = mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train))
            print("\nTrain MSE: {:.3f}".format(train_mse))

            score = r2_score(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))
            mse = mean_squared_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))
            mae = mean_absolute_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))

            print("Test results:")
            print("R^2: {:.3f}".format(score))
            print("MSE: {:.3f}".format(mse))
            print("MAE: {:.3f}\n\n".format(mae))

            r2_list.append(score)
            mse_list.append(mse)
            mae_list.append(mae)
            
            with open('loopvars.pkl', 'wb') as file:
                pickle.dump([r2_list, mse_list, mae_list, mse_confidence_list, n_test, i], file)
              
            
        r2_list = np.array(r2_list)
        mse_list = np.array(mse_list)
        mae_list = np.array(mae_list)

        print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
        print("mean MSE: {:.4f} +- {:.4f}".format(np.mean(mse_list), np.std(mse_list)/np.sqrt(len(mse_list))))
        
        return(mse_confidence_list, n_test)
    
    def run_k_fold(self, 
                   k, 
                   random_state = 4, 
                   shuffle = False, 
                   print_log = False):
        
        
        kf = KFold(n_splits=k, random_state=random_state, shuffle=shuffle)
        
        training_results = {}
        
        fold = 0
        
        exec_time_all = []
        trained_models_all = []
        train_mse_all = []
        val_r2_score_all = []
        val_mse_all = []
        val_mae_all = []
        
        X_train_all = []
        X_val_all = []
        y_val_all = []
        y_pred_all = []
        y_train_all = []
        
        best_mse = 1000.0
        best_index = 0
        
        for train_index, val_index in kf.split(self.X_train_val):
        #for train_index, test_index in kf.split(self.X):
            
            if print_log==True:
                print("Fold: ", fold)
            
            #X_train, X_val = self.X_train_val.iloc[train_index],self.X_train_val.iloc[val_index]
            #y_train, y_val = self.y_train_val.iloc[train_index],self.y_train_val.iloc[val_index]
                      
            #X_train, X_test = self.X.iloc[train_index],self.X.iloc[test_index]
            #y_train, y_test = self.y.iloc[train_index],self.y.iloc[test_index]
            
            X_train, X_val = self.X.iloc[train_index],self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index],self.y.iloc[val_index]
            
            print("Training set size: ",X_train.shape)
            print("Validation set size: ", X_val.shape)
            
            #self.X_train_val = X_train
            #self.y_train_val = y_train
            #self.X_test = X_test
            #self.y_test = y_test
            
            #temp = X_train.to_numpy()
            #print(temp[0])
            
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            
            X_scaler, X_train, X_val = self.transform_data(X_train.to_numpy(), X_val.to_numpy())
            y_scaler, y_train, y_val = self.transform_data(y_train.to_numpy(), y_val.to_numpy())
                       
            y_val_all.append(y_scaler.inverse_transform(y_val))
            #print(X_train.shape, y_train.shape)

            train_data = (X_train, y_train)
            
            ## Initialize model
            self.init_model(train_data)

            ## Training the model           
            
            opt = gpf.optimizers.Scipy()
            start_time = time.time()
            opt.minimize(self.m.training_loss, self.m.trainable_variables)
            end_time = time.time()
            exec_time = end_time-start_time
            print("Training execution time (seconds): {:.5f}".format(end_time-start_time))
            #print_summary(m)

            ## Calculating errors over the validation set

            self.y_pred, y_var = self.m.predict_f(X_val)
            y_pred_all.append(y_scaler.inverse_transform(self.y_pred))

            self.y_pred_train, _ = self.m.predict_f(X_train)
            train_mse = mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(self.y_pred_train))
            print("Train MSE: {:.5f}".format(train_mse))


            score = r2_score(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mse = mean_squared_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mae = mean_absolute_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))

            print("\nVal R^2: {:.5f}".format(score))
            print("Val MSE: {:.5f} ".format(mse))
            print("Val MAE: {:.5f} \n\n".format(mae))
            
            # Keep best prediction
            if mse<best_mse:
                best_mse = mse
                best_index = fold
            
            exec_time_all.append(exec_time)
            trained_models_all.append(self.m)
            train_mse_all.append(train_mse)
            val_r2_score_all.append(score)
            val_mse_all.append(mse)
            val_mae_all.append(mae)
            
            X_val_all.append(X_scaler.inverse_transform(X_val))
            
            print(self.m)
            
            fold+=1
            
        
        training_results['execution_times']=exec_time_all
        training_results['trained_models']=trained_models_all
        training_results['training_mse']=train_mse_all
        training_results['validation_mse']=val_mse_all
        training_results['validation_mae']=val_mae_all
        training_results['validation_r2']=val_r2_score_all

        training_results['train_index']=train_index
        training_results['val_index']=val_index
        
        training_results['X_train_all']=X_train_all
        training_results['X_val_all']=X_val_all
        training_results['best_index']=best_index
        training_results['y_val_all']=y_val_all
        training_results['y_pred_all']=y_pred_all
        
        training_results['y_train_all']=y_train_all
        
        return training_results
            
        
    def run_k_fold_with_splits(self, inner_cv=False):
        
        training_results = {}
        
       
        exec_time_all = []
        trained_models_all = []
        train_mse_all = []
        val_r2_score_all = []
        val_mse_all = []
        val_mae_all = []
        
        X_train_all = []
        X_val_all = []
        y_val_all = []
        y_pred_all = []
        y_train_all = []
        
        best_models_dict = dict()
        best_models_dict["Kernel_Index"] = list()
        best_models_dict["Name"] = list()
        best_models_dict["Parameters"] = list()
        best_models_dict["MSE"] = list()
        
        best_mse = 1000.0
        best_index = 0
        
        if inner_cv:
            all_inner_results = []
        
            outer_results_mean = np.zeros(shape=(len(self.all_kernels),5))
            ## Each row is a kernel; columns: mean inner fold 0, mean inner fold 1, ...
            outer_results_std = np.zeros(shape=(len(self.all_kernels),5))
            ## Each row is a kernel; columns: std inner fold 0, std inner fold 1, ...

            if os.path.exists('outer_results_power.pkl'):
                print("File exists!")
                with open('outer_results_power.pkl', 'rb') as file:
                    outer_results_mean, outer_results_std, i_temp, fold_temp = pickle.load(file)
        
        for fold in range(5):
            
            print("Fold: ", fold)
            
            X_val = self.folds_X[fold].drop('split', axis=1)
            y_val = self.folds_y[fold].drop('split', axis=1)
                       
            X_train = pd.DataFrame()
            y_train = pd.DataFrame()
            for i in range(5):
                if i != fold:
                    X_train = pd.concat([X_train,self.folds_X[i].drop('split', axis=1)])
                    y_train = pd.concat([y_train,self.folds_y[i].drop('split', axis=1)])

                  
            print("Training set size: ",X_train.shape)
            print("Validation set size: ", X_val.shape)
            
            X_train_all.append(X_train)
            y_train_all.append(y_train)
            
            X_scaler, X_train, X_val = self.transform_data(X_train.to_numpy(), X_val.to_numpy())
            y_scaler, y_train, y_val = self.transform_data(y_train.to_numpy(), y_val.to_numpy())
                       
            y_val_all.append(y_scaler.inverse_transform(y_val))

            train_data = (X_train, y_train)
            
            ##################### Inner cross validation for hyperparameters tunning ##
            
            if inner_cv:
            
                cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

                inner_results = np.zeros(shape=(len(self.all_kernels),5))
                # Each row is one kernel, each column: mse inner fold 0, mse inner fold 1, msr inner dold 2, avg mse, sd mse

                best_inner_mse = 1000.0
                best_model_index = 0

                for i in range(0, len(self.all_kernels)):
                    start_time = time.time()
                    print("Kernel: ",i)
                    
                    f = 0
                    mse_temp = list()
                    for train_index, val_index in cv_inner.split(X_train_all[-1]):

                        print("Fold inner: ", f)

                        X_inner_train, X_inner_val = X_train_all[-1].iloc[train_index],X_train_all[-1].iloc[val_index]
                        y_inner_train, y_inner_val = y_train_all[-1].iloc[train_index],y_train_all[-1].iloc[val_index]

                        X_inner_scaler, X_inner_train, X_inner_val = self.transform_data(X_inner_train.to_numpy(), X_inner_val.to_numpy())
                        y_inner_scaler, y_inner_train, y_inner_val = self.transform_data(y_inner_train.to_numpy(), y_inner_val.to_numpy())

                        train_inner_data = (X_inner_train, y_inner_train)

                        #print("Training inner set size: ",X_inner_train.shape, y_inner_train.shape)
                        #print("Validation inner set size: ", X_inner_val.shape, y_inner_val.shape)

                        k = self.all_kernels[i]
                        m = gpf.models.GPR(data=train_inner_data, kernel=k)
                        opt = gpf.optimizers.Scipy()
                        #start_time = time.time()
                        opt.minimize(m.training_loss, m.trainable_variables)
                        #end_time = time.time()
                        #exec_time = end_time-start_time
                        #print("Training execution time (seconds): {:.5f}".format(end_time-start_time))

                        y_inner_pred, y_inner_var = m.predict_f(X_inner_val)
                        y_inner_pred_train, _ = m.predict_f(X_inner_train)
                        inner_train_mse = mean_squared_error(y_inner_scaler.inverse_transform(y_inner_train), y_inner_scaler.inverse_transform(y_inner_pred_train))
                        inner_val_mse = mean_squared_error(y_scaler.inverse_transform(y_inner_val), y_inner_scaler.inverse_transform(y_inner_pred))

                        inner_results[i][f] = inner_val_mse
                        mse_temp.append(inner_val_mse)
                        print("Inner validation mse: ",inner_val_mse)
                        f = f + 1

                    inner_results[i][3] = np.mean(mse_temp)
                    inner_results[i][4] = np.std(mse_temp)

                    self.all_kernels_dict["MSE"].append(inner_results[i][3])

                    if inner_results[i][3]<best_inner_mse:
                        best_inner_mse = inner_results[i][3]
                        best_model_index = i
                        
                    outer_results_mean[i][fold] = np.mean(mse_temp)
                    outer_results_std[i][fold] = np.std(mse_temp)
                    
                    with open('outer_results_power.pkl', 'wb') as file:
                        pickle.dump([outer_results_mean, outer_results_std, i, fold], file)
                        
                    end_time = time.time()
                    exec_time = end_time-start_time
                    print("Inner cross-validation execution time (seconds): {:.5f}".format(end_time-start_time))
                    

                all_inner_results.append(inner_results)   
                ##########################################################


                ## Get best model and train in the whole data

                best_model_name = self.all_kernels_dict["Name"][best_model_index]
                best_model_parameters = self.all_kernels_dict["Parameters"][best_model_index]
                #best_model_mse = self.all_kernels_dict["MSE"][best_model_index]

                best_models_dict["Kernel_Index"].append(best_model_index)
                best_models_dict["Name"].append(best_model_name)
                best_models_dict["Parameters"].append(best_model_parameters)
                best_models_dict["MSE"].append(best_model_mse)
            
            ### end if inner_cv loop
            
            else:
                best_model_index = 0
                best_model_name = self.all_kernels_dict["Name"][best_model_index]
                best_model_parameters = self.all_kernels_dict["Parameters"][best_model_index]
                #best_model_mse = self.all_kernels_dict["MSE"][best_model_index]

                best_models_dict["Kernel_Index"].append(best_model_index)
                best_models_dict["Name"].append(best_model_name)
                best_models_dict["Parameters"].append(best_model_parameters)
                #best_models_dict["MSE"].append(best_model_mse)
            
            if inner_cv:
                print("Best model for this fold: ",best_models_dict["Kernel_Index"][fold], best_models_dict["Name"][fold], best_models_dict["Parameters"][fold], best_models_dict["MSE"][fold])
            
            k = self.all_kernels[best_model_index]
            m = gpf.models.GPR(data=train_data, kernel=k)
            opt = gpf.optimizers.Scipy()
            start_time = time.time()
            opt.minimize(m.training_loss, m.trainable_variables)
            end_time = time.time()
            exec_time = end_time-start_time
            print("Training execution time (seconds): {:.5f}".format(end_time-start_time))

            self.y_pred, y_var = m.predict_f(X_val)
            y_pred_all.append(y_scaler.inverse_transform(self.y_pred))

            self.y_pred_train, _ = m.predict_f(X_train)
            train_mse = mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(self.y_pred_train))
            print("Train MSE: {:.5f}".format(train_mse))

            score = r2_score(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mse = mean_squared_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))
            mae = mean_absolute_error(y_scaler.inverse_transform(y_val), y_scaler.inverse_transform(self.y_pred))

            print("\nVal R^2: {:.5f}".format(score))
            print("Val MSE: {:.5f} ".format(mse))
            print("Val MAE: {:.5f} \n\n".format(mae))
            
            # Keep best prediction
            if mse<best_mse:
                best_mse = mse
                best_index = fold
            
            exec_time_all.append(exec_time)
            trained_models_all.append(m)
            train_mse_all.append(train_mse)
            val_r2_score_all.append(score)
            val_mse_all.append(mse)
            val_mae_all.append(mae)
            
            X_val_all.append(X_scaler.inverse_transform(X_val))
            
            #print_summary(self.m)
            
            fold+=1
            
        
        training_results['execution_times']=exec_time_all
        training_results['trained_models']=trained_models_all
        training_results['training_mse']=train_mse_all
        training_results['validation_mse']=val_mse_all
        training_results['validation_mae']=val_mae_all
        training_results['validation_r2']=val_r2_score_all

        #training_results['train_index']=train_index
        #training_results['val_index']=val_index
        
        training_results['X_train_all']=X_train_all
        training_results['X_val_all']=X_val_all
        training_results['best_index']=best_index
        training_results['y_val_all']=y_val_all
        training_results['y_pred_all']=y_pred_all
        
        training_results['y_train_all']=y_train_all
        
        training_results['best_models_dict'] = best_models_dict
        
        if inner_cv:
            training_results['all_inner_results'] = all_inner_results
        
        return training_results
        
    def get_device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
          
