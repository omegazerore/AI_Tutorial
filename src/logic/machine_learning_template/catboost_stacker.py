import os
import pickle
import json
from functools import partial

import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from src.io.path_definition import get_target_folder

# Feature mapping for numerical and categorical features
features_map = {
    'numerical': ['_quantity_imputed_', 'y_pred', '_july_benchmark_diff_prev_',
                  '_dec_benchmark_diff_prev_', "_july_benchmark_diff_", "_dec_benchmark_diff_",
                  '_july_benchmark_', "_dec_benchmark_", "_july_benchmark_projectile_",
                  '_dec_benchmark_projectile_', 'L', 'A', 'B'],
    'categorical': ['month', 'brand', '_type_imputed_',
                    '_july_benchmark_prev_type_imputed_', '_dec_benchmark_prev_type_imputed_',
                    '_july_benchmark_type_imputed_', '_dec_benchmark_type_imputed_']}

# File paths for model and hyperparameters storage
model_filename = os.path.join(get_target_folder(), "catboost_models.pkl")
hyperparameters_filename = os.path.join(get_target_folder(), "catboost_best_hyperparameters.json")

class CatBoostWrapper:
    """
    A wrapper class for CatBoostRegressor to facilitate model training, prediction,
    cross-validation, and hyperparameter tuning.
    """

    def __init__(self):
        """
        Initializes the CatBoostWrapper instance.
        """
        self.models = {}
        self.features = features_map['numerical'] + features_map['categorical']

    def fit(self, df: pd.DataFrame, label: str, **kwargs) -> CatBoostRegressor:
        """
       Trains a CatBoostRegressor model.

       Args:
           df (pd.DataFrame): The dataset containing features and labels.
           label (str): The name of the target column.
           **kwargs: Additional hyperparameters for the CatBoostRegressor.

       Returns:
           CatBoostRegressor: The trained model.
       """
        train_, eval_ = train_test_split(df, test_size=0.2, random_state=42)

        params = {"min_data_in_leaf": kwargs.get('min_data_in_leaf', 1),
                  "l2_leaf_reg": kwargs.get('l2_leaf_reg', 0),
                  "random_strength": kwargs.get('random_strength', 0),
                  "learning_rate": kwargs.get("learning_rate",  0.03),
                  "depth": kwargs.get("depth", 6),
                  "subsample": kwargs.get("subsample", 1)}

        train_pool = Pool(train_[self.features],
                          label=train_[label],
                          cat_features=features_map['categorical'])
        eval_pool = Pool(eval_[self.features],
                         label=eval_[label],
                         cat_features=features_map['categorical'])

        model = CatBoostRegressor(random_state=42, iterations=3000, loss_function='RMSE',
                                  early_stopping_rounds=10)

        model.set_params(**params)

        model.fit(train_pool, eval_set=eval_pool, verbose=100)

        return model

    def predict(self, df: pd.DataFrame):
        """
        Generates predictions using trained models.

        Args:
            df (pd.DataFrame): The dataset for which predictions are to be made.

        Returns:
            np.ndarray: The mean predictions from all trained models.
        """
        pool = Pool(df[self.features], cat_features=features_map['categorical'])

        y_pred = np.zeros((len(df), len(self.models)))

        for idx, (_, model) in enumerate(self.models.items()):
            y_pred[:, idx] = model.predict(pool)

        return y_pred.mean(axis=1)

    def cross_validation(self, df: pd.DataFrame, label: str, **kwargs):
        """
        Performs k-fold cross-validation on the dataset.

        Args:
            df (pd.DataFrame): The dataset containing features and labels.
            label (str): The name of the target column.
            **kwargs: Additional hyperparameters for the CatBoostRegressor.

        Returns:
            np.ndarray: The cross-validation predictions.
        """
        # df[features_map['categorical']] = df[features_map['categorical']].astype("category")

        print(pd.isnull(df[self.features]).any(), flush=True)

        kfold = KFold(n_splits=5, random_state=42, shuffle=True)

        y_pred = np.zeros(len(df))

        for idx, (train_index, eval_index) in enumerate(kfold.split(df)):
            data_train, data_eval = df.iloc[train_index], df.iloc[eval_index]

            self.models[idx] = self.fit(data_train, label, **kwargs)

            eval_pool = Pool(data_eval[self.features],
                             cat_features=features_map['categorical'])

            y_pred[eval_index] = self.models[idx].predict(eval_pool)

        return y_pred

    def _save_models(self):
        """
        Saves trained models to a file.
        """
        global model_filename

        with open(model_filename, "wb") as f:
            pickle.dump(self.models, f)

    def _load_models(self):
        """
        Loads trained models from a file.
        """
        global model_filename

        with open(model_filename, "rb") as f:
            self.models = pickle.load(f)

    def _save_hyperparameters(self, study):
        """
        Saves the best hyperparameters obtained from tuning.

        Args:
            study (optuna.Study): The Optuna study object.
        """
        global hyperparameters_filename

        with open(hyperparameters_filename, "w") as f:
            json.dump(study.best_params, f)

    def load_hyperparameters(self):
        """
        Loads previously saved hyperparameters.

        Returns:
            dict: The hyperparameters dictionary.
        """
        global hyperparameters_filename

        with open(hyperparameters_filename, "r") as f:
            return json.load(f)

    def objective(self, trial, df: pd.DataFrame, label: str):
        """
        Defines the objective function for hyperparameter tuning.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            df (pd.DataFrame): The dataset.
            label (str): The target column.

        Returns:
            float: The RMSE score.
        """
        params = {"min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 3, 11),
                  "l2_leaf_reg": trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10),
                  "random_strength": trial.suggest_uniform('random_strength', 0, 1),
                  "learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 1),
                  "depth": trial.suggest_int('depth', 3, 7),
                  "subsample": trial.suggest_uniform('subsample', 0.4, 1.0)}

        y_pred = self.cross_validation(df, label, **params)

        return root_mean_squared_error(df[label], y_pred)

    def hyperparameter_tuning(self, df: pd.DataFrame, label: str, n_trials: int = 10):
        """
        Runs hyperparameter tuning using Optuna.
        https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

        Args:
            df (pd.DataFrame): The dataset.
            label (str): The target column.
            n_trials (int, optional): Number of trials. Defaults to 10.
        """

        objective = partial(self.objective, df=df, label=label)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Show best hyperparameters
        print("Best hyperparameters:", study.best_params, flush=True)
        print("Best RMSE", study.best_value, flush=True)

        self._save_hyperparameters(study)