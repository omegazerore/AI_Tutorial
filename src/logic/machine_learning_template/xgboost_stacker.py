import os
import pickle
import json
from functools import partial

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

from src.io.path_definition import get_target_folder

features_map = {
    'numerical': ['_quantity_imputed_', 'y_pred', '_july_benchmark_diff_prev_',
                  '_dec_benchmark_diff_prev_', "_july_benchmark_diff_", "_dec_benchmark_diff_",
                  '_july_benchmark_', "_dec_benchmark_", "_july_benchmark_projectile_",
                  '_dec_benchmark_projectile_', 'L', 'A', 'B'],
    'categorical': ['month', 'brand', '_type_imputed_',
                    '_july_benchmark_prev_type_imputed_', '_dec_benchmark_prev_type_imputed_',
                    '_july_benchmark_type_imputed_', '_dec_benchmark_type_imputed_']
}

model_filename = os.path.join(get_target_folder(), "xgboost_models.pkl")
hyperparameters_filename = os.path.join(get_target_folder(), "xgboost_best_hyperparameters.json")


class XGBoostWrapper:
    """
    A wrapper class for XGBoost regression model, providing utilities for training,
    prediction, cross-validation, model persistence, and hyperparameter tuning.
    """
    def __init__(self):

        self.models = {}
        self.features = features_map['numerical'] + features_map['categorical']

    @staticmethod
    def _get_callback():
        """
        Returns the early stopping callback for XGBoost training.

        Returns:
            list: A list containing an EarlyStopping callback.
        """
        return [EarlyStopping(rounds=100, metric_name='rmse')]

    def fit(self, df: pd.DataFrame, label: str, **kwargs) -> XGBRegressor:
        """
        Trains an XGBoost regressor on the given dataset.

        Args:
            df (pd.DataFrame): The input training dataset.
            label (str): The target variable column name.
            **kwargs: Hyperparameters for the model.

        Returns:
            XGBRegressor: The trained XGBoost model.
        """
        train_, eval_ = train_test_split(df, test_size=0.2, random_state=42)

        params = {"min_child_weight": kwargs.get('min_child_weight', 1),
                  "reg_alpha": kwargs.get('reg_alpha', 0),
                  "reg_lambda": kwargs.get('reg_lambda', 0),
                  "learning_rate": kwargs.get("learning_rate", 0.3),
                  "max_depth": kwargs.get("max_depth", 6),
                  "subsample": kwargs.get("subsample", 1)}

        model = XGBRegressor(random_state=42, n_estimators=3000, objective='reg:squarederror', n_jobs=-1,
                             enable_categorical=True, callbacks=self._get_callback(), eval_metric="rmse")
        model.set_params(**params)

        eval_set = [(eval_[self.features], eval_[label])]

        model.fit(train_[self.features], train_[label], eval_set=eval_set, verbose=100)

        return model

    def predict(self, df: pd.DataFrame):
        """
        Generates predictions using trained XGBoost models.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            np.ndarray: The average predicted values across models.
        """
        df[features_map['categorical']] = df[features_map['categorical']].astype("category")

        y_pred = np.zeros((len(df), len(self.models)))

        for idx, (_, model) in enumerate(self.models.items()):
            y_pred[:, idx] = model.predict(df[self.features])

        return y_pred.mean(axis=1)

    def cross_validation(self, df: pd.DataFrame, label: str, **kwargs):
        """
        Performs K-Fold cross-validation.

        Args:
            df (pd.DataFrame): The input dataset.
            label (str): The target variable column name.
            **kwargs: Hyperparameters for the model.

        Returns:
            np.ndarray: Predicted values from cross-validation.
        """
        df[features_map['categorical']] = df[features_map['categorical']].astype("category")

        print(pd.isnull(df[self.features]).any(), flush=True)

        kfold = KFold(n_splits=5, random_state=42, shuffle=True)

        y_pred = np.zeros(len(df))

        for idx, (train_index, eval_index) in enumerate(kfold.split(df)):
            data_train, data_eval = df.iloc[train_index], df.iloc[eval_index]
            self.models[idx] = self.fit(data_train, label, **kwargs)
            y_pred[eval_index] = self.models[idx].predict(data_eval[self.features])

        return y_pred

    def _save_models(self):
        """
        Saves trained models to a file using pickle.
        """
        global model_filename

        with open(model_filename, "wb") as f:
            pickle.dump(self.models, f)

    def _load_models(self):
        """
        Loads trained models from a file using pickle.
        """
        global model_filename

        with open(model_filename, "rb") as f:
            self.models = pickle.load(f)

    def _save_hyperparameters(self, study):
        """
        Saves the best hyperparameters from an Optuna study.
        """
        global hyperparameters_filename

        with open(hyperparameters_filename, "w") as f:
            json.dump(study.best_params, f)

    def load_hyperparameters(self):
        """
        Loads saved hyperparameters from a JSON file.

        Returns:
            dict: The loaded hyperparameters.
        """
        global hyperparameters_filename

        with open(hyperparameters_filename, "r") as f:
            return json.load(f)

    def objective(self, trial, df: pd.DataFrame, label: str):
        """
        Objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): The Optuna trial object.
            df (pd.DataFrame): The input dataset.
            label (str): The target variable column name.

        Returns:
            float: The RMSE score for the given trial.
        """
        params = {"min_child_weight": trial.suggest_uniform('min_child_weight', 0.5, 10),
                  "reg_alpha": trial.suggest_loguniform('reg_alpha', 1e-4, 10),
                  "reg_lambda": trial.suggest_loguniform('reg_lambda', 1e-4, 10),
                  "learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 1),
                  "max_depth": trial.suggest_int('max_depth', 3, 11),
                  "subsample": trial.suggest_uniform('subsample', 0.4, 1.0)}

        y_pred = self.cross_validation(df, label, **params)

        return root_mean_squared_error(df[label], y_pred)

    def hyperparameter_tuning(self, df: pd.DataFrame, label: str, n_trials: int = 10):

        """
        https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
        Performs hyperparameter tuning using Optuna.

        Args:
            df (pd.DataFrame): The dataset used for tuning.
            label (str): The target column name.
            n_trials (int, optional): The number of trials for optimization. Defaults to 10.
        """

        objective = partial(self.objective, df=df, label=label)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Show best hyperparameters
        print("Best hyperparameters:", study.best_params, flush=True)
        print("Best RMSE", study.best_value, flush=True)

        self._save_hyperparameters(study)