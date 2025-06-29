# 2709.5719037087615
# 0.859739711715511
import os
import pickle
import json
from functools import partial

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from src.io.path_definition import get_target_folder

features_map = {'numerical': ['_quantity_imputed_', 'y_pred', '_july_benchmark_diff_prev_',
                              '_dec_benchmark_diff_prev_', "_july_benchmark_diff_", "_dec_benchmark_diff_",
                              '_july_benchmark_', "_dec_benchmark_", "_july_benchmark_projectile_",
                              '_dec_benchmark_projectile_', 'L', 'A', 'B'],
                'categorical': ['month', 'brand', '_type_imputed_',
                                '_july_benchmark_prev_type_imputed_', '_dec_benchmark_prev_type_imputed_',
                                '_july_benchmark_type_imputed_', '_dec_benchmark_type_imputed_']}

model_filename = os.path.join(get_target_folder(), "lightgbm_models.pkl")
hyperparameters_filename = os.path.join(get_target_folder(), "lightgbm_best_hyperparameters.json")

class LightGBMWrapper:
    """
    Wrapper class for LightGBM regression model, including training, prediction,
    cross-validation, and hyperparameter tuning.
    """
    def __init__(self):
        """Initializes the LightGBMWrapper with model storage and feature lists."""
        self.models = {}
        self.features = features_map['numerical'] + features_map['categorical']

    @staticmethod
    def _get_callback():
        """Returns a list of callbacks for early stopping and logging evaluation metrics.

        Returns:
            list: List of LightGBM callbacks for early stopping and logging evaluation.
        """
        return [early_stopping(stopping_rounds=100), log_evaluation(period=100)]

    def fit(self, df: pd.DataFrame, label: str, **kwargs):
        """Trains a LightGBM model on the provided dataset.

        Args:
            df (pd.DataFrame): Input dataset containing features and target variable.
            label (str): Name of the target variable column.
            **kwargs: Additional hyperparameters for LightGBM model.

        Returns:
            LGBMRegressor: Trained LightGBM regression model.
        """
        train_, eval_ = train_test_split(df, test_size=0.2, random_state=42)

        params = {"min_child_samples": kwargs.get('min_child_samples', 1),
                  "reg_alpha": kwargs.get('reg_alpha', 0),
                  "reg_lambda": kwargs.get('reg_lambda', 0),
                  "random_state": 42,
                  "n_estimators": 3000,
                  "learning_rate": kwargs.get("learning_rate", 0.1),
                  "max_depth": kwargs.get("max_depth", -1),
                  "categorical_features": features_map['categorical'],
                  "subsample_freq": kwargs.get("subsample", 1)}

        model = LGBMRegressor()
        model.set_params(**params)

        eval_set = [(eval_[self.features], eval_[label])]

        model.fit(train_[self.features], train_[label], eval_set=eval_set, callbacks=self._get_callback(),
                  eval_metric=['l2'])

        return model

    def predict(self, df: pd.DataFrame):
        """
        Generates predictions using the trained LightGBM models.

        Args:
            df (pd.DataFrame): Dataset containing feature values.

        Returns:
            np.ndarray: Array of predicted values.
        """
        df[features_map['categorical']] = df[features_map['categorical']].astype("category")

        y_pred = np.zeros((len(df), len(self.models)))

        for idx, (_, model) in enumerate(self.models.items()):
            y_pred[:, idx] = model.predict(df[self.features])

        return y_pred.mean(axis=1)

    def cross_validation(self, df: pd.DataFrame, label: str, **kwargs):
        """
        Performs cross-validation to evaluate model performance.

        Args:
            df (pd.DataFrame): Input dataset.
            label (str): Name of the target variable column.
            **kwargs: Additional hyperparameters for training.

        Returns:
            np.ndarray: Predictions for the validation dataset.
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
        """Saves trained models to disk using pickle."""
        global model_filename

        with open(model_filename, "wb") as f:
            pickle.dump(self.models, f)

    def _load_models(self):
        """Loads trained models from disk."""
        global model_filename

        with open(model_filename, "rb") as f:
            self.models = pickle.load(f)

    def _save_hyperparameters(self, study):
        """
        Saves best hyperparameters to a JSON file.

        Args:
            study (optuna.Study): Optuna study containing best hyperparameters.
        """
        global hyperparameters_filename

        with open(hyperparameters_filename, "w") as f:
            json.dump(study.best_params, f)

    def load_hyperparameters(self):
        """
        Loads saved hyperparameters from JSON file.

        Returns:
            dict: Dictionary containing best hyperparameters.
        """
        global hyperparameters_filename

        with open(hyperparameters_filename, "r") as f:
            return json.load(f)

    def objective(self, trial, df: pd.DataFrame, label: str):
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.
            df (pd.DataFrame): Dataset for training.
            label (str): Name of the target variable column.

        Returns:
            float: Root mean squared error of the model.
        """
        params = {"min_child_samples": trial.suggest_int('min_child_samples', 3, 11),
                  "reg_alpha": trial.suggest_loguniform('reg_alpha', 1e-4, 10),
                  "reg_lambda": trial.suggest_loguniform('reg_lambda', 1e-4, 10),
                  "learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 1),
                  "max_depth": trial.suggest_int('max_depth', 3, 7),
                  "subsample_freq": trial.suggest_uniform('subsample_freq', 0.4, 1.0)}

        y_pred = self.cross_validation(df, label, **params)

        return root_mean_squared_error(df[label], y_pred)

    def hyperparameter_tuning(self, df: pd.DataFrame, label: str, n_trials: int = 10):

        """
        Performs hyperparameter tuning using Optuna.
        https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

        Args:
            df (pd.DataFrame): Input dataset.
            label (str): Name of the target variable column.
            n_trials (int, optional): Number of trials for optimization. Defaults to 10.
        """

        objective = partial(self.objective, df=df, label=label)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Show best hyperparameters
        print("Best hyperparameters:", study.best_params, flush=True)
        print("Best RMSE", study.best_value, flush=True)

        self._save_hyperparameters(study)