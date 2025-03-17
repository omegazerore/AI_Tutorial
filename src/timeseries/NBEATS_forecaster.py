"""
Initialize NBeats Model - use its :py:meth:`~from_dataset` method if possible.

Based on the article
`N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting <http://arxiv.org/abs/1905.10437>`_. The network has (if used as ensemble) outperformed all
other methods
including ensembles of traditional statical methods in the M4 competition. The M4 competition is arguably
the most
important benchmark for univariate time series forecasting.

The :py:class:`~pytorch_forecasting.models.nhits.NHiTS` network has recently shown to consistently outperform
N-BEATS.

Args:
    stack_types: One of the following values: “generic”, “seasonality" or “trend". A list of strings
        of length 1 or ‘num_stacks’. Default and recommended value
        for generic mode: [“generic”] Recommended value for interpretable mode: [“trend”,”seasonality”]
    num_blocks: The number of blocks per stack. A list of ints of length 1 or ‘num_stacks’.
        Default and recommended value for generic mode: [1] Recommended value for interpretable mode: [3]
    num_block_layers: Number of fully connected layers with ReLu activation per block. A list of ints of length
        1 or ‘num_stacks’.
        Default and recommended value for generic mode: [4] Recommended value for interpretable mode: [4]
    width: Widths of the fully connected layers with ReLu activation in the blocks.
        A list of ints of length 1 or ‘num_stacks’. Default and recommended value for generic mode: [512]
        Recommended value for interpretable mode: [256, 2048]
    sharing: Whether the weights are shared with the other blocks per stack.
        A list of ints of length 1 or ‘num_stacks’. Default and recommended value for generic mode: [False]
        Recommended value for interpretable mode: [True]
    expansion_coefficient_length: If the type is “G” (generic), then the length of the expansion
        coefficient.
        If type is “T” (trend), then it corresponds to the degree of the polynomial. If the type is “S”
        (seasonal) then this is the minimum period allowed, e.g. 2 for changes every timestep.
        A list of ints of length 1 or ‘num_stacks’. Default value for generic mode: [32] Recommended value for
        interpretable mode: [3]
    prediction_length: Length of the prediction. Also known as 'horizon'.
    context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
        Should be between 1-10 times the prediction length.
    backcast_loss_ratio: weight of backcast in comparison to forecast when calculating the loss.
        A weight of 1.0 means that forecast and backcast loss is weighted the same (regardless of backcast and
        forecast lengths). Defaults to 0.0, i.e. no weight.
    loss: loss to optimize. Defaults to MASE().
    log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
        failures
    reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
    logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
        Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
    **kwargs: additional arguments to :py:class:`~BaseModel`.
"""

import os
import logging
import warnings
from argparse import Namespace
from typing import Optional
from functools import partial

import optuna
import numpy as np
import pandas as pd
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.models.base_model import Prediction
from pytorch_forecasting.metrics import QuantileLoss

from src import timeseries as ts
from src.io.path_definition import get_datafetch
from src.timeseries.base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBeatsForecaster(BaseForecaster):
    """Class for training and making predictions using Temporal Fusion Transformer."""

    MODEL_FOLDER = os.path.join(get_datafetch(), 'model', 'N-Beats')
    NAME = 'N-Beats'

    def __init__(self, args: Namespace, filename: Optional[str]=None):

        ts.OPTIMIZER_TYPE = args.optimizer # Define optimizer as a constant for clarity
        ts.DEFAULT_MAX_EPOCHS = args.max_epochs
        ts.MAX_PREDICTION_LENGTH = args.max_prediction_length
        ts.MAX_ENCODER_LENGTH = args.max_encoder_length
        ts.BATCH_SIZE = args.batch_size
        ts.GROUP_IDS = args.group_ids
        ts.TIME_VARYING_UNKNOWN_REALS = [ts.TARGET]

        super().__init__(filename=filename)  # Call the BaseForecaster constructor if needed

    def fit(self, df: pd.DataFrame, validation: bool, **kwargs):
        """Fits the Temporal Fusion Transformer model.

        Args:
            df (pd.DataFrame): Training dataset.
            validation (bool): Whether to use validation data.
            **kwargs: Additional hyperparameters for model training.
        """
        model_filename = os.path.join(self.MODEL_FOLDER, f'{self._filename}.ckpt')

        if os.path.isfile(model_filename):
            os.remove(model_filename)

        assert ts.TIME_IDX in df.columns, f"{ts.TIME_IDX} does not exist in df"
        assert ts.TARGET in df.columns, f"{ts.TARGET} does not exist in df"

        self._learning_rate = kwargs.get('learning_rate', 0.03)
        self._width_a = kwargs.get('width_a', 256)
        self._width_b = kwargs.get('width_b', 2048)
        self._dropout = kwargs.get("dropout", 0.1)
        self._gradient_clip_val = kwargs.get("gradient_clip_val", 0.1)
        self._weight_decay = kwargs.get("weight_decay", 0.01)
        self._backcast_loss_ratio = kwargs.get("backcast_loss_ratio", 1)

        # Call parent `fit` method to handle preprocessing
        self._assert(df, validation)

        self._pipeline()

    def _pipeline(self):
        """Executes the end-to-end pipeline for model training.

        This function:
            - Loads the training configuration.
            - Builds training and validation dataloaders.
            - Initializes the PyTorch Lightning Trainer.
            - Builds the Temporal Fusion Transformer (TFT) model.
            - Starts the model training process.
            - Determines the optimized number of epochs if not pre-defined.

        Logs:
            - Pipeline execution steps for better debugging and monitoring.

        Returns:
            None: This method does not return anything; it sets up and trains the model.
        """
        logger.info("Starting pipeline execution.")

        # Load training configuration
        self._load_training_config()
        logger.info("Training configuration loaded.")

        # Build train and validation dataloaders
        if self._validation:
            train_dataloaders, val_dataloaders = self._build_train_val_dataloader(self._training, self._df,
                                                                                  predict=False)
        else:
            train_dataloaders = self._training.to_dataloader(train=True, batch_size=ts.BATCH_SIZE, num_workers=0)
            val_dataloaders = None

        logger.info("Train and validation dataloaders created.")

        # Initialize Trainer
        self._build_trainer()
        logger.info("Trainer initialized.")

        # Build NBeats model
        self._build_model()
        logger.info("N-Beats model is built.")

        # Train the model
        logger.info("Starting model training...")
        self._trainer.fit(self._model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
        logger.info("Model training completed.")

        self._get_optimized_epoch()

        logger.info("Pipeline execution completed successfully.")

    def _load_training_config(self):
        """Loads and configures the training dataset.

        This method:
        - Ensures the required columns exist in `self._df`.
        - Converts `ts.TIME_IDX` to integers for consistency.
        - Defines a training dataset using `TimeSeriesDataSet`.

        Raises:
            AssertionError: If any required column is missing in `self._df`.
            KeyError: If `ts.TIME_IDX` column is missing while trying to convert its type.
        """

        required_columns = {"brand", "retailer", ts.TIME_IDX}
        missing_columns = required_columns - set(self._df.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns in self._df: {missing_columns}")

        self._df[ts.TIME_IDX] = self._df[ts.TIME_IDX].astype(int)

        if self._validation:
            cutoff = self._df[ts.TIME_IDX].max() - ts.MAX_PREDICTION_LENGTH
        else:
            cutoff = self._df[ts.TIME_IDX].max()

        self._training = TimeSeriesDataSet(
            self._df.loc[lambda x: x.time_idx <= cutoff],
            time_idx=ts.TIME_IDX,
            target=ts.TARGET,
            group_ids=ts.GROUP_IDS,
            max_encoder_length=ts.MAX_ENCODER_LENGTH,
            max_prediction_length=ts.MAX_PREDICTION_LENGTH,
            time_varying_unknown_reals=ts.TIME_VARYING_UNKNOWN_REALS,
            # categorical_encoders=ts.CATEGORICAL_ENCODERS,
        )

    def predict(self, df: pd.DataFrame, filename: Optional[str]=None, plot: bool=False) -> pd.DataFrame:
        """Makes predictions using the trained Temporal Fusion Transformer model.

        Args:
            df (pd.DataFrame): Input dataframe for prediction.

        Returns:
            pd.DataFrame: Dataframe containing predictions.
        """

        if filename is None:
            model_filename = os.path.join(self.MODEL_FOLDER, f'best_model.ckpt')
        else:
            model_filename = filename
        assert os.path.isfile(model_filename), f"{model_filename} does not exist."
        self._load_best_model(model_filename)

        self._df = df
        self._load_training_config()
        _, predict_dataloader = self._build_train_val_dataloader(self._training, df, predict=True)

        predictions = self.best_model.predict(predict_dataloader,
                                              trainer_kwargs=dict(accelerator="cpu"),
                                              return_index=True)

        return self.prediction2dataframe(predictions).dropna(subset=['y_pred'])

    def _load_best_model(self, filename: str):
        """Loads the best trained model from checkpoint.

        Args:
            filename (str): Path to the model checkpoint.
        """
        self.best_model = NBeats.load_from_checkpoint(filename)

    @staticmethod
    def prediction2dataframe(predictions: Prediction) -> pd.DataFrame:
        """Converts predictions into a structured dataframe.

        Args:
            predictions (Prediction): Model predictions.

        Returns:
            pd.DataFrame: Dataframe containing structured predictions.
        """
        data = []

        for idx, row in predictions.index.iterrows():
            time_idx = row[ts.TIME_IDX]
            brand = row['brand']
            retailer = row['retailer']
            prediction = predictions[0][idx]
            for idy, pred in enumerate(prediction):
                data.append([brand, retailer, time_idx + idy, float(pred)])

        return pd.DataFrame(data, columns=['brand', 'retailer', ts.TIME_IDX, 'y_pred'])

    def _build_model(self):

        net = NBeats.from_dataset(
            self._training,
            stack_types=['trend', 'seasonality'],
            learning_rate=self._learning_rate,
            weight_decay=self._weight_decay,
            widths=[self._width_a, self._width_b],
            backcast_loss_ratio=self._backcast_loss_ratio,
            dropout=self._dropout,
            optimizer=ts.OPTIMIZER_TYPE,
            # loss=QuantileLoss()
        )

        logger.info(f"Number of parameters in network: {net.size() / 1e3:.1f}k")

        self._model = net

    def objective(self, trial: optuna.Trial, df: pd.DataFrame):

        params = {"learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                  # "width_a":  trial.suggest_int('width_a', 64, 512),
                  # "width_b": trial.suggest_int('width_b', 768, 4096),
                  "dropout": trial.suggest_uniform("dropout", 0, 0.3),
                  # "gradient_clip_val": trial.suggest_uniform("gradient_clip_val", 0.05, 0.3),
                  "weight_decay": trial.suggest_loguniform("weight_decay", 1e-4, 1e-2),
                  "backcast_loss_ratio": trial.suggest_uniform("backcast_loss_ratio", 0, 1)}

        self._validation = True

        try:
            self.fit(df, validation=True, **params)
        except RuntimeError as e:
            logger.info(e)
            return np.inf

        self.optimized_epoch = None

        best_score = float(self._trainer.early_stopping_callbacks[0].state_dict()['best_score'])

        return best_score

    def optimize_hyperparameters(self, df: pd.DataFrame, n_trials: int = 20):

        objective = partial(self.objective, df=df)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Show best hyperparameters
        logger.info("Best hyperparameters:", study.best_params)
        logger.info("Best score", study.best_value)

        self._save_hyperparameters(study)


if __name__ == "__main__":

    import argparse

    from datetime import datetime

    from src.io.path_definition import get_project_dir

    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default='AdamW', help="The neural network optimizer")
    parser.add_argument('--max_epochs', type=int, default='20')
    parser.add_argument('--max_prediction_length', type=int, default=12)
    parser.add_argument('--max_encoder_length', type=int, default=36)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--group_ids', type=str, nargs='+')

    args = parser.parse_args()

    # ts.CATEGORICAL_ENCODERS = {"brand": NaNLabelEncoder(),
    #                            "retailer": NaNLabelEncoder()}

    filename = os.path.join(get_project_dir(), 'src', 'timeseries', 'sellout_experiement.csv')

    df = pd.read_csv(filename, index_col=0, dtype={"quantity": float})

    df = df[(df['country_iso'] == 'DE')]
    df_agg = df.groupby(['retailer', "month", "year", "brand"]).agg(y=("quantity", "sum"))
    df_agg.reset_index(inplace=True)

    df_agg['time'] = df_agg[['year', 'month']].apply(lambda x: datetime.strptime(f"{x[0]}-{x[1]}", "%Y-%m"),
                                                     axis=1)

    df_agg[ts.TIME_IDX] = df_agg[['year', 'month']].apply(lambda x: 12 * x[0] + x[1], axis=1)
    df_agg[ts.TIME_IDX] = df_agg[ts.TIME_IDX] - df_agg[ts.TIME_IDX].min()
    df_agg[ts.TIME_IDX] = df_agg[ts.TIME_IDX].astype(int)

    df_agg = df_agg[df_agg['brand'].isin(['Catrice', 'essence'])]

    nbeats = NBeatsForecaster(args=args)

    # train-test split
    cutoff = df_agg[ts.TIME_IDX].max() - ts.MAX_PREDICTION_LENGTH

    df_train = df_agg[df_agg[ts.TIME_IDX] <= cutoff]

    # nbeats.optimize_hyperparameters(df=df_train, n_trials=5)

    params = nbeats.load_hyperparameters()

    nbeats.fit(df=df_train, validation=True, **params)
    nbeats.fit(df=df_train, validation=False, **params)

    df_prediction = nbeats.predict(df=df_agg, plot=True)
    #
    # final_df = pd.merge(df_agg, df_prediction, on=[ts.TIME_IDX, 'retailer', 'brand'], how='left')
    #
    # final_df.to_csv(os.path.join(get_datafetch(), "sellout_forecast_NBeats.csv"))

