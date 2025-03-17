import logging
import json
import os
import pickle
from typing import Optional

import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.base_model import Prediction
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src import timeseries as ts
from src.io.path_definition import get_datafetch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseForecaster:

    def __init__(self, filename: Optional[str] = None):

        if filename is None:
            self._filename = 'best_model'
        else:
            self._filename = filename
        self.best_model = None
        self.optimized_epoch = None

    def fit(self, df: pd.DataFrame, validation: bool, **kwargs):

        pass

    def _assert(self, df: pd.DataFrame, validation:bool):

        assert ts.TIME_IDX in df.columns, f"{ts.TIME_IDX} does not exist in df"
        assert ts.TARGET in df.columns, f"{ts.TARGET} does not exist in df"

        self._validation = validation
        df[ts.TIME_IDX] = df[ts.TIME_IDX].astype(int)
        self._df = df
        self._target = ts.TARGET

    def _pipeline(self):

        pass

    def _load_training_config(self):

        pass

    @staticmethod
    def _build_train_val_dataloader(training: TimeSeriesDataSet, df: pd.DataFrame, predict: bool):
        """Builds training and validation dataloaders.

        Args:
            training (TimeSeriesDataSet): Training dataset.
            df (pd.DataFrame): Original dataframe.

        Returns:
            Tuple: Training and validation dataloaders.
        """
        train_dataloader = training.to_dataloader(train=True, batch_size=ts.BATCH_SIZE, num_workers=0)
        cutoff = df[ts.TIME_IDX].max() - ts.MAX_PREDICTION_LENGTH
        validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=cutoff + 1, predict=predict)
        val_dataloader = validation.to_dataloader(train=False, batch_size=ts.BATCH_SIZE, num_workers=0)
        return train_dataloader, val_dataloader

    def _build_trainer(self):
        """Builds the PyTorch Lightning Trainer with necessary callbacks and logging.

        This function initializes a `pl.Trainer` instance with various callbacks and configurations.
        It also sets up early stopping, learning rate monitoring, model checkpointing, and logging.

        Features:
            - **Early Stopping**: Monitors `val_loss`, stops training if no improvement.
            - **Learning Rate Monitor**: Logs learning rate during training.
            - **Model Checkpointing**: Saves the model with a specified filename.
            - **TensorBoard Logger**: Logs training details for visualization.

        Trainer Settings:
            - Uses CPU acceleration (modify as needed).
            - Supports gradient clipping.
            - Custom max epochs based on `self.optimized_epoch`.

        Logs:
            - Configured training settings.

        Returns:
            None: This method initializes `self._trainer`.
        """
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode="min"
        )
        lr_logger = LearningRateMonitor()  # log the learning rate
        model_checkpoint = ModelCheckpoint(dirpath=self.MODEL_FOLDER,
                                           filename=self._filename,
                                           save_top_k=1)
        tensorboard_logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        # Select appropriate callbacks based on validation usage
        callbacks = [lr_logger, early_stop_callback, model_checkpoint] if self._validation else [lr_logger,
                                                                                                 model_checkpoint]
        # Determine the max number of training epochs
        max_epochs = self.optimized_epoch if self.optimized_epoch is not None else ts.DEFAULT_MAX_EPOCHS

        # Initialize PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            enable_model_summary=True,
            gradient_clip_val=self._gradient_clip_val,
            # limit_train_batches=50,  # comment in for training, running valiation every 50 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=callbacks,
            logger=tensorboard_logger,
        )

        self._trainer = trainer

        # Log trainer initialization
        logger.info(f"Trainer initialized with max_epochs={max_epochs}, gradient_clip_val={self._gradient_clip_val}.")

    def _build_model(self):

        pass

    def _get_optimized_epoch(self):

        # Determine optimized epochs if not pre-defined
        if self.optimized_epoch is None:
            try:
                stopped_epoch = max(self._trainer.early_stopping_callbacks[0].state_dict()['stopped_epoch'],
                                    ts.DEFAULT_MAX_EPOCHS)
                patience = self._trainer.early_stopping_callbacks[0].state_dict()['patience']
                self.optimized_epoch = int(stopped_epoch - patience)
                logger.info(f"Optimized epoch determined: {self.optimized_epoch}")
            except (IndexError, KeyError, AttributeError) as e:
                logger.warning(f"Failed to determine optimized epoch: {e}")

    def _load_best_model(self, filename: str):

        pass

    def optimize_hyperparameters(self, df: pd.DataFrame, n_trials: int = 20):

        pass

    def load_hyperparameters(self):

        hyperparameters_filename = os.path.join(get_datafetch(), f"{self.NAME}_best_hyperparameters.json")

        with open(hyperparameters_filename, "r") as fout:
            params = json.load(fout)

        return params

    def predict(self, df: pd.DataFrame, filename: Optional[str]=None, plot: bool=False):

        pass

    @staticmethod
    def prediction2dataframe(predictions: Prediction) -> pd.DataFrame:
        pass

    def _save_hyperparameters(self, study):

        hyperparameters_filename = os.path.join(get_datafetch(), f"{self.NAME}_best_hyperparameters.json")

        with open(hyperparameters_filename, "w") as f:
            json.dump(study.best_params, f)
