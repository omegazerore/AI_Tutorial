import argparse
import os
from datetime import datetime

import pandas as pd

import src.logic.timeseries as ts
from src.io.path_definition import get_project_dir, get_datafetch
from src.logic.timeseries.NHiTS_forecaster import NHiTSForecaster
from src.logic.timeseries.NBEATS_forecaster import NBeatsForecaster
from src.logic.timeseries.TFT_forecaster import TFTForecaster

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default='AdamW', help="The neural network optimizer")
    parser.add_argument('--max_epochs', type=int, default='20')
    parser.add_argument('--max_prediction_length', type=int, default=12)
    parser.add_argument('--max_encoder_length', type=int, default=36)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--group_ids', type=str, nargs='+')
    parser.add_argument('--n_trials', type=int, default=5)

    args = parser.parse_args()

    ts.MAX_ENCODER_LENGTH = args.max_encoder_length
    ts.BATCH_SIZE = args.batch_size
    ts.OPTIMIZER_TYPE = args.optimizer
    ts.MAX_EPOCHS = args.max_epochs
    ts.MAX_PREDICTION_LENGTH = args.max_prediction_length
    ts.GROUP_IDS = args.group_ids
    n_trials = args.n_trials

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

    # train-test split
    cutoff = df_agg[ts.TIME_IDX].max() - ts.MAX_PREDICTION_LENGTH
    df_train = df_agg[df_agg[ts.TIME_IDX] <= cutoff]

    # 1. N-Beats
    nbeats = NBeatsForecaster(args=args)

    nbeats.optimize_hyperparameters(df=df_train, n_trials=n_trials)

    params = nbeats.load_hyperparameters()

    nbeats.fit(df=df_train, validation=True, **params)
    nbeats.fit(df=df_train, validation=False, **params)

    df_prediction = nbeats.predict(df=df_agg, plot=True)

    final_df = pd.merge(df_agg, df_prediction, on=[ts.TIME_IDX, 'retailer', 'brand'], how='left')

    final_df.to_csv(os.path.join(get_datafetch(), "sellout_forecast_NBeats.csv"))

    #2. N-HiTS
    nhits = NHiTSForecaster(args=args)

    nhits.optimize_hyperparameters(df=df_train, n_trials=n_trials)

    params = nhits.load_hyperparameters()

    nhits.fit(df=df_train, validation=True, **params)
    nhits.fit(df=df_train, validation=False, **params)

    df_prediction = nhits.predict(df=df_agg, plot=True)

    final_df = pd.merge(df_agg, df_prediction, on=[ts.TIME_IDX, 'retailer', 'brand'], how='left')

    final_df.to_csv(os.path.join(get_datafetch(), "sellout_forecast_NHiTS.csv"))

    #3. TFT
    tft = TFTForecaster(args=args)

    tft.optimize_hyperparameters(df=df_train, n_trials=n_trials)

    params = tft.load_hyperparameters()

    tft.fit(df=df_train, validation=True, **params)
    tft.fit(df=df_train, validation=False, **params)

    df_prediction = tft.predict(df=df_agg, plot=True)

    final_df = pd.merge(df_agg, df_prediction, on=[ts.TIME_IDX, 'retailer', 'brand'], how='left')

    final_df.to_csv(os.path.join(get_datafetch(), "sellout_forecast_TFT.csv"))