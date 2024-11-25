from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMRegressor, early_stopping, log_evaluation, LGBMClassifier


feature_engineering = {"Gender": "label_encoding",
                       "Company Type": "label_encoding",
                       "WFH Setup Available": "label_encoding",
                       "Designation": "one_hot_encoding"}

def get_callback():

    return [early_stopping(stopping_rounds=100), log_evaluation(period=100)]


def cross_validation(data: pd.DataFrame, label: str, features: list, **kwargs):

    kfold = KFold(n_splits=5, random_state=42, shuffle=True)

    y_pred = np.zeros(len(data))

    for train_index, eval_index in kfold.split(data):
        data_train, data_eval = data.iloc[train_index], data.iloc[eval_index]

        for key, value in feature_engineering.items():
            if value in ['label_encoding', 'one_hot_encoding']:
                data_eval[key] = data_eval[key].astype('category')

        model = model_training(data_train, label, features)

        y_pred[eval_index] = model.predict(data_eval)

    return y_pred


def model_training(data: pd.DataFrame, label: str, features: list) -> Union[LGBMRegressor, LGBMClassifier]:

    train_, eval_ = train_test_split(data, test_size=0.2, random_state=42)

    model = LGBMRegressor(random_state=42, n_estimators=3000, min_child_samples=5)

    categorical_feature = []

    for key, value in feature_engineering.items():
        if value in ['label_encoding', 'one_hot_encoding']:
            train_[key] = train_[key].astype('category')
            eval_[key] = eval_[key].astype('category')
            categorical_feature.append(key)

    eval_set = [(eval_[features], eval_[label])]

    model.fit(train_[features], train_[label], eval_set=eval_set, callbacks=get_callback(),
              eval_metric=['l1'], categorical_feature=categorical_feature)

    return model


def prediction(model, data):

    for key, value in feature_engineering.items():
        if value in ['label_encoding', 'one_hot_encoding']:
            data[key] = data[key].astype('category')

    return model.predict(data)