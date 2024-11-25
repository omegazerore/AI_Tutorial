import inspect

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split



def meta_model_training(base_models: list, meta_model, data: pd.DataFrame, features: list, label: str = 'label'):

    kfold = KFold(n_splits=5, random_state=42, shuffle=True)

    y_pred_meta = np.zeros((len(data), len(base_models)))

    for train_index, eval_index in kfold.split(data):
        data_train, data_eval = data.iloc[train_index], data.iloc[eval_index]

        for idx, model in enumerate(base_models):
            # To extend the flexibility of the model, we need the following extension
            """
            import importlib
            module = importlib.import_module(f'train.logic.{task}_classification.training_process_{algorithm}')
            process = getattr(module, 'process')
            """
            # And send the model training to the process function
            if 'callbacks' in inspect.signature(model.fit).parameters:
                train_, eval_ = train_test_split(data_train, test_size=0.2, random_state=42)

                eval_set = [(eval_[features], eval_[label])]
                callbacks = [early_stopping(stopping_rounds=100), log_evaluation(period=100)]
                model.fit(train_[features], train_[label], eval_set=eval_set, callbacks=callbacks,
                          eval_metric=['l1'])
                y_pred_meta[eval_index, idx] = model.predict(data_eval[features])
            else:
                resource_allocation_mean = data_train['Resource Allocation'].mean()
                mental_fatigue_score_mean = data_train['Mental Fatigue Score'].mean()

                data_train['Resource Allocation'].fillna(resource_allocation_mean, inplace=True)
                data_train['Mental Fatigue Score'].fillna(mental_fatigue_score_mean, inplace=True)
                model.fit(data_train[features], data_train[label])

                data_eval['Resource Allocation'].fillna(resource_allocation_mean, inplace=True)
                data_eval['Mental Fatigue Score'].fillna(mental_fatigue_score_mean, inplace=True)

                y_pred_meta[eval_index, idx] = model.predict(data_eval[features])



    for train_index, eval_index in kfold.split(y_pred_meta):
        data_train, data_eval = y_pred_meta[train_index], y_pred_meta[eval_index]

        meta_model.fit(data_train, data.iloc[train_index,label])


        meta_model.predict(data_eval)
