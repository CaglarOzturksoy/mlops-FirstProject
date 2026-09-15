import logging
import mlflow
import pandas as pd
import numpy as np
from typing import Any

from zenml import step
from zenml.client import Client

from .config import ModelNameConfig

from src.model_dev import (
    HyperParameterTuner,
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBModel)

mlflow.autolog()
def get_experiment_tracker_name():
    try:
        tracker = Client().active_stack.experiment_tracker
        return tracker.name if tracker else None
    except Exception:
        return None

@step(experiment_tracker=get_experiment_tracker_name())
def train_model(X_train:np.ndarray,
                X_val:np.ndarray,
                y_train:pd.Series,
                y_val:pd.Series,
                config:ModelNameConfig) -> Any:

    try:

        if config.model_name == 'random_forest':
            model = RandomForestModel()

        elif config.model_name == 'lightgbm':
            model = LightGBMModel()

        elif config.model_name == 'xgboost':
            model = XGBModel()

        elif config.model_name == 'logistic_regression':
            model = LogisticRegressionModel()

        else:
            raise ValueError(f'Unsupported model: {config.model_name}')

        tuner = HyperParameterTuner(model=model,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_val=X_val,
                                    y_val=y_val)

        if config.fine_tuning:
            best_params = tuner.optimize(n_trials=200)
            logging.info(f'Best hyperparameters: {best_params}')

            trained_model = model.train(X_train,y_train,**best_params)
            logging.info(f'Best parameters: {best_params}')

        else:
            trained_model = model.train(X_train,y_train)

        return trained_model

    except Exception as e:
        logging.error(f'model training error {e}')
        raise