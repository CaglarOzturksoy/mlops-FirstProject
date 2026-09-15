"""
Unit tests for model_dev.py

Run:
    pytest tests/test_model_dev.py -v
"""

import numpy as np
import optuna
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

from lightgbm import LGBMClassifier

from src.model_dev import (
    Model,
    RandomForestModel,
    XGBModel,
    LightGBMModel,
    LogisticRegressionModel,
    HyperParameterTuner,
)


# Optuna loglarını sessize al
optuna.logging.set_verbosity(logging.WARNING)


def make_small_dataset(
    n_samples=60,
    n_features=6,
    random_state=42,
):
    """
    Hızlı model testleri için küçük ve dengeli
    bir binary classification dataset oluşturur.
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val


class FakeTrial:
    """
    Optuna Trial nesnesinin küçük bir taklidi.

    Model optimize() fonksiyonlarının gerçekten
    çalışıp çalışmadığını test etmek için kullanılır.
    """

    def __init__(
        self,
        int_overrides=None,
        float_overrides=None,
    ):
        self.int_overrides = int_overrides or {}
        self.float_overrides = float_overrides or {}

    def suggest_int(self, name, low, high):
        return self.int_overrides.get(name, low)

    def suggest_float(self, name, low, high, log=False):
        return self.float_overrides.get(name, low)


# =========================================================
# Model Base Class
# =========================================================


def test_model_cannot_be_instantiated():
    """Abstract Model sınıfı doğrudan oluşturulamamalı."""

    try:
        Model() #type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError(
            "Abstract Model class should not be instantiable."
        )


def test_model_subclasses_are_model_instances():
    """Model sınıflarının Model'den türediğini kontrol eder."""

    model_classes = (
        RandomForestModel,
        XGBModel,
        LightGBMModel,
        LogisticRegressionModel,
    )

    for model_class in model_classes:
        assert isinstance(model_class(), Model)


# =========================================================
# Random Forest
# =========================================================


def test_random_forest_train_returns_fitted_classifier():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = RandomForestModel()

    model = model_wrapper.train(
        X_train,
        y_train,
        n_estimators=5,
        max_depth=3,
    )

    assert isinstance(model, ClassifierMixin)

    predictions = model.predict(X_val)

    assert len(predictions) == len(X_val)


def test_random_forest_optimize_returns_valid_f1():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = RandomForestModel()

    trial = FakeTrial(
        int_overrides={
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_split": 2,
        }
    )

    score = model_wrapper.optimize(
        trial,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# =========================================================
# XGBoost
# =========================================================


def test_xgboost_train_returns_fitted_classifier():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = XGBModel()

    model = model_wrapper.train(
        X_train,
        y_train,
        n_estimators=5,
        max_depth=2,
    )

    predictions = model.predict(X_val)

    assert len(predictions) == len(X_val)


def test_xgboost_optimize_returns_valid_f1():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = XGBModel()

    trial = FakeTrial(
        int_overrides={
            "n_estimators": 5,
            "max_depth": 2,
        },
        float_overrides={
            "learning_rate": 0.1,
        },
    )

    score = model_wrapper.optimize(
        trial,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# =========================================================
# LightGBM
# =========================================================


def test_lightgbm_train_returns_fitted_classifier():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = LightGBMModel()

    model = model_wrapper.train(
        X_train,
        y_train,
        n_estimators=5,
        max_depth=2,
        verbosity=-1,
    )

    assert isinstance(model, LGBMClassifier)

    predictions = model.predict(X_val)

    assert len(predictions) == len(X_val)


def test_lightgbm_optimize_returns_valid_f1():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = LightGBMModel()

    trial = FakeTrial(
        int_overrides={
            "n_estimators": 5,
            "max_depth": 2,
        },
        float_overrides={
            "learning_rate": 0.1,
        },
    )

    score = model_wrapper.optimize(
        trial,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# =========================================================
# Logistic Regression
# =========================================================


def test_logistic_regression_train_returns_fitted_classifier():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = LogisticRegressionModel()

    model = model_wrapper.train(
        X_train,
        y_train,
        max_iter=1000,
    )

    assert isinstance(model, ClassifierMixin)

    predictions = model.predict(X_val)

    assert len(predictions) == len(X_val)


def test_logistic_regression_optimize_returns_valid_f1():

    X_train, X_val, y_train, y_val = make_small_dataset()

    model_wrapper = LogisticRegressionModel()

    trial = FakeTrial()

    score = model_wrapper.optimize(
        trial,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# =========================================================
# HyperParameterTuner
# =========================================================


def test_hyperparameter_tuner_with_logistic_regression():

    X_train, X_val, y_train, y_val = make_small_dataset()

    tuner = HyperParameterTuner(
        LogisticRegressionModel(),
        X_train,
        y_train,
        X_val,
        y_val,
    )

    best_params = tuner.optimize(n_trials=2)

    assert isinstance(best_params, dict)

    # LogisticRegression optimize() herhangi bir
    # hyperparameter önermediği için params boş olmalı.
    assert best_params == {}


def test_hyperparameter_tuner_with_random_forest():

    X_train, X_val, y_train, y_val = make_small_dataset()

    tuner = HyperParameterTuner(
        RandomForestModel(),
        X_train,
        y_train,
        X_val,
        y_val,
    )

    best_params = tuner.optimize(n_trials=1)

    assert isinstance(best_params, dict)

    expected_keys = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
    }

    assert set(best_params.keys()) == expected_keys

    assert 1 <= best_params["n_estimators"] <= 500
    assert 1 <= best_params["max_depth"] <= 20
    assert 2 <= best_params["min_samples_split"] <= 32