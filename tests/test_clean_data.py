"""
Unit tests for steps/clean_data.py (clean_df ZenML step)

Run with:
    pytest tests/test_clean_data.py -v
"""

import numpy as np
import pandas as pd
import pytest

from steps.clean_data import clean_df


@pytest.fixture
def sample_df():
    """
    20 satırlık, dengeli ve Spaceship Titanic veri setiyle
    aynı sütun yapısına sahip örnek veri oluşturur.
    """
    n = 20

    return pd.DataFrame({
        "PassengerId": [f"{i:04d}_01" for i in range(1, n + 1)],

        "HomePlanet": [
            "Earth", "Europa", "Mars", "Earth", "Europa",
            "Mars", "Earth", "Europa", "Mars", "Earth",
            "Europa", "Mars", "Earth", "Europa", "Mars",
            "Earth", "Europa", "Mars", "Earth", "Europa",
        ],

        "CryoSleep": [False, True] * 10,

        "Cabin": [
            "B/0/P", "F/1/S", "A/2/P", "G/3/S", "C/4/P",
            np.nan, "D/5/S", "E/6/P", "F/7/S", "G/8/P",
            "A/9/P", "B/10/S", "C/11/P", "D/12/S", "E/13/P",
            "F/14/S", "G/15/P", "A/16/S", "B/17/P", "C/18/S",
        ],

        "Destination": [
            "TRAPPIST-1e", "55 Cancri e", "TRAPPIST-1e",
            "PSO J318.5-22", "TRAPPIST-1e", "55 Cancri e",
            "PSO J318.5-22", "TRAPPIST-1e", "55 Cancri e",
            "TRAPPIST-1e", "TRAPPIST-1e", "55 Cancri e",
            "TRAPPIST-1e", "PSO J318.5-22", "TRAPPIST-1e",
            "55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e",
            "55 Cancri e", "TRAPPIST-1e",
        ],

        "Age": [
            39, 24, 58, 33, 21,
            45, 30, 27, 19, 40,
            35, 28, 50, 22, 33,
            41, 19, 60, 25, 38,
        ],

        "VIP": [False] * n,

        "RoomService": [
            0, 100, 50, 0, 20,
            30, 0, 10, 50, 0,
            5, 0, 80, 0, 15,
            0, 0, 60, 0, 10,
        ],

        "FoodCourt": [
            0, 9, 3576, 0, 15,
            0, 0, 20, 5, 0,
            0, 0, 50, 0, 0,
            0, 0, 100, 0, 0,
        ],

        "ShoppingMall": [
            0, 25, 0, 0, 10,
            0, 0, 5, 0, 0,
            0, 0, 20, 0, 0,
            0, 0, 15, 0, 0,
        ],

        "Spa": [
            0, 549, 6715, 0, 30,
            0, 0, 100, 0, 0,
            0, 0, 200, 0, 0,
            0, 0, 300, 0, 0,
        ],

        "VRDeck": [
            0, 44, 49, 0, 5,
            0, 0, 15, 0, 0,
            0, 0, 30, 0, 0,
            0, 0, 40, 0, 0,
        ],

        "Name": [f"Person {i}" for i in range(1, n + 1)],

        "Transported": [True, False] * 10,
    })


def test_returns_six_outputs(sample_df):
    result = clean_df.entrypoint(sample_df)

    assert len(result) == 6


def test_X_outputs_are_numpy_arrays(sample_df):
    X_train, X_val, X_test, y_train, y_val, y_test = (
        clean_df.entrypoint(sample_df)
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(X_test, np.ndarray)


def test_y_outputs_are_pandas_series(sample_df):
    X_train, X_val, X_test, y_train, y_val, y_test = (
        clean_df.entrypoint(sample_df)
    )

    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_no_split_is_empty(sample_df):
    """
    Train, validation ve test splitlerinin boş olmadığını kontrol eder.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = (
        clean_df.entrypoint(sample_df)
    )

    assert X_train.size > 0
    assert X_val.size > 0
    assert X_test.size > 0

    assert len(y_train) > 0
    assert len(y_val) > 0
    assert len(y_test) > 0


def test_row_counts_match_between_X_and_y(sample_df):
    X_train, X_val, X_test, y_train, y_val, y_test = (
        clean_df.entrypoint(sample_df)
    )

    assert X_train.shape[0] == len(y_train)
    assert X_val.shape[0] == len(y_val)
    assert X_test.shape[0] == len(y_test)


def test_total_rows_preserved(sample_df):
    X_train, X_val, X_test, y_train, y_val, y_test = (
        clean_df.entrypoint(sample_df)
    )

    total = (
        X_train.shape[0]
        + X_val.shape[0]
        + X_test.shape[0]
    )

    assert total == len(sample_df)


def test_same_feature_count_across_splits(sample_df):
    """
    Train üzerinde fit edilen preprocessing pipeline'ın
    validation ve test üzerinde de aynı feature sayısını
    üretmesini kontrol eder.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = (
        clean_df.entrypoint(sample_df)
    )

    assert X_train.shape[1] == X_val.shape[1]
    assert X_train.shape[1] == X_test.shape[1]


def test_no_nans_in_transformed_features(sample_df):
    """
    SimpleImputer sonrasında transformed feature'larda
    NaN kalmadığını kontrol eder.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = (
        clean_df.entrypoint(sample_df)
    )

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()


def test_missing_transported_column_raises(sample_df):
    bad_df = sample_df.drop(columns=["Transported"])

    with pytest.raises(Exception):
        clean_df.entrypoint(bad_df)


def test_polynomial_features_disabled(sample_df):
    """
    clean_df step'inin use_polynomial=False kullandığını
    dolaylı olarak kontrol eder.
    """

    from src.data_cleaning import (
        DataPreprocessStrategy,
        DataDivideStrategy,
        SklearnPipeline,
    )

    # Önce preprocessing
    preprocessed = DataPreprocessStrategy().handle_data(sample_df)

    # Train / validation / test ayrımı
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = DataDivideStrategy().handle_data(preprocessed)

    # Polynomial olmadan
    (
        X_no_poly,
        _,
        _,
        _,
    ) = SklearnPipeline(
        use_scaling=True,
        use_polynomial=False,
    ).handle_data(
        X_train,
        X_val,
        X_test,
    )

    # Polynomial ile
    (
        X_with_poly,
        _,
        _,
        _,
    ) = SklearnPipeline(
        use_scaling=True,
        use_polynomial=True,
    ).handle_data(
        X_train,
        X_val,
        X_test,
    )

    # Polynomial features daha fazla feature üretmeli
    assert X_no_poly.shape[1] < X_with_poly.shape[1]

    # clean_df de polynomial=False kullandığı için
    # kendi çıktısının feature sayısı X_no_poly ile aynı olmalı
    X_train_step, *_ = clean_df.entrypoint(sample_df)

    assert X_train_step.shape[1] == X_no_poly.shape[1]