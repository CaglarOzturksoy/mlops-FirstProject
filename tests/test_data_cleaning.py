import pandas as pd
import numpy as np
import pytest

from src.data_cleaning import (DataPreprocessStrategy,DataDivideStrategy,SklearnPipeline,DataCleaning)
@pytest.fixture
def sample_data():

    return pd.DataFrame({
        "PassengerId": [f"{i:04d}_01" for i in range(1, 21)],
        "HomePlanet": [
            "Earth", "Europa", "Mars", "Earth", "Europa",
            "Mars", "Earth", "Europa", "Mars", "Earth",
            "Europa", "Mars", "Earth", "Europa", "Mars",
            "Earth", "Europa", "Mars", "Earth", "Europa",
        ],
        "CryoSleep": [False, True] * 10,
        "Cabin": [
            "B/0/P", "F/1/S", "A/2/P", "G/3/S", "C/4/P",
            np.nan,  "D/5/S", "E/6/P", "F/7/S", "G/8/P",
            "A/9/P", "B/10/S", "C/11/P", "D/12/S", "E/13/P",
            "F/14/S", "G/15/P", "A/16/S", "B/17/P", "C/18/S",
        ],
        "Destination": [
            "TRAPPIST-1e", "55 Cancri e", "TRAPPIST-1e", "PSO J318.5-22", "TRAPPIST-1e",
            "55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e", "55 Cancri e", "TRAPPIST-1e",
            "TRAPPIST-1e", "55 Cancri e", "TRAPPIST-1e", "PSO J318.5-22", "TRAPPIST-1e",
            "55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e", "55 Cancri e", "TRAPPIST-1e",
        ],
        "Age": [39, 24, 58, 33, 21, 45, 30, 27, 19, 40, 35, 28, 50, 22, 33, 41, 19, 60, 25, 38],
        "VIP": [False] * 20,
        "RoomService": [0, 100, 50, 0, 20, 30, 0, 10, 50, 0, 5, 0, 80, 0, 15, 0, 0, 60, 0, 10],
        "FoodCourt": [0, 9, 3576, 0, 15, 0, 0, 20, 5, 0, 0, 0, 50, 0, 0, 0, 0, 100, 0, 0],
        "ShoppingMall": [0, 25, 0, 0, 10, 0, 0, 5, 0, 0, 0, 0, 20, 0, 0, 0, 0, 15, 0, 0],
        "Spa": [0, 549, 6715, 0, 30, 0, 0, 100, 0, 0, 0, 0, 200, 0, 0, 0, 0, 300, 0, 0],
        "VRDeck": [0, 44, 49, 0, 5, 0, 0, 15, 0, 0, 0, 0, 30, 0, 0, 0, 0, 40, 0, 0],
        "Name": [f"Person {i}" for i in range(1, 21)],
        "Transported": [True, False] * 10,
    })



def test_data_preprocess_strategy(sample_data):
    """Verinin preprocessing aşamasını test eder."""

    strategy = DataPreprocessStrategy()
    result = strategy.handle_data(sample_data)

    # PassengerId ve Name kaldırılmış olmalı
    assert "PassengerId" not in result.columns
    assert "Name" not in result.columns

    # Cabin parçalanmış olmalı
    assert "Deck" in result.columns
    assert "Num" in result.columns
    assert "Side" in result.columns
    assert "Cabin" not in result.columns

    # Target integer olmalı
    assert pd.api.types.is_integer_dtype(result["Transported"])

    # Deck ve Side sayısal hale gelmiş olmalı
    assert pd.api.types.is_numeric_dtype(result["Deck"])
    assert pd.api.types.is_numeric_dtype(result["Side"])

    # Num sayısal olmalı
    assert pd.api.types.is_numeric_dtype(result["Num"])


def test_data_preprocess_missing_values(sample_data):
    """Eksik Cabin değerlerinin düzgün işlendiğini test eder."""

    strategy = DataPreprocessStrategy()
    result = strategy.handle_data(sample_data)

    # Cabin eksik olduğunda Deck ve Side için U,
    # Num için -1 atanmalı.
    assert result["Deck"].notna().all()
    assert result["Num"].notna().all()
    assert result["Side"].notna().all()


def test_data_divide_strategy(sample_data):
    """Verinin train/validation/test olarak ayrılmasını test eder."""

    preprocess_strategy = DataPreprocessStrategy()
    processed_data = preprocess_strategy.handle_data(sample_data)

    strategy = DataDivideStrategy()

    X_train,X_val,X_test,y_train,y_val,y_test = strategy.handle_data(processed_data)

    # Toplam örnek sayısı korunmalı
    total_samples = len(X_train) + len(X_val) + len(X_test)

    assert total_samples == len(processed_data)

    # Train yaklaşık %80
    assert len(X_train) == 16

    # Validation %10
    assert len(X_val) == 2

    # Test %10
    assert len(X_test) == 2

    # Target uzunlukları X ile aynı olmalı
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # Target X içerisinde olmamalı
    assert "Transported" not in X_train.columns
    assert "Transported" not in X_val.columns
    assert "Transported" not in X_test.columns


def test_sklearn_pipeline(sample_data):
    """Scikit-learn preprocessing pipeline'ını test eder."""

    preprocess_strategy = DataPreprocessStrategy()
    processed_data = preprocess_strategy.handle_data(sample_data)

    divide_strategy = DataDivideStrategy()

    X_train,X_val,X_test,y_train,y_val,y_test = divide_strategy.handle_data(processed_data)

    pipeline_strategy = SklearnPipeline(use_scaling=True,use_polynomial=True)

    X_train_transformed,X_val_transformed,X_test_transformed,pipeline = pipeline_strategy.handle_data(X_train,X_val,X_test,)

    # Çıktılar numpy array olmalı
    assert isinstance(X_train_transformed, np.ndarray)
    assert isinstance(X_val_transformed, np.ndarray)
    assert isinstance(X_test_transformed, np.ndarray)

    # Pipeline ColumnTransformer olmalı
    from sklearn.compose import ColumnTransformer

    assert isinstance(pipeline, ColumnTransformer)

    # Train/validation/test aynı feature sayısına sahip olmalı
    assert X_train_transformed.shape[1] == X_val_transformed.shape[1]
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]

    # Satır sayıları korunmalı
    assert X_train_transformed.shape[0] == len(X_train)
    assert X_val_transformed.shape[0] == len(X_val)
    assert X_test_transformed.shape[0] == len(X_test)

    # NaN bulunmamalı
    assert not np.isnan(X_train_transformed).any()
    assert not np.isnan(X_val_transformed).any()
    assert not np.isnan(X_test_transformed).any()


def test_data_cleaning_class(sample_data):
    """DataCleaning sınıfının strategy pattern ile çalışmasını test eder."""
    strategy = DataPreprocessStrategy()
    cleaning = DataCleaning(strategy)

    result = cleaning.handle_data(sample_data)

    assert isinstance(result, pd.DataFrame)
    assert "Transported" in result.columns
    assert "Cabin" not in result.columns