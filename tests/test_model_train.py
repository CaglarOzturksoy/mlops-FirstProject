"""
train_model adımı (steps/model_train.py) için unit testler.

ÖNEMLİ: train_model bir ZenML @step'i olduğu için doğrudan çağrılması
("train_model(...)") ZenML'in gerçek bir "single step pipeline" çalıştırmasına
ve aktif ZenML sunucusuna bağlanmaya çalışmasına yol açar (bkz. base_step.py
__call__ -> run_as_single_step_pipeline). Bu, testlerde ağ hatalarına/timeout'a
sebep olur. Bunun yerine dekoratörün sardığı orijinal fonksiyona
"train_model.entrypoint(...)" ile erişiyoruz; bu hiçbir ZenML altyapısını
tetiklemeden saf Python fonksiyonu gibi çalışır.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# --- Test edilen modül ---
from steps.model_train import train_model, get_experiment_tracker_name
from steps.config import ModelNameConfig

MODULE_PATH = "steps.model_train"


# ---------------------------------------------------------------------------
# Ortak fixture'lar
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    X_train = np.random.rand(50, 4)
    X_val = np.random.rand(10, 4)
    y_train = pd.Series(np.random.randint(0, 2, 50))
    y_val = pd.Series(np.random.randint(0, 2, 10))
    return X_train, X_val, y_train, y_val


@pytest.fixture
def mock_tuner():
    """HyperParameterTuner sınıfını mock'lar."""
    with patch(f"{MODULE_PATH}.HyperParameterTuner") as mock_cls:
        instance = MagicMock()
        instance.optimize.return_value = {"n_estimators": 100, "max_depth": 5}
        mock_cls.return_value = instance
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# get_experiment_tracker_name testleri
# ---------------------------------------------------------------------------

class TestGetExperimentTrackerName:

    @patch(f"{MODULE_PATH}.Client")
    def test_tracker_varsa_ismini_dondurur(self, mock_client_cls):
        mock_tracker = MagicMock()
        mock_tracker.name = "mlflow_tracker"
        mock_client_cls.return_value.active_stack.experiment_tracker = mock_tracker

        result = get_experiment_tracker_name()

        assert result == "mlflow_tracker"

    @patch(f"{MODULE_PATH}.Client")
    def test_tracker_yoksa_none_dondurur(self, mock_client_cls):
        mock_client_cls.return_value.active_stack.experiment_tracker = None

        result = get_experiment_tracker_name()

        assert result is None

    @patch(f"{MODULE_PATH}.Client")
    def test_client_hata_verirse_none_dondurur(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("aktif stack bulunamadı")

        result = get_experiment_tracker_name()

        assert result is None


# ---------------------------------------------------------------------------
# train_model - model seçimi testleri
# ---------------------------------------------------------------------------

class TestTrainModelModelSelection:

    @patch(f"{MODULE_PATH}.RandomForestModel")
    def test_random_forest_secilir(self, mock_model_cls, sample_data, mock_tuner):
        X_train, X_val, y_train, y_val = sample_data
        mock_instance = MagicMock()
        mock_instance.train.return_value = "trained_rf_model"
        mock_model_cls.return_value = mock_instance

        config = ModelNameConfig(model_name="random_forest", fine_tuning=False)
        result = train_model.entrypoint(X_train, X_val, y_train, y_val, config)

        mock_model_cls.assert_called_once()
        mock_instance.train.assert_called_once_with(X_train, y_train)
        assert result == "trained_rf_model"

    @patch(f"{MODULE_PATH}.LightGBMModel")
    def test_lightgbm_secilir(self, mock_model_cls, sample_data, mock_tuner):
        X_train, X_val, y_train, y_val = sample_data
        mock_instance = MagicMock()
        mock_instance.train.return_value = "trained_lgbm_model"
        mock_model_cls.return_value = mock_instance

        config = ModelNameConfig(model_name="lightgbm", fine_tuning=False)
        result = train_model.entrypoint(X_train, X_val, y_train, y_val, config)

        mock_model_cls.assert_called_once()
        assert result == "trained_lgbm_model"

    @patch(f"{MODULE_PATH}.XGBModel")
    def test_xgboost_secilir(self, mock_model_cls, sample_data, mock_tuner):
        X_train, X_val, y_train, y_val = sample_data
        mock_instance = MagicMock()
        mock_instance.train.return_value = "trained_xgb_model"
        mock_model_cls.return_value = mock_instance

        config = ModelNameConfig(model_name="xgboost", fine_tuning=False)
        result = train_model.entrypoint(X_train, X_val, y_train, y_val, config)

        mock_model_cls.assert_called_once()
        assert result == "trained_xgb_model"

    @patch(f"{MODULE_PATH}.LogisticRegressionModel")
    def test_logistic_regression_secilir(self, mock_model_cls, sample_data, mock_tuner):
        X_train, X_val, y_train, y_val = sample_data
        mock_instance = MagicMock()
        mock_instance.train.return_value = "trained_logreg_model"
        mock_model_cls.return_value = mock_instance

        config = ModelNameConfig(model_name="logistic_regression", fine_tuning=False)
        result = train_model.entrypoint(X_train, X_val, y_train, y_val, config)

        mock_model_cls.assert_called_once()
        assert result == "trained_logreg_model"

    def test_desteklenmeyen_model_hata_firlatir(self, sample_data):
        X_train, X_val, y_train, y_val = sample_data
        config = ModelNameConfig(model_name="bilinmeyen_model", fine_tuning=False)

        with pytest.raises(ValueError, match="Unsupported model"):
            train_model.entrypoint(X_train, X_val, y_train, y_val, config)


# ---------------------------------------------------------------------------
# train_model - fine tuning davranışı
# ---------------------------------------------------------------------------

class TestTrainModelFineTuning:

    @patch(f"{MODULE_PATH}.RandomForestModel")
    def test_fine_tuning_kapaliyken_optimize_cagrilmaz(
        self, mock_model_cls, sample_data, mock_tuner
    ):
        X_train, X_val, y_train, y_val = sample_data
        mock_instance = MagicMock()
        mock_instance.train.return_value = "model"
        mock_model_cls.return_value = mock_instance
        tuner_cls, tuner_instance = mock_tuner

        config = ModelNameConfig(model_name="random_forest", fine_tuning=False)
        train_model.entrypoint(X_train, X_val, y_train, y_val, config)

        tuner_instance.optimize.assert_not_called()
        mock_instance.train.assert_called_once_with(X_train, y_train)

    @patch(f"{MODULE_PATH}.RandomForestModel")
    def test_fine_tuning_aciksa_optimize_cagrilir_ve_best_params_kullanilir(
        self, mock_model_cls, sample_data, mock_tuner
    ):
        X_train, X_val, y_train, y_val = sample_data
        mock_instance = MagicMock()
        mock_instance.train.return_value = "model"
        mock_model_cls.return_value = mock_instance
        tuner_cls, tuner_instance = mock_tuner
        best_params = {"n_estimators": 100, "max_depth": 5}
        tuner_instance.optimize.return_value = best_params

        config = ModelNameConfig(model_name="random_forest", fine_tuning=True)
        train_model.entrypoint(X_train, X_val, y_train, y_val, config)

        tuner_instance.optimize.assert_called_once_with(n_trials=200)
        mock_instance.train.assert_called_once_with(X_train, y_train, **best_params)

    @patch(f"{MODULE_PATH}.RandomForestModel")
    def test_tuner_dogru_parametrelerle_olusturulur(
        self, mock_model_cls, sample_data, mock_tuner
    ):
        X_train, X_val, y_train, y_val = sample_data
        mock_model_cls.return_value = MagicMock()
        tuner_cls, _ = mock_tuner

        config = ModelNameConfig(model_name="random_forest", fine_tuning=False)
        train_model.entrypoint(X_train, X_val, y_train, y_val, config)

        _, kwargs = tuner_cls.call_args
        assert kwargs["X_train"] is X_train
        assert kwargs["X_val"] is X_val
        assert kwargs["y_train"] is y_train
        assert kwargs["y_val"] is y_val


# ---------------------------------------------------------------------------
# train_model - hata durumları
# ---------------------------------------------------------------------------

class TestTrainModelErrorHandling:

    @patch(f"{MODULE_PATH}.RandomForestModel")
    def test_model_train_hata_verirse_yeniden_firlatilir(
        self, mock_model_cls, sample_data, mock_tuner
    ):
        X_train, X_val, y_train, y_val = sample_data
        mock_instance = MagicMock()
        mock_instance.train.side_effect = RuntimeError("egitim basarisiz")
        mock_model_cls.return_value = mock_instance

        config = ModelNameConfig(model_name="random_forest", fine_tuning=False)

        with pytest.raises(RuntimeError, match="egitim basarisiz"):
            train_model.entrypoint(X_train, X_val, y_train, y_val, config)

    @patch(f"{MODULE_PATH}.RandomForestModel")
    def test_optimize_hata_verirse_yeniden_firlatilir(
        self, mock_model_cls, sample_data, mock_tuner
    ):
        X_train, X_val, y_train, y_val = sample_data
        mock_model_cls.return_value = MagicMock()
        tuner_cls, tuner_instance = mock_tuner
        tuner_instance.optimize.side_effect = RuntimeError("optimizasyon basarisiz")

        config = ModelNameConfig(model_name="random_forest", fine_tuning=True)

        with pytest.raises(RuntimeError, match="optimizasyon basarisiz"):
            train_model.entrypoint(X_train, X_val, y_train, y_val, config)
