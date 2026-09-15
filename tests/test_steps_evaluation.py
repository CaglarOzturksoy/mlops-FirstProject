import sys
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------
# Fake ZenML Client
# ---------------------------------------------------------

class FakeActiveStack:
    experiment_tracker = None


class FakeClient:
    @property
    def active_stack(self):
        return FakeActiveStack()


fake_zenml_client = types.ModuleType("zenml.client")
fake_zenml_client.Client = FakeClient #type: ignore[abstract]

# steps.evaluation import edilmeden ÖNCE fake Client'ı yerleştir
sys.modules["zenml.client"] = fake_zenml_client


from steps.evaluation import evaluate_model


# ---------------------------------------------------------
# Fake Models
# ---------------------------------------------------------

class FakeModel:

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, X):
        return self.predictions


class RaisingModel:

    def predict(self, X):
        raise ValueError("model tahmin sırasında bozuldu")


# ---------------------------------------------------------
# Test Data
# ---------------------------------------------------------

Y_TRUE = [1, 0, 1, 1, 0]
Y_PRED = [1, 0, 0, 1, 0]

EXPECTED_ACCURACY = 0.8
EXPECTED_PRECISION = 1.0
EXPECTED_RECALL = 2 / 3
EXPECTED_F1 = 0.8


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def fake_model():
    return FakeModel(Y_PRED)


@pytest.fixture
def x_test():
    return np.zeros((len(Y_TRUE), 3))


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def test_returns_correct_metric_values(
    monkeypatch,
    fake_model,
    x_test,
):

    monkeypatch.setattr(
        "steps.evaluation.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )

    y_test = pd.Series(Y_TRUE)

    accuracy, precision, recall, f1 = evaluate_model.entrypoint(
        fake_model,
        x_test,
        y_test,
    )

    assert accuracy == pytest.approx(EXPECTED_ACCURACY)
    assert precision == pytest.approx(EXPECTED_PRECISION)
    assert recall == pytest.approx(EXPECTED_RECALL)
    assert f1 == pytest.approx(EXPECTED_F1)


def test_return_values_are_python_floats(
    monkeypatch,
    fake_model,
    x_test,
):

    monkeypatch.setattr(
        "steps.evaluation.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )

    y_test = pd.Series(Y_TRUE)

    result = evaluate_model.entrypoint(
        fake_model,
        x_test,
        y_test,
    )

    for value in result:
        assert isinstance(value, float)


def test_accepts_numpy_array_as_y_test(
    monkeypatch,
    fake_model,
    x_test,
):

    monkeypatch.setattr(
        "steps.evaluation.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )

    y_test = np.array(Y_TRUE)

    accuracy, precision, recall, f1 = evaluate_model.entrypoint(
        fake_model,
        x_test,
        y_test,
    )

    assert accuracy == pytest.approx(EXPECTED_ACCURACY)
    assert f1 == pytest.approx(EXPECTED_F1)


def test_mlflow_log_metric_called_for_each_metric(
    monkeypatch,
    fake_model,
    x_test,
):

    calls = []

    def fake_log_metric(name, value):
        calls.append((name, value))

    monkeypatch.setattr(
        "steps.evaluation.mlflow.log_metric",
        fake_log_metric,
    )

    y_test = pd.Series(Y_TRUE)

    evaluate_model.entrypoint(
        fake_model,
        x_test,
        y_test,
    )

    assert len(calls) == 4

    metric_names = [name for name, value in calls]

    assert metric_names == [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]


def test_mlflow_log_metric_called_with_correct_values(
    monkeypatch,
    fake_model,
    x_test,
):

    calls = []

    def fake_log_metric(name, value):
        calls.append((name, value))

    monkeypatch.setattr(
        "steps.evaluation.mlflow.log_metric",
        fake_log_metric,
    )

    y_test = pd.Series(Y_TRUE)

    evaluate_model.entrypoint(
        fake_model,
        x_test,
        y_test,
    )

    logged = dict(calls)

    assert logged["accuracy"] == pytest.approx(
        EXPECTED_ACCURACY
    )

    assert logged["precision"] == pytest.approx(
        EXPECTED_PRECISION
    )

    assert logged["recall"] == pytest.approx(
        EXPECTED_RECALL
    )

    assert logged["f1_score"] == pytest.approx(
        EXPECTED_F1
    )


def test_perfect_predictions_give_all_ones(
    monkeypatch,
    x_test,
):

    monkeypatch.setattr(
        "steps.evaluation.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )

    model = FakeModel(Y_TRUE)

    y_test = pd.Series(Y_TRUE)

    accuracy, precision, recall, f1 = evaluate_model.entrypoint(
        model,
        x_test,
        y_test,
    )

    assert accuracy == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_model_predict_exception_propagates(
    monkeypatch,
    x_test,
):

    calls = []

    def fake_log_metric(name, value):
        calls.append((name, value))

    monkeypatch.setattr(
        "steps.evaluation.mlflow.log_metric",
        fake_log_metric,
    )

    y_test = pd.Series(Y_TRUE)

    with pytest.raises(
        ValueError,
        match="model tahmin sırasında bozuldu",
    ):
        evaluate_model.entrypoint(
            RaisingModel(),
            x_test,
            y_test,
        )

    assert calls == []