"""
Unit tests for evaluation.py

Run with:
    pytest tests/test_evaluation.py -v
"""

import warnings
import pytest
import numpy as np
from src.evaluation import (Evaluation,Accuracy,Precision,Recall,F1Score)


def test_evaluation_is_abstract():
    """Evaluation abstract class'ından doğrudan nesne oluşturulamamalı."""

    try:
        Evaluation() #type: ignore[abstract ]
        assert False, "Evaluation abstract class olmamalı"
    except TypeError:
        pass


def test_subclasses_are_evaluation_instances():
    """Tüm metric sınıfları Evaluation'dan türemeli."""

    for cls in (Accuracy, Precision, Recall, F1Score):
        metric = cls()

        assert isinstance(metric, Evaluation)


# ------------------------------------------------------------------
# Accuracy Tests
# ------------------------------------------------------------------

def test_accuracy_perfect_predictions():
    """Tamamen doğru tahminlerde accuracy 1.0 olmalı."""

    metric = Accuracy()

    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0])

    result = metric.calculate_score(y_true, y_pred)

    assert result == 1.0


def test_accuracy_all_wrong_predictions():
    """Bütün tahminler yanlış olduğunda accuracy 0.0 olmalı."""

    metric = Accuracy()

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1])

    result = metric.calculate_score(y_true, y_pred)

    assert result == 0.0


def test_accuracy_partial():
    """3 doğru / 4 tahmin = 0.75."""

    metric = Accuracy()

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1])

    result = metric.calculate_score(y_true, y_pred)

    assert result == 0.75


def test_accuracy_return_type():
    """Sonuç Python float olmalı."""

    metric = Accuracy()

    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])

    result = metric.calculate_score(y_true, y_pred)

    assert isinstance(result, float)
    assert not isinstance(result, np.floating)


def test_accuracy_mismatched_lengths():
    """Farklı uzunluktaki diziler hata vermeli."""

    metric = Accuracy()

    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])

    try:
        metric.calculate_score(y_true, y_pred)
        assert False, "Farklı uzunluktaki diziler hata vermeli"
    except Exception:
        pass


# ------------------------------------------------------------------
# Precision Tests
# ------------------------------------------------------------------

def test_precision_perfect_predictions():
    """Tamamen doğru tahminlerde precision 1.0 olmalı."""

    metric = Precision()

    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0])

    result = metric.calculate_score(y_true, y_pred)

    assert result == 1.0


def test_precision_known_value():
    """
    Pozitif tahminler: 3
    Doğru pozitifler: 2

    precision = 2 / 3
    """

    metric = Precision()

    y_true = np.array([1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0])

    result = metric.calculate_score(y_true, y_pred)

    assert result == pytest.approx(2 / 3)


def test_precision_no_positive_predictions():
    """Pozitif tahmin yoksa precision 0.0 dönmeli."""

    metric = Precision()

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = metric.calculate_score(y_true, y_pred)

    assert result == 0.0


def test_precision_return_type():
    """Precision sonucu Python float olmalı."""

    metric = Precision()

    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])

    result = metric.calculate_score(y_true, y_pred)

    assert isinstance(result, float)


# ------------------------------------------------------------------
# Recall Tests
# ------------------------------------------------------------------

def test_recall_perfect_predictions():
    """Tamamen doğru tahminlerde recall 1.0 olmalı."""

    metric = Recall()

    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0])

    result = metric.calculate_score(y_true, y_pred)

    assert result == 1.0


def test_recall_known_value():
    """
    Gerçek pozitifler: 3
    Doğru yakalanan pozitifler: 2

    recall = 2 / 3
    """

    metric = Recall()

    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    result = metric.calculate_score(y_true, y_pred)

    assert result == pytest.approx(2 / 3)


def test_recall_no_actual_positives():
    """Gerçek pozitif yoksa recall 0.0 dönmeli."""

    metric = Recall()

    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = metric.calculate_score(y_true, y_pred)

    assert result == 0.0


def test_recall_return_type():
    """Recall sonucu Python float olmalı."""

    metric = Recall()

    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])

    result = metric.calculate_score(y_true, y_pred)

    assert isinstance(result, float)


# ------------------------------------------------------------------
# F1 Score Tests
# ------------------------------------------------------------------

def test_f1_perfect_predictions():
    """Tamamen doğru tahminlerde F1 1.0 olmalı."""

    metric = F1Score()

    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0])

    result = metric.calculate_score(y_true, y_pred)

    assert result == 1.0


def test_f1_known_value():
    """
    TP = 2
    FP = 0
    FN = 1

    precision = 1.0
    recall = 2/3
    F1 = 0.8
    """

    metric = F1Score()

    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    result = metric.calculate_score(y_true, y_pred)

    assert result == pytest.approx(0.8)


def test_f1_between_precision_and_recall():
    """F1 score precision ve recall arasında olmalı."""

    y_true = np.array([1, 1, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1, 0, 0])

    precision = Precision().calculate_score(y_true, y_pred)
    recall = Recall().calculate_score(y_true, y_pred)
    f1 = F1Score().calculate_score(y_true, y_pred)

    assert f1 >= min(precision, recall) - 1e-9
    assert f1 <= max(precision, recall) + 1e-9


def test_f1_return_type():
    """F1 sonucu Python float olmalı."""

    metric = F1Score()

    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])

    result = metric.calculate_score(y_true, y_pred)

    assert isinstance(result, float)


# ------------------------------------------------------------------
# All Metrics Tests
# ------------------------------------------------------------------

def test_all_metrics_perfect_predictions():
    """Bütün metricler kusursuz tahminlerde 1.0 vermeli."""

    y_true = np.array([1, 0, 1, 0, 1, 1, 0])
    y_pred = y_true.copy()

    for cls in (Accuracy, Precision, Recall, F1Score):
        score = cls().calculate_score(y_true, y_pred)

        assert score == 1.0


def test_all_metrics_completely_wrong_predictions():
    """Tamamen ters tahminlerde bütün metricler 0.0 vermeli."""

    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 1, 1, 1])

    for cls in (Accuracy, Precision, Recall, F1Score):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            score = cls().calculate_score(y_true, y_pred)

        assert score == 0.0