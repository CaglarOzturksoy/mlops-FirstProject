import logging
import pandas as pd
import mlflow
import numpy as np
from typing import Annotated,Tuple,Any
from zenml import step
from zenml.client import Client
from src.evaluation import Accuracy,Precision,Recall,F1Score

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def evaluate_model(
    model: Any, x_test: np.ndarray, y_test: pd.Series
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
    Annotated[float, "f1_score"],
]:
    try:
        prediction = model.predict(x_test)

        y_true_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
        pred_np = np.asarray(prediction)

        accuracy = Accuracy().calculate_score(y_true_np, pred_np)
        mlflow.log_metric("accuracy", accuracy)

        precision = Precision().calculate_score(y_true_np, pred_np)
        mlflow.log_metric("precision", precision)

        recall = Recall().calculate_score(y_true_np, pred_np)
        mlflow.log_metric("recall", recall)

        f1 = F1Score().calculate_score(y_true_np, pred_np)
        mlflow.log_metric("f1_score", f1)

        return accuracy, precision, recall, f1

    except Exception as e:
        logging.error(f"Model değerlendirmesi sırasında hata oluştu: {e}")
        raise 