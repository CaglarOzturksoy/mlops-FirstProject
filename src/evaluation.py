import logging
from abc import ABC,abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class Evaluation(ABC):

    @abstractmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray)->float:
        pass

class Accuracy(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        try:
            accuracy = accuracy_score(y_true,y_pred)
            logging.info(f'Accuracy score value is: {accuracy}')
            return float(accuracy)

        except Exception as e:
            logging.error('Exception occurred in calculate_score method of the Accuracy class')
            raise e

class Precision(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            precision = precision_score(y_true,y_pred)
            logging.info(f'Precision score value is: {precision}')
            return float(precision)

        except Exception as e:
            logging.error('Exception occurred in calculate_score method of the Precision class')
            raise e

class Recall(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            recall = recall_score(y_true,y_pred)
            logging.info(f'Recall score value is: {recall}')
            return float(recall)

        except Exception as e:
            logging.error('Exception occurred in calculate_score method of the Recall class')
            raise e

class F1Score(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            f1 = f1_score(y_true,y_pred)
            logging.info(f'f1 score value is: {f1}')
            return float(f1)

        except Exception as e:
            logging.error('Exception occurred in calculate_score method of the F1Score class')
            raise e
