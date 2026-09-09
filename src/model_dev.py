import logging
from abc import ABC,abstractmethod
from typing import Union,cast,Any
import numpy as np
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score

class Model(ABC):

    @abstractmethod
    def train(self,X_train,y_train)->Union[ClassifierMixin, LGBMClassifier]:
        pass

    @abstractmethod
    def optimize(self,trial,X_train,y_train,X_val,y_val)->Union[float,np.floating]:
        pass

class RandomForestModel(Model):

    def train(self, X_train, y_train,**kwargs):
        model = RandomForestClassifier(**kwargs)
        model.fit(X_train,y_train)

        return model

    def optimize(self, trial, X_train, y_train, X_val, y_val):
        n_estimators = trial.suggest_int('n_estimators',1,500)
        max_depth = trial.suggest_int('max_depth',1,20)
        min_samples_split = trial.suggest_int('min_samples_split',2,32)
        model = self.train(X_train,y_train,n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
        predictions = model.predict(X_val)
        return float(f1_score(y_val,predictions,average='macro'))

class XGBModel(Model):

    def train(self, X_train, y_train,**kwargs):
        model = xgb.XGBClassifier(**kwargs)
        model.fit(X_train,y_train)

        return model

    def optimize(self, trial, X_train, y_train, X_val, y_val):
        n_estimators = trial.suggest_int('n_estimators',1,500)
        max_depth = trial.suggest_int('max_depth',1,20)
        learning_rate = trial.suggest_float('learning_rate',1e-5,10,log=True)
        model = self.train(X_train,y_train,n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        predictions = model.predict(X_val)
        return float(f1_score(y_val,predictions,average='macro'))

class LightGBMModel(Model):

    def train(self, X_train, y_train,**kwargs):
        model = LGBMClassifier(**kwargs)
        model.fit(X_train,y_train)

        return model

    def optimize(self, trial, X_train, y_train, X_val, y_val):
        n_estimators = trial.suggest_int('n_estimators',1,500)
        max_depth = trial.suggest_int('max_depth',1,20)
        learning_rate = trial.suggest_float('learning_rate',1e-5,10,log=True)
        model = self.train(X_train,y_train,n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        predictions = np.asarray(model.predict(X_val))
        return float(f1_score(y_val,predictions,average='macro'))

class LogisticRegressionModel(Model):

    def train(self, X_train, y_train,**kwargs):
        model = LogisticRegression(**kwargs)
        model.fit(X_train,y_train)
        return model

    def optimize(self, trial, X_train, y_train, X_val, y_val):
        model = self.train(X_train, y_train)
        predictions = model.predict(X_val)
        return float(f1_score(y_val,predictions,average='macro'))

class HyperParameterTuner:

    def __init__(self,model,X_train,y_train,X_val,y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def optimize(self,n_trials=20):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train, self.X_val, self.y_val), n_trials=n_trials)
        return study.best_trial.params