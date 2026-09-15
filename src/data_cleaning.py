import logging
from abc import ABC, abstractmethod
from typing import Union,Tuple,Any,List,cast

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,StandardScaler


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(
        self,*args: Any, **kwargs: Any
        ) -> Union[pd.DataFrame, 
                   Tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame,
                         pd.DataFrame ,pd.Series, pd.Series],
                   Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]]:
        pass

class DataPreprocessStrategy(DataStrategy):

    def _col_replace(self,df:pd.DataFrame,column_name:str,
                     new_cols_name:list,splitter:str) -> pd.DataFrame:

        if column_name in df.columns:
            col_index = int(df.columns.get_indexer(pd.Index([column_name]))[0])
            new_cols = df[column_name].astype(str).str.split(splitter,expand=True)
            new_cols.columns = list(new_cols_name)
            df = df.drop(column_name,axis=1)

            for i,col in enumerate(new_cols.columns):
                df.insert(col_index + i,col,new_cols[col])

        return df


    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            df = df.drop(['PassengerId','Name'],axis = 1)
            df['Transported'] = df['Transported'].astype(int)

            if 'Cabin' in df.columns:
                df  = self._col_replace(df,'Cabin',['Deck','Num','Side'],'/')

            for _col in ('Deck', 'Num', 'Side'):
                if _col in df.columns:
                    df[_col] = df[_col].replace(['nan', 'None', 'NaN', 'none', '<NA>'], np.nan)

            df['Deck'] = df['Deck'].fillna('U')
            df['Num'] = df['Num'].fillna(-1)
            df['Side'] = df['Side'].fillna('U')

            deck_map = {"F": 0, "G": 1, "E": 2, "B": 3, "C": 4, "D": 5, "A": 6, "U": 7, "T": 8}
            side_map = {"U": -1, "S": 1, "P": 2}

            df['Deck'] = df['Deck'].map(deck_map)
            df['Side'] = df['Side'].map(side_map)
            df["Num"] = pd.to_numeric(df["Num"], errors="coerce")

            return df

        except Exception as e:
            logging.error(f'Data Preprocessing Strategy error {e}')
            raise e


class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,
                                                       pd.DataFrame,
                                                       pd.Series,pd.Series]:

        try:
            X  = data.drop('Transported',axis=1)
            y = data['Transported']

            X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,
                                                             random_state=42,shuffle=True,
                                                             stratify=y)

            X_val,X_test,y_val,y_test = train_test_split(X_val,y_val,test_size=0.5,
                                                                         random_state=42,shuffle=True,
                                                                         stratify=y_val)

            return X_train,X_val,X_test,y_train,y_val,y_test     

        except Exception as e:
            logging.error(f'Data Divide strategy error {e}')
            raise e

class SklearnPipeline(DataStrategy):

    def __init__(self,use_scaling:bool = True, use_polynomial:bool = True) -> None:
        self.use_scaling = use_scaling
        self.use_polynomial = use_polynomial

    def build_pipeline(self,X:pd.DataFrame) -> ColumnTransformer:
        numeric_cols = X.select_dtypes(exclude='object').columns
        object_cols = X.select_dtypes(include='object').columns

        numeric_steps: List[Tuple[str,Any]] = [('imputer',SimpleImputer(strategy='mean'))]

        if self.use_scaling:
            numeric_steps.append(('scaler',StandardScaler()))

        if self.use_polynomial:
            numeric_steps.append(('poly',PolynomialFeatures(degree=2, include_bias=False)))

        numeric_pipe = Pipeline(numeric_steps)

        categoric_pipe = Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
        ])

        return ColumnTransformer([
            ('numeric',numeric_pipe,numeric_cols),
            ('categoric',categoric_pipe,object_cols)
        ])

    def handle_data(
            self,X_train:pd.DataFrame,X_val:pd.DataFrame,X_test:pd.DataFrame
            )-> Tuple[np.ndarray,np.ndarray,np.ndarray,ColumnTransformer]:

        try:
            pipeline = self.build_pipeline(X_train)

            X_train_transformed = pipeline.fit_transform(X_train)
            X_val_transformed = pipeline.transform(X_val)
            X_test_transformed = pipeline.transform(X_test)

            X_train_transformed = np.asarray(X_train_transformed)
            X_val_transformed = np.asanyarray(X_val_transformed)
            X_test_transformed = np.asarray(X_test_transformed)
            

            return X_train_transformed, X_val_transformed, X_test_transformed, pipeline

        except Exception as e:
            logging.error(f'Sklearn Pipeline error {e}')
            raise e

class DataCleaning:

    def __init__(self, strategy: DataStrategy) -> None:
        self.strategy = strategy

    def handle_data(self, *args: Any, **kwargs: Any) -> Any:
        return self.strategy.handle_data(*args, **kwargs)