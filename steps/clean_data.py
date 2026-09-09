import logging
import pandas as pd
from zenml import step
import numpy as np
from typing import Annotated,Tuple
from src.data_cleaning import DataCleaning,DataPreprocessStrategy,DataDivideStrategy,SklearnPipeline

@step
def clean_df(
    df: pd.DataFrame
    ) -> Tuple[
        Annotated[np.ndarray,'X_train'],
        Annotated[np.ndarray,'X_val'],
        Annotated[np.ndarray,'X_test'],
        Annotated[pd.Series,'y_train'],
        Annotated[pd.Series,'y_val'],
        Annotated[pd.Series,'y_test']
    ]:

    try:
        preprocess_df = DataCleaning(DataPreprocessStrategy()).handle_data(df)
        X_train,X_val,X_test,y_train,y_val,y_test = DataCleaning(DataDivideStrategy()).handle_data(preprocess_df)
        X_train_trans,X_val_trans,X_test_trans,pipeline = DataCleaning(
            SklearnPipeline(use_scaling=True,use_polynomial=False)
            ).handle_data(X_train,X_val,X_test)
        
        return X_train_trans,X_val_trans,X_test_trans,y_train,y_val,y_test

    except Exception as e:
        logging.error(f'Error in Cleaning Data {e}')
        raise e
    
    