from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig


@pipeline
def training_pipeline(data_path: str):
    df = ingest_df(data_path)

    X_train, X_val, X_test, y_train, y_val, y_test = clean_df(df)

    model = train_model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        config = ModelNameConfig(model_name='lightgbm',fine_tuning=True)
    )

    evaluate_model(
        model=model,
        x_test=X_test,
        y_test=y_test
    )

    return model


if __name__ == "__main__":
    training_pipeline(
        data_path="data/Space_train.csv")