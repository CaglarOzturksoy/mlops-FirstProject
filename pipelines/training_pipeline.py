from zenml import pipeline

from steps import ingest_df,clean_df,train_model,evaluate_model

@pipeline(enable_cache=False) #if nothing changed enable_cache = True
def train_pipeline(data_path:str):
    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)