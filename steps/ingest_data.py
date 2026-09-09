import logging
import pandas as pd
from zenml import step

class IngestData:

    def __init__(self,datapath: str):
        self.datapath = datapath

    def get_data(self):
        logging.info(f'Ingesting data from {self.datapath}')
        return pd.read_csv(self.datapath)

@step
def ingest_df(data_path:str) -> pd.DataFrame:

    try:
        ingestor = IngestData(data_path)
        df = ingestor.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e