import pandas as pd
from src.connections.mongodb_connection import MongoDBConnection
import logging

logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

def data_ingestion():
    try:
        # GET COLLECTION AND LOAD DATA
        data_connector = MongoDBConnection()
        collection = data_connector.get_collection()
        df = pd.DataFrame(list(collection.find({})))

        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        logging.info(f"data has been successfully loaded...")
        print(df.head())
        return df

    except Exception as e:
        logging.error(f"error occured while loading the dataset from the database {e}")
        return None 


data_ingestion()       
