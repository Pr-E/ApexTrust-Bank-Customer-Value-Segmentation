# NO 1     
import pandas as pd
import logging

from config.constant import INPUT_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def data_ingestion():
    try:
        logging.info("Loading dataset from local storage...")

        # =========================
        # LOAD CSV
        # =========================
        df = pd.read_csv(INPUT_DATA_PATH)

        logging.info(f"Dataset loaded successfully")
        logging.info(f"Dataset Shape: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


# =========================
# TEST
# =========================
if __name__ == "__main__":
    df = data_ingestion()
    print(df.head())