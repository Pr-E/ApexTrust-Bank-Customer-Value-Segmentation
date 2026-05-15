# NO 4
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.data_ingestion import data_ingestion
from src.data.data_validation import data_validation
from src.features.feature_engineering import FeatureEngineering

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def data_preprocessor(data: pd.DataFrame):
    try:
        logging.info("Data preprocessing started...")

        # =========================
        # SELECT RFM FEATURES
        # =========================
        rfm_processed_data = data[["Recency", "Frequency", "Monetary"]].copy()

        # =========================
        # LOG TRANSFORMATION (reduce skew)
        # =========================
        rfm_processed_data = np.log1p(rfm_processed_data)

        # =========================
        # SCALING
        # =========================
        scaler = StandardScaler()
        rfm_scaled_data = scaler.fit_transform(rfm_processed_data)

        rfm_scaled_df = pd.DataFrame(
            rfm_scaled_data,
            columns=["Recency", "Frequency", "Monetary"],
            index=data.index
        )

        logging.info("Data successfully transformed and scaled")

        return rfm_scaled_df, scaler

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def processing_engine():
    try:
        logging.info("Processing pipeline started...")

        # =========================
        # LOAD DATA
        # =========================
        customer_data = data_ingestion()

        # =========================
        # VALIDATE DATA
        # =========================
        customer_data = data_validation(customer_data)

        # =========================
        # FEATURE ENGINEERING
        # =========================
        fe = FeatureEngineering

        customer_data = fe.calculate_rfm_metrics(customer_data)
        customer_data = fe.calculate_rfm_scores(customer_data)

        # =========================
        # PREPROCESSING
        # =========================
        processed_customer_data, scaler = data_preprocessor(customer_data)

        logging.info("Processing pipeline completed successfully")

        return processed_customer_data, customer_data, scaler

    except Exception as e:
        logging.error(f"Error in processing pipeline: {e}")
        raise


# =========================
# EXECUTION
# =========================
if __name__ == "__main__":
    processed_data, raw_data, scaler = processing_engine()

    print("Processed Data Sample:")
    print(processed_data.head())

    print("\nRaw Data Sample:")
    print(raw_data.head())
