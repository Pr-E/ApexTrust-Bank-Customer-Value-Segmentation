import pandas as pd
import numpy as np
from src.data.data_ingestion import data_ingestion
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def data_validation(data: pd.DataFrame):

    try:
        customer_data = data.copy()

        # =========================
        # DUPLICATES
        # =========================
        duplicates = customer_data.duplicated().sum()
        duplicate_transaction = customer_data['TransactionID'].duplicated().sum()

        logging.info(f"Total duplicate rows: {duplicates}")
        logging.info(f"Duplicate TransactionIDs: {duplicate_transaction}")

        if duplicate_transaction > 0:
            customer_data = customer_data.drop_duplicates(
                subset='TransactionID',
                keep='first'
            )
            logging.info("Duplicate TransactionIDs removed successfully")

        # =========================
        # MISSING VALUES
        # =========================
        missing_values = customer_data.isna().sum()
        logging.info(f"Missing values by column:\n{missing_values}")

        customer_data = customer_data.dropna()
        logging.info("Missing values successfully dropped")

        # =========================
        # REMOVE SPECIFIC CUSTOMER
        # =========================
        before = len(customer_data)

        customer_data = customer_data[
            customer_data['CustomerID'] != 'C2867825'
        ]

        after = len(customer_data)

        logging.info(f"Removed {before - after} records for CustomerID C2867825")

        # =========================
        # OTHER CHECKS
        # =========================
        unique_customers = customer_data['CustomerID'].nunique()
        gender_distribution = customer_data['CustGender'].value_counts()

        logging.info(f"Unique Customers: {unique_customers}")
        logging.info(f"Gender Distribution:\n{gender_distribution}")

        # =========================
        # DATE HANDLING
        # =========================
        customer_data["TransactionDate"] = pd.to_datetime(
            customer_data['TransactionDate'],
            errors="coerce"
        )

        logging.info("TransactionDate successfully converted")

        return customer_data

    except Exception as e:
        logging.error(f"Error during data validation: {e}")
        raise


# Run pipeline
customer_data = data_ingestion()
clean_data = data_validation(customer_data)