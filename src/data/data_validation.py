# NO 2
import pandas as pd
import logging

from src.data.data_ingestion import data_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def data_validation(data: pd.DataFrame):

    try:
        customer_data = data.copy()

        # =========================
        # DUPLICATE CHECKS
        # =========================
        duplicates = customer_data.duplicated().sum()

        duplicate_transaction = customer_data[
            'TransactionID'
        ].duplicated().sum()

        logging.info(f"Duplicate Rows: {duplicates}")
        logging.info(f"Duplicate TransactionIDs: {duplicate_transaction}")

        # Remove duplicate transaction IDs
        if duplicate_transaction > 0:
            customer_data = customer_data.drop_duplicates(
                subset='TransactionID',
                keep='first'
            )

            logging.info("Duplicate TransactionIDs removed")

        # =========================
        # MISSING VALUES
        # =========================
        missing_values = customer_data.isna().sum()

        logging.info(f"Missing Values:\n{missing_values}")

        customer_data = customer_data.dropna()

        logging.info("Missing values removed successfully")

        # =========================
        # REMOVE PROBLEMATIC CUSTOMER
        # =========================
        customer_data = customer_data[
            customer_data['CustomerID'] != 'C2867825'
        ]

        logging.info("Problematic customer removed")

        # =========================
        # DATE CONVERSION
        # =========================
        customer_data['TransactionDate'] = pd.to_datetime(
            customer_data['TransactionDate'],
            errors='coerce'
        )

        logging.info("TransactionDate converted successfully")

        # =========================
        # BASIC CHECKS
        # =========================
        logging.info(
            f"Unique Customers: {customer_data['CustomerID'].nunique()}"
        )

        return customer_data

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        raise


# =========================
# TEST
# =========================
if __name__ == "__main__":

    customer_data = data_ingestion()

    clean_data = data_validation(customer_data)

    print(clean_data.head())