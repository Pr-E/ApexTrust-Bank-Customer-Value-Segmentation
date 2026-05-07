import pandas as pd
import numpy as np
from datetime import datetime

from src.data.data_ingestion import data_ingestion
from src.data.data_validation import data_validation

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class feature_eng:

    @staticmethod
    def calculate_rfm_metrics(data: pd.DataFrame):
        try:
            # =========================
            # CREATE AGE FROM DOB
            # =========================
            data['CustomerDOB'] = pd.to_datetime(data['CustomerDOB'], errors='coerce')

            today = pd.Timestamp.now()
            data['Age'] = data['CustomerDOB'].apply(
                lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                if pd.notnull(dob) else np.nan
            )

            # Drop rows where Age couldn't be calculated
            data = data.dropna(subset=['Age'])

            logging.info("Age column successfully created from CustomerDOB")

            # =========================
            # Reference Date
            # =========================
            reference_date = data['TransactionDate'].max() + pd.Timedelta(days=1)
            logging.info(f"Reference date: {reference_date}")

            # =========================
            # RFM METRICS
            # =========================
            rfm_data = data.groupby('CustomerID').agg({
                'TransactionDate': lambda x: (reference_date - x.max()).days,
                'TransactionID': 'count',
                'TransactionAmount': 'sum'
            }).reset_index()

            rfm_data.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
            logging.info(f"RFM data created successfully:\n{rfm_data.head()}")

            # =========================
            # 👤 CUSTOMER DEMOGRAPHICS (USE AGE)
            # =========================
            customer_demographics = data.groupby('CustomerID').agg({
                'Age': 'first',
                'CustGender': 'first',
                'CustLocation': 'first',
                'CustAccountBalance': 'last'
            }).reset_index()

            logging.info(f"Customer demographics created:\n{customer_demographics.head()}")

            # =========================
            # 🔗 MERGE DATA
            # =========================
            rfm_data = rfm_data.merge(customer_demographics, on='CustomerID', how='left')

            # =========================
            # CREATE AGE GROUPS
            # =========================
            rfm_data['Age_Group'] = pd.cut(
                rfm_data['Age'],
                bins=[18, 30, 45, 60, 100],
                labels=['Young', 'Mid-age', 'Senior', 'Elder']
            )

            logging.info("Age groups successfully created")

            return rfm_data

        except Exception as e:
            logging.error(f"Error calculating RFM metrics: {e}")
            raise


    @staticmethod
    def calculate_rfm_scores(data: pd.DataFrame):
        try:
            # =========================
            # RFM SCORING
            # =========================
            data["R_Score"] = pd.qcut(data["Recency"], q=5, labels=[5, 4, 3, 2, 1])
            data["F_Score"] = pd.qcut(data["Frequency"], q=5, labels=[1, 2, 3, 4, 5])
            data["M_Score"] = pd.qcut(data["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

            # Convert to integers
            data[['R_Score', 'F_Score', 'M_Score']] = data[
                ['R_Score', 'F_Score', 'M_Score']
            ].astype(int)

            # Combined RFM Score
            data["RFM_Score"] = (
                data['R_Score'] +
                data['F_Score'] +
                data['M_Score']
            )

            logging.info(f"RFM scores calculated successfully:\n{data.head()}")

            return data

        except Exception as e:
            logging.error(f"Error calculating RFM scores: {e}")
            raise


# =========================
# PIPELINE EXECUTION
# =========================
if __name__ == "__main__":

    # Load data
    customer_data = data_ingestion()

    # Validate data
    customer_data = data_validation(customer_data)

    # Feature Engineering
    fe = feature_eng

    customer_data = fe.calculate_rfm_metrics(customer_data)
    customer_data = fe.calculate_rfm_scores(customer_data)

    print(customer_data.head())