# =========================================
# NO 3: FEATURE ENGINEERING
# =========================================

import pandas as pd
import numpy as np
import logging

from src.data.data_ingestion import data_ingestion
from src.data.data_validation import data_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================
# FEATURE ENGINEERING CLASS
# =========================================
class FeatureEngineering:

    # =====================================
    # CALCULATE RFM METRICS
    # =====================================
    @staticmethod
    def calculate_rfm_metrics(
        data: pd.DataFrame
    ):

        try:

            logging.info(
                "Starting feature engineering..."
            )

            # =================================
            # DATE CONVERSION
            # =================================
            data["CustomerDOB"] = pd.to_datetime(
                data["CustomerDOB"],
                errors="coerce"
            )

            data["TransactionDate"] = pd.to_datetime(
                data["TransactionDate"],
                errors="coerce"
            )

            # =================================
            # REMOVE INVALID DATES
            # =================================
            data = data.dropna(
                subset=[
                    "CustomerDOB",
                    "TransactionDate"
                ]
            )

            # =================================
            # AGE FEATURE
            # =================================
            today = pd.Timestamp.now()

            data["Age"] = data[
                "CustomerDOB"
            ].apply(

                lambda dob:
                today.year
                - dob.year
                - (
                    (
                        today.month,
                        today.day
                    ) < (
                        dob.month,
                        dob.day
                    )
                )

                if pd.notnull(dob)
                else np.nan
            )

            # =================================
            # REMOVE INVALID AGES
            # =================================
            data = data[

                (data["Age"] >= 18)

                &

                (data["Age"] <= 100)

            ]

            logging.info(
                "Age feature created successfully"
            )

            # =================================
            # REFERENCE DATE
            # =================================
            reference_date = (

                data["TransactionDate"].max()

                + pd.Timedelta(days=1)
            )

            logging.info(
                f"Reference Date: {reference_date}"
            )

            # =================================
            # RFM METRICS
            # =================================
            rfm_data = data.groupby(
                "CustomerID"
            ).agg({

                "TransactionDate":
                    lambda x:
                    (
                        reference_date
                        - x.max()
                    ).days,

                "TransactionID":
                    "count",

                "TransactionAmount":
                    "sum"
            }).reset_index()

            rfm_data.columns = [

                "CustomerID",

                "Recency",

                "Frequency",

                "Monetary"
            ]

            logging.info(
                "RFM metrics generated successfully"
            )

            # =================================
            # CUSTOMER DEMOGRAPHICS
            # =================================
            customer_demographics = (
                data.groupby("CustomerID")
                .agg({

                    "Age":
                        "first",

                    "CustGender":
                        "first",

                    "CustLocation":
                        "first",

                    "CustAccountBalance":
                        "last"
                })
                .reset_index()
            )

            # =================================
            # MERGE DATA
            # =================================
            rfm_data = rfm_data.merge(

                customer_demographics,

                on="CustomerID",

                how="left"
            )

            # =================================
            # AGE GROUPS
            # =================================
            rfm_data["Age_Group"] = pd.cut(

                rfm_data["Age"],

                bins=[18, 30, 45, 60, 100],

                labels=[
                    "Young",
                    "Mid-age",
                    "Senior",
                    "Elder"
                ]
            )

            logging.info(
                "Feature engineering completed successfully"
            )

            return rfm_data

        except Exception as e:

            logging.error(
                f"Feature engineering error: {e}"
            )

            raise

    # =====================================
    # CALCULATE RFM SCORES
    # =====================================
    @staticmethod
    def calculate_rfm_scores(
        data: pd.DataFrame
    ):

        try:

            logging.info(
                "Calculating RFM scores..."
            )

            # =================================
            # RECENCY SCORE
            # LOWER RECENCY = BETTER
            # =================================
            data["R_Score"] = pd.qcut(

                data["Recency"],

                q=5,

                labels=[5, 4, 3, 2, 1],

                duplicates="drop"
            )

            # =================================
            # FREQUENCY SCORE
            # HIGHER FREQUENCY = BETTER
            # =================================
            data["F_Score"] = pd.qcut(

                data["Frequency"],

                q=5,

                labels=[1, 2, 3, 4, 5],

                duplicates="drop"
            )

            # =================================
            # MONETARY SCORE
            # HIGHER MONETARY = BETTER
            # =================================
            data["M_Score"] = pd.qcut(

                data["Monetary"],

                q=5,

                labels=[1, 2, 3, 4, 5],

                duplicates="drop"
            )

            # =================================
            # CONVERT TO INTEGER
            # =================================
            score_columns = [

                "R_Score",

                "F_Score",

                "M_Score"
            ]

            data[score_columns] = (
                data[score_columns]
                .astype(int)
            )

            # =================================
            # TOTAL RFM SCORE
            # =================================
            data["RFM_Score"] = (

                data["R_Score"]

                +

                data["F_Score"]

                +

                data["M_Score"]
            )

            logging.info(
                "RFM scoring completed successfully"
            )

            return data

        except Exception as e:

            logging.error(
                f"RFM scoring error: {e}"
            )

            raise


# =========================================
# PIPELINE EXECUTION
# =========================================
if __name__ == "__main__":

    # =====================================
    # LOAD DATA
    # =====================================
    customer_data = data_ingestion()

    # =====================================
    # VALIDATE DATA
    # =====================================
    customer_data = data_validation(
        customer_data
    )

    # =====================================
    # FEATURE ENGINEERING
    # =====================================
    fe = FeatureEngineering

    customer_data = (
        fe.calculate_rfm_metrics(
            customer_data
        )
    )

    customer_data = (
        fe.calculate_rfm_scores(
            customer_data
        )
    )

    print(customer_data.head())
