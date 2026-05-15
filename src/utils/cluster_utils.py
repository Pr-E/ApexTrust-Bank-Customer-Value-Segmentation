# =========================================
# CUSTOMER CLUSTER ANALYSIS
# =========================================

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================
# CLUSTER ANALYSIS
# =========================================
def cluster_analyzer(
    rfm_data: pd.DataFrame
):

    try:

        logging.info(
            "Starting cluster analysis..."
        )

        cluster_analysis = rfm_data.groupby(
            "Cluster"
        ).agg({

            "Age":
                "mean",

            "Recency":
                "mean",

            "Frequency":
                "mean",

            "Monetary":
                "mean",

            "R_Score":
                "mean",

            "F_Score":
                "mean",

            "M_Score":
                "mean",

            "RFM_Score":
                "mean",

            "CustomerID":
                "count",

            "CustAccountBalance":
                "mean"

        }).round(2)

        # =====================================
        # RENAME COLUMNS
        # =====================================
        cluster_analysis.columns = [

            "Avg_Age",

            "Avg_Recency",

            "Avg_Frequency",

            "Avg_Monetary",

            "Avg_R_Score",

            "Avg_F_Score",

            "Avg_M_Score",

            "Avg_RFM_Score",

            "Customer_Count",

            "Avg_Account_Balance"
        ]

        logging.info(
            "Cluster analysis completed successfully"
        )

        return cluster_analysis

    except Exception as e:

        logging.error(
            f"Cluster analysis error: {e}"
        )

        raise


# =========================================
# CUSTOMER VALUE LEVEL
# =========================================
def generate_customer_value_level(
    rfm_score
):

    if rfm_score >= 12:

        return "High Value"

    elif rfm_score >= 8:

        return "Moderate Value"

    else:

        return "Low Value"





# =========================================
# BUSINESS SEGMENT ASSIGNMENT
# =========================================
def assign_name_to_cluster(stats):

    recency = stats["Avg_Recency"]

    frequency = stats["Avg_Frequency"]

    monetary = stats["Avg_Monetary"]

    # =====================================
    # LOYAL HIGH VALUE CUSTOMER
    # =====================================
    if (

        monetary >= 50000
        and frequency >= 12
        and recency <= 120

    ):

        return "Loyal High-Value Customer"

    # =====================================
    # AT-RISK PREMIUM CUSTOMER
    # =====================================
    elif (

        monetary >= 15000
        and recency > 120
        and frequency < 10

    ):

        return "At-Risk Premium Customer"

    # =====================================
    # ACTIVE EVERYDAY CUSTOMER
    # =====================================
    elif (

        frequency >= 10
        and recency <= 120
        and monetary < 15000

    ):

        return "Active Everyday Customer"

    # =====================================
    # LOW ENGAGEMENT CUSTOMER
    # =====================================
    else:

        return "Low Engagement Customer"

# =========================================
# APPLY CLUSTER NAMES
# =========================================
def assign_cluster_names(
    rfm_data: pd.DataFrame,
    cluster_analysis: pd.DataFrame
):

    try:

        logging.info(
            "Assigning business segment names..."
        )

        cluster_mapping = {}

        for cluster_id in cluster_analysis.index:

            cluster_mapping[cluster_id] = (

                assign_name_to_cluster(
                    cluster_analysis.loc[cluster_id]
                )
            )

        # =====================================
        # DISPLAY CLUSTER MAPPING
        # =====================================
        print("\nCLUSTER MAPPING\n")

        for cluster_id, cluster_name in (
            cluster_mapping.items()
        ):

            print(
                f"Cluster {cluster_id} "
                f"--> {cluster_name}"
            )

        # =====================================
        # MAP CLUSTER NAMES
        # =====================================
        rfm_data["Cluster_Name"] = (
            rfm_data["Cluster"]
            .map(cluster_mapping)
        )

        logging.info(
            "Cluster names assigned successfully"
        )

        return rfm_data

    except Exception as e:

        logging.error(
            f"Cluster naming error: {e}"
        )

        raise


# =========================================
# FINAL CLUSTER PROFILE
# =========================================
def cluster_grouping(
    rfm_data: pd.DataFrame
):

    try:

        logging.info(
            "Creating final cluster profiles..."
        )

        cluster_profile = rfm_data.groupby(
            "Cluster_Name"
        ).agg({

            "Recency":
                "mean",

            "Frequency":
                "mean",

            "Monetary":
                "mean",

            "R_Score":
                "mean",

            "F_Score":
                "mean",

            "M_Score":
                "mean",

            "RFM_Score":
                "mean",

            "CustomerID":
                "count",

            "CustAccountBalance":
                "mean",

            "Age":
                "mean"

        }).round(2)

        # =====================================
        # RENAME COLUMNS
        # =====================================
        cluster_profile.columns = [

            "Avg_Recency",

            "Avg_Frequency",

            "Avg_Monetary",

            "Avg_R_Score",

            "Avg_F_Score",

            "Avg_M_Score",

            "Avg_RFM_Score",

            "Customer_Count",

            "Avg_Account_Balance",

            "Avg_Age"
        ]

        # =====================================
        # CUSTOMER VALUE LEVEL
        # =====================================
        cluster_profile[
            "Customer_Value_Level"
        ] = cluster_profile[
            "Avg_RFM_Score"
        ].apply(
            generate_customer_value_level
        )

        # =====================================
        # CUSTOMER PERCENTAGE
        # =====================================
        cluster_profile[
            "Customer_Percentage"
        ] = (

            cluster_profile[
                "Customer_Count"
            ]

            /

            cluster_profile[
                "Customer_Count"
            ].sum()

        ) * 100

        cluster_profile[
            "Customer_Percentage"
        ] = cluster_profile[
            "Customer_Percentage"
        ].round(2)

        # =====================================
        # ESTIMATED TOTAL VALUE
        # =====================================
        cluster_profile[
            "Estimated_Total_Value"
        ] = (

            cluster_profile[
                "Avg_Monetary"
            ]

            *

            cluster_profile[
                "Customer_Count"
            ]

        ).round(2)

        # =====================================
        # SORT BY MONETARY VALUE
        # =====================================
        cluster_profile = (
            cluster_profile.sort_values(
                by="Avg_Monetary",
                ascending=False
            )
        )

        logging.info(
            "Cluster profiles created successfully"
        )

        return cluster_profile

    except Exception as e:

        logging.error(
            f"Cluster profile error: {e}"
        )

        raise