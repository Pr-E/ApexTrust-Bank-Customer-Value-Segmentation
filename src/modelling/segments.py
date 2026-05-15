# =========================
# NO 9
# CUSTOMER SEGMENTATION ENGINE
# =========================

import logging

from src.modelling.clusters import (
    ClusteringEngine
)

from src.utils.model_loader import (
    load_model_and_params
)

from src.utils.cluster_utils import (

    cluster_analyzer,

    assign_cluster_names,

    cluster_grouping
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SegmentEngine:

    @staticmethod
    def cluster_grouper():

        try:

            logging.info(
                "Customer segmentation pipeline started..."
            )

            # =====================================
            # INITIALIZE CLUSTER ENGINE
            # =====================================
            clustering_engine = (
                ClusteringEngine()
            )

            # =====================================
            # LOAD BEST KMEANS MODEL
            # =====================================
            model, optimal_k = (
                load_model_and_params()
            )

            logging.info(
                f"Loaded clustering model "
                f"with {optimal_k} clusters"
            )

            # =====================================
            # COPY CUSTOMER DATA
            # =====================================
            clustered_customer_data = (
                clustering_engine
                .customer_data
                .copy()
            )

            # =====================================
            # ONLY USE RFM FEATURES
            # =====================================
            clustering_features = (

                clustering_engine
                .processed_customer_data[
                    [
                        "Recency",
                        "Frequency",
                        "Monetary"
                    ]
                ]
            )

            # =====================================
            # APPLY CLUSTER PREDICTIONS
            # =====================================
            clustered_customer_data[
                "Cluster"
            ] = model.predict(
                clustering_features
            )

            logging.info(
                "Customer clustering completed"
            )

            # =====================================
            # CLUSTER ANALYSIS
            # =====================================
            cluster_analysis = (
                cluster_analyzer(
                    clustered_customer_data
                )
            )

            logging.info(
                "Cluster analysis completed"
            )

            # =====================================
            # ASSIGN BUSINESS SEGMENTS
            # =====================================
            segmented_customer_data = (

                assign_cluster_names(

                    clustered_customer_data,

                    cluster_analysis
                )
            )

            logging.info(
                "Business segment names assigned"
            )

            # =====================================
            # FINAL SEGMENT PROFILE
            # =====================================
            customer_segment_profile = (

                cluster_grouping(
                    segmented_customer_data
                )
            )

            logging.info(
                "Customer segment profile created"
            )

            # =====================================
            # OUTPUT PREVIEW
            # =====================================
            print("\nSEGMENTED CUSTOMER DATA\n")

            print(
                segmented_customer_data.head().T
            )

            print("\nCUSTOMER SEGMENT PROFILE\n")

            print(
                customer_segment_profile.head().T
            )

            logging.info(
                "Customer segmentation pipeline "
                "completed successfully"
            )

            return (

                segmented_customer_data,

                customer_segment_profile
            )

        except Exception as e:

            logging.error(
                f"Segmentation pipeline error: {e}"
            )

            raise


# =========================================
# MAIN EXECUTION
# =========================================
def main():

    engine = SegmentEngine()

    segmented_data, segment_profile = (
        engine.cluster_grouper()
    )

    return (
        segmented_data,
        segment_profile
    )


# =========================================
# EXECUTION
# =========================================
if __name__ == "__main__":

    segmented_data, segment_profile = (
        main()
    )





