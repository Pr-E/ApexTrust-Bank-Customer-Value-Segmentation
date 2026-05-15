# NO 5: CUSTOMER CLUSTERING ENGINE
# DETERMINE OPTIMAL CLUSTERS

import logging
import numpy as np
import pandas as pd
import mlflow

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.mlflow_config import setup_mlflow

from src.utils.model_loader import (
    is_new_model_better,
    load_model_and_params
)

from src.data.data_preprocessing import (
    processing_engine
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClusteringEngine:

    def __init__(self):

        (
            processed_customer_data,
            customer_data,
            scaler
        ) = processing_engine()

        # =====================================
        # ONLY USE CORE RFM FEATURES
        # =====================================
        self.processed_customer_data = (
            processed_customer_data[
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
        )

        self.customer_data = customer_data

        self.scaler = scaler

        self.model = None

        self.optimal_k = None

    # =========================================
    # FIND OPTIMAL CLUSTERS
    # =========================================
    def find_optimal_cluster(self):

        try:

            logging.info(
                "Searching for optimal clusters..."
            )

            silhouette_scores = []

            cluster_range = range(2, 11)

            best_score = -1

            best_model = None

            for k in cluster_range:

                kmeans = KMeans(

                    n_clusters=k,

                    random_state=42,

                    n_init=10
                )

                labels = kmeans.fit_predict(
                    self.processed_customer_data
                )

                score = silhouette_score(

                    self.processed_customer_data,

                    labels
                )

                silhouette_scores.append(
                    score
                )

                if score > best_score:

                    best_score = score

                    best_model = kmeans

            optimal_k = cluster_range[
                np.argmax(silhouette_scores)
            ]

            self.model = best_model

            self.optimal_k = optimal_k

            logging.info(
                f"Optimal K: {optimal_k}"
            )

            logging.info(
                f"Best Silhouette Score: "
                f"{best_score:.4f}"
            )

            return optimal_k, best_score

        except Exception as e:

            logging.error(
                f"Error finding optimal clusters: {e}"
            )

            raise

    # =========================================
    # TRAIN AND LOG MODEL
    # =========================================
    def train_and_log_model(self):

        try:

            setup_mlflow()

            optimal_k, best_silhouette = (
                self.find_optimal_cluster()
            )

            is_better, previous_best = (
                is_new_model_better(
                    best_silhouette,
                    experiment_name="customer_value_segmentation"
                )
            )

            if is_better:

                logging.info(
                    "New clustering model is better"
                )

                with mlflow.start_run():

                    mlflow.log_param(
                        "optimal_k",
                        optimal_k
                    )

                    mlflow.log_metric(
                        "silhouette_score",
                        best_silhouette
                    )

                    mlflow.sklearn.log_model(
                        sk_model=self.model,
                        artifact_path="model",
                        registered_model_name="Apex_Trust_Model"
                    )

                logging.info(
                    "Model logged successfully"
                )

            else:

                logging.info(
                    "Skipping model logging"
                )

        except Exception as e:

            logging.error(
                f"Error training clustering model: {e}"
            )

            raise

    # =========================================
    # APPLY CLUSTERING
    # =========================================
    def apply_clustering(self):

        try:

            setup_mlflow()

            model, optimal_k = (
                load_model_and_params()
            )

            logging.info(
                f"Applying clustering with "
                f"{optimal_k} clusters..."
            )

            # =====================================
            # ONLY RFM FEATURES
            # =====================================
            clustering_features = (
                self.processed_customer_data[
                    [
                        "Recency",
                        "Frequency",
                        "Monetary"
                    ]
                ]
            )

            self.customer_data["Cluster"] = (
                model.predict(
                    clustering_features
                )
            )

            logging.info(
                "Clustering applied successfully"
            )

            return self.customer_data

        except Exception as e:

            logging.error(
                f"Error applying clustering: {e}"
            )

            raise


# =========================================
# EXECUTION
# =========================================
if __name__ == "__main__":

    clustering_engine = ClusteringEngine()

    clustering_engine.train_and_log_model()

    clustered_customer_data = (
        clustering_engine.apply_clustering()
    )

    print(
        clustered_customer_data.head()
    )



