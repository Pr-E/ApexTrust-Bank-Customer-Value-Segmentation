# =========================================
# APEX TRUST SEGMENT CLASSIFIER
# =========================================

import pandas as pd
import logging
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (

    classification_report,

    accuracy_score,

    confusion_matrix
)

from src.modelling.segments import (
    main
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SegmentClassifier:

    def __init__(self):

        logging.info(
            "Initializing classifier..."
        )

        self.rfm_df, self.cluster_profile = (
            main()
        )

        # =====================================
        # FEATURES
        # =====================================

        self.features = [

            "Recency",

            "Frequency",

            "Monetary",

            "Age",

            "CustAccountBalance"
        ]

        # =====================================
        # TARGET
        # =====================================
        self.target = "Cluster_Name"

        self.model = None

        self.label_encoder = LabelEncoder()

        self.label_mapping = {}

        self.reverse_label_mapping = {}

    # =========================================
    # PREPARE DATA
    # =========================================
    def prepare_data(self):

        logging.info(
            "Preparing dataset..."
        )

        X = self.rfm_df[
            self.features
        ]

        y = self.label_encoder.fit_transform(
            self.rfm_df[self.target]
        )

        self.label_mapping = {

            label: int(idx)

            for idx, label in enumerate(
                self.label_encoder.classes_
            )
        }

        self.reverse_label_mapping = {

            int(idx): label

            for idx, label in enumerate(
                self.label_encoder.classes_
            )
        }

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,

                y,

                test_size=0.2,

                random_state=42,

                stratify=y
            )
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    # =========================================
    # TRAIN MODEL
    # =========================================
    def train_model(self):

        try:

            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = self.prepare_data()

            model = RandomForestClassifier(

                n_estimators=300,

                max_depth=8,

                min_samples_split=10,

                min_samples_leaf=5,

                max_features="sqrt",

                bootstrap=True,

                random_state=42,

                n_jobs=-1
            )

            model.fit(
                X_train,
                y_train
            )

            self.model = model

            predictions = model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            logging.info(
                f"Accuracy: {accuracy:.4f}"
            )

            print(
                classification_report(
                    y_test,
                    predictions,
                    target_names=(
                        self.label_encoder.classes_
                    )
                )
            )

            print(
                confusion_matrix(
                    y_test,
                    predictions
                )
            )

            # =====================================
            # FEATURE IMPORTANCE
            # =====================================
            feature_importance = pd.DataFrame({

                "Feature":
                    self.features,

                "Importance":
                    model.feature_importances_

            }).sort_values(
                by="Importance",
                ascending=False
            )

            print(feature_importance)

            # =====================================
            # LOG TO MLFLOW
            # =====================================
            with mlflow.start_run():

                mlflow.log_metric(
                    "classification_accuracy",
                    accuracy
                )

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="segment_classifier"
                )

            logging.info(
                "Classifier trained successfully"
            )

            return (
                model,
                X_test
            )

        except Exception as e:

            logging.error(
                f"Classifier error: {e}"
            )

            raise