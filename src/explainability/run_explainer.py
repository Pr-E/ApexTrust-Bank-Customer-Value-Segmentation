# APEX TRUST CUSTOMER EXPLAINABILITY
# =========================================

import shap
import logging

from src.modelling.segment_classifier import (
    SegmentClassifier
)

from src.utils.model_loader import (
    load_model_and_params
)

from src.explainability.customer_explainer import (
    explain_customer_segment
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================
# LOAD CLUSTER MODEL INFO
# =========================================
cluster_model, optimal_k = (
    load_model_and_params()
)

logging.info(
    f"Optimal K Loaded: {optimal_k}"
)


# =========================================
# INITIALIZE CLASSIFIER
# =========================================
classifier = SegmentClassifier()

classification_model, X_test = (
    classifier.train_model()
)

label_encoder = (
    classifier.label_encoder
)

logging.info(
    "LightGBM classifier "
    "trained successfully"
)


# =========================================
# CREATE SHAP EXPLAINER
# =========================================
explainer = shap.TreeExplainer(
    classification_model
)

logging.info(
    "SHAP TreeExplainer "
    "created successfully"
)


# =========================================
# SAMPLE CUSTOMER
# =========================================
sample_customer = X_test.iloc[[0]]

predicted_label = int(
    classification_model.predict(
        sample_customer
    )[0]
)

predicted_segment = (
    label_encoder.inverse_transform(
        [predicted_label]
    )[0]
)

logging.info(
    f"Predicted Segment: "
    f"{predicted_segment}"
)


# =========================================
# GENERATE EXPLANATION
# =========================================
response, report = (
    explain_customer_segment(
        model=classification_model,
        explainer=explainer,
        X=sample_customer,
        feature_names=sample_customer.columns,
        cluster_name=predicted_segment
    )
)


# =========================================
# PRINT RESULTS
# =========================================
print(
    "\n==================================="
)

print(
    "APEX TRUST CUSTOMER INTELLIGENCE"
)

print(
    "===================================\n"
)

print(
    f"Optimal Number of Clusters (K): "
    f"{optimal_k}\n"
)

print(report)

print("\nSTRUCTURED RESPONSE:\n")

print(response)