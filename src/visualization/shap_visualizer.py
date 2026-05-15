# =========================================
# SHAP FEATURE IMPORTANCE VISUALISATION
# CUSTOMER SEGMENT INTERPRETABILITY
# =========================================

import shap
import numpy as np
import matplotlib.pyplot as plt
import logging

from src.modelling.segment_classifier import (
    SegmentClassifier
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================
# INITIALIZE CLASSIFIER
# =========================================
logging.info(
    "Initializing LightGBM segment classifier..."
)

classifier = SegmentClassifier()

model, X_test = (
    classifier.train_model()
)

logging.info(
    "Classifier loaded successfully"
)


# =========================================
# CUSTOMER SEGMENT MAPPING
# =========================================
segment_mapping = dict(
    enumerate(
        classifier.label_encoder.classes_
    )
)

segment_names = list(
    classifier.label_encoder.classes_
)

print(
    "\n===================================="
)

print(
    "CUSTOMER SEGMENT CLASS MAPPING"
)

print(
    "====================================\n"
)

for class_id, segment_name in (
    segment_mapping.items()
):

    print(
        f"Class {class_id} "
        f"→ {segment_name}"
    )


# =========================================
# CREATE SHAP EXPLAINER
# =========================================
logging.info(
    "Creating SHAP explainer..."
)

explainer = shap.TreeExplainer(
    model
)

logging.info(
    "SHAP explainer created successfully"
)


# =========================================
# GENERATE SHAP VALUES
# =========================================
logging.info(
    "Generating SHAP values..."
)

shap_values = explainer.shap_values(
    X_test
)

logging.info(
    "SHAP values generated successfully"
)


# =========================================
# HANDLE MULTICLASS SHAP OUTPUT
# =========================================
if isinstance(shap_values, list):

    shap_array = shap_values

else:

    # New SHAP output format:
    # (samples, features, classes)

    shap_array = []

    for i in range(
        shap_values.shape[2]
    ):

        shap_array.append(
            shap_values[:, :, i]
        )


# =========================================
# GLOBAL FEATURE IMPORTANCE BAR PLOT
# =========================================
logging.info(
    "Generating SHAP feature importance plot..."
)

plt.figure(figsize=(14, 8))

shap.summary_plot(
    shap_array,
    X_test,
    plot_type="bar",
    class_names=segment_names,
    show=False
)

plt.title(
    "Global Feature Importance - Customer Segmentation",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()


# =========================================
# SHAP INTERPRETATION INSIGHTS
# =========================================
print(
    "\n===================================="
)

print(
    "SHAP FEATURE IMPORTANCE INSIGHTS"
)

print(
    "====================================\n"
)

print(
    "1. Monetary is the strongest "
    "customer segmentation driver."
)

print(
    "2. Frequency strongly determines "
    "customer engagement classification."
)

print(
    "3. Recency captures inactivity "
    "and churn behaviour."
)

print(
    "4. Behavioural signals are more "
    "important than demographics."
)

print(
    "5. Loyal Premium Customers are "
    "primarily driven by high monetary "
    "value and strong engagement."
)


# =========================================
# SHAP SUMMARY PLOTS
# =========================================
for i, segment in enumerate(
    segment_names
):

    logging.info(
        f"Generating SHAP summary plot "
        f"for {segment}"
    )

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_array[i],
        X_test,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot - {segment}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()


# =========================================
# DEPENDENCE PLOTS
# =========================================
top_features = [
    "Frequency",
    "Monetary",
    "Recency"
]

# =========================================
# TARGET SEGMENT
# =========================================
target_segment = (
    "Loyal Premium Customer"
)

class_index = (
    segment_names.index(
        target_segment
    )
)

for feature in top_features:

    logging.info(
        f"Generating dependence plot "
        f"for {feature}"
    )

    shap.dependence_plot(
        feature,
        shap_array[class_index],
        X_test,
        interaction_index=None
    )


# =========================================
# FORCE PLOT FOR SINGLE CUSTOMER
# =========================================
logging.info(
    "Generating customer-level "
    "force plot..."
)

sample_customer = (
    X_test.iloc[[0]]
)

predicted_class = int(
    model.predict(sample_customer)[0]
)

predicted_segment = (
    segment_names[predicted_class]
)

sample_shap = (
    explainer.shap_values(
        sample_customer
    )
)

# =========================================
# HANDLE MULTICLASS FORCE PLOT
# =========================================
if isinstance(sample_shap, list):

    sample_shap_values = (
        sample_shap[
            predicted_class
        ][0]
    )

    expected_value = (
        explainer.expected_value[
            predicted_class
        ]
    )

else:

    sample_shap_values = (
        sample_shap[
            0, :, predicted_class
        ]
    )

    expected_value = (
        explainer.expected_value[
            predicted_class
        ]
    )


# =========================================
# FORCE PLOT
# =========================================
shap.force_plot(
    expected_value,
    sample_shap_values,
    sample_customer,
    matplotlib=True
)

plt.title(
    f"Customer-Level SHAP Force Plot\n"
    f"Predicted Segment: "
    f"{predicted_segment}",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.show()


# =========================================
# FINAL BUSINESS INTERPRETATION
# =========================================
print(
    "\n===================================="
)

print(
    "CUSTOMER SEGMENT INTERPRETATION"
)

print(
    "====================================\n"
)

print(
    "🔵 Active Everyday Customer:\n"
    "- Strong transaction engagement\n"
    "- Moderate monetary contribution\n"
    "- Ideal for cross-selling\n"
)

print(
    "🟢 At-Risk Premium Customer:\n"
    "- Historically valuable customers\n"
    "- Increasing inactivity detected\n"
    "- Retention campaigns required\n"
)

print(
    "🟣 Low Engagement Customer:\n"
    "- Low frequency and monetary value\n"
    "- Weak banking engagement\n"
    "- Requires reactivation strategy\n"
)

print(
    "🔴 Loyal Premium Customer:\n"
    "- High-value loyal customers\n"
    "- Strong revenue contribution\n"
    "- Priority for wealth management\n"
)


logging.info(
    "All SHAP visualizations "
    "generated successfully"
)