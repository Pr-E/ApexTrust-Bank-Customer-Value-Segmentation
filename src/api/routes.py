# =========================================
# APEX TRUST API ROUTES
# =========================================

import logging
import pandas as pd
import shap

from fastapi import APIRouter

from src.api.schemas import CustomerRequest

from src.modelling.segment_classifier import (
    SegmentClassifier
)

from src.explainability.customer_explainer import (
    explain_customer_segment
)

# =========================================
# LOGGING
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================
# ROUTER
# =========================================
router = APIRouter()

# =========================================
# LOAD CLASSIFIER
# =========================================
logging.info(
    "Loading segmentation classifier..."
)

classifier = SegmentClassifier()

model, X_test = classifier.train_model()

logging.info(
    "Segmentation classifier loaded successfully"
)

# =========================================
# SHAP EXPLAINER
# =========================================
explainer = shap.TreeExplainer(model)

logging.info(
    "SHAP explainer initialized successfully"
)

# =========================================
# LABEL MAPPING
# =========================================
label_mapping = (
    classifier.reverse_label_mapping
)

# =========================================
# HEALTH CHECK
# =========================================
@router.get("/")
def health_check():

    return {

        "message":
            "ApexTrust Bank Customer Intelligence API Running"
    }


# =========================================
# RECENCY SCORE
# =========================================
def generate_r_score(recency):

    if recency <= 30:

        return 5

    elif recency <= 90:

        return 4

    elif recency <= 180:

        return 3

    elif recency <= 365:

        return 2

    else:

        return 1


# =========================================
# FREQUENCY SCORE
# =========================================
def generate_f_score(frequency):

    if frequency >= 25:

        return 5

    elif frequency >= 18:

        return 4

    elif frequency >= 10:

        return 3

    elif frequency >= 5:

        return 2

    else:

        return 1


# =========================================
# MONETARY SCORE
# =========================================
def generate_m_score(monetary):

    if monetary >= 80000:

        return 5

    elif monetary >= 40000:

        return 4

    elif monetary >= 15000:

        return 3

    elif monetary >= 5000:

        return 2

    else:

        return 1


# =========================================
# CUSTOMER VALUE LEVEL
# =========================================
def generate_customer_value_level(
    r_score,
    f_score,
    m_score
):

    average_score = (
        r_score +
        f_score +
        m_score
    ) / 3

    if average_score >= 4:

        return "High Value"

    elif average_score >= 3:

        return "Moderate Value"

    else:

        return "Low Value"


# =========================================
# BUILD MODEL INPUT
# ONLY TRUE MODEL FEATURES
# =========================================
def build_customer_dataframe(customer):

    input_data = pd.DataFrame([{

        "Recency":
            customer.Recency,

        "Frequency":
            customer.Frequency,

        "Monetary":
            customer.Monetary,

        "Age":
            customer.Age,

        "CustAccountBalance":
            customer.CustAccountBalance
    }])

    return input_data


# =========================================
# PREDICT CUSTOMER SEGMENT
# =========================================
@router.post("/predict")
def predict_customer_segment(
    customer: CustomerRequest
):

    try:

        # =====================================
        # MODEL FEATURES
        # =====================================
        input_data = build_customer_dataframe(
            customer
        )

        # =====================================
        # MODEL PREDICTION
        # =====================================
        prediction = int(
            model.predict(input_data)[0]
        )

        cluster_name = (
            label_mapping[prediction]
        )

        # =====================================
        # BUSINESS RFM SCORES
        # REPORTING ONLY
        # =====================================
        r_score = generate_r_score(
            customer.Recency
        )

        f_score = generate_f_score(
            customer.Frequency
        )

        m_score = generate_m_score(
            customer.Monetary
        )

        customer_value_level = (
            generate_customer_value_level(
                r_score,
                f_score,
                m_score
            )
        )

        # =====================================
        # RESPONSE
        # =====================================
        return {

            "predicted_cluster":
                prediction,

            "cluster_name":
                cluster_name,

            "customer_value_level":
                customer_value_level,

            "engineered_features": {

                "R_Score":
                    r_score,

                "F_Score":
                    f_score,

                "M_Score":
                    m_score
            }
        }

    except Exception as e:

        logging.error(
            f"Prediction error: {e}"
        )

        return {

            "error":
                str(e)
        }


# =========================================
# CUSTOMER EXPLAINABILITY
# =========================================
@router.post("/explain")
def explain_customer(
    customer: CustomerRequest
):

    try:

        # =====================================
        # MODEL FEATURES
        # =====================================
        input_data = build_customer_dataframe(
            customer
        )

        # =====================================
        # PREDICTION
        # =====================================
        prediction = int(
            model.predict(input_data)[0]
        )

        cluster_name = (
            label_mapping[prediction]
        )

        # =====================================
        # BUSINESS RFM SCORES
        # REPORTING ONLY
        # =====================================
        r_score = generate_r_score(
            customer.Recency
        )

        f_score = generate_f_score(
            customer.Frequency
        )

        m_score = generate_m_score(
            customer.Monetary
        )

        customer_value_level = (
            generate_customer_value_level(
                r_score,
                f_score,
                m_score
            )
        )

        # =====================================
        # SHAP EXPLAINABILITY
        # =====================================
        response, report = (

            explain_customer_segment(

                model=model,

                explainer=explainer,

                X=input_data,

                feature_names=input_data.columns,

                cluster_name=cluster_name
            )
        )

        # =====================================
        # FINAL RESPONSE
        # =====================================
        return {

            "structured_output":
                response,

            "customer_review":
                report,

            "customer_value_level":
                customer_value_level,

            "engineered_features": {

                "R_Score":
                    r_score,

                "F_Score":
                    f_score,

                "M_Score":
                    m_score
            }
        }

    except Exception as e:

        logging.error(
            f"Explainability error: {e}"
        )

        return {

            "error":
                str(e)
        }