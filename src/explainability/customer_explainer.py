# =========================================
# CUSTOMER SEGMENT EXPLAINER
# =========================================

import shap
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================
# BUSINESS REASON CODES
# =========================================
reason_code_map = {

    "Recency":
        "Customer recency behaviour",

    "Frequency":
        "Transaction engagement frequency",

    "Monetary":
        "Customer transaction value",

    "Age":
        "Customer demographic influence",

    "CustAccountBalance":
        "Account balance contribution"
}


# =========================================
# BUSINESS RECOMMENDATION ENGINE
# =========================================
def generate_business_recommendation(
    cluster_name
):

    # =====================================
    # LOYAL HIGH VALUE CUSTOMER
    # =====================================
    if cluster_name == (
        "Loyal High-Value Customer"
    ):

        return {

            "recommended_action":
                "Retain & Upsell",

            "business_strategy":
                (
                    "Provide premium banking "
                    "services, personalised offers, "
                    "investment products, loyalty "
                    "programs, and wealth retention "
                    "campaigns."
                )
        }

    # =====================================
    # AT-RISK PREMIUM CUSTOMER
    # =====================================
    elif cluster_name == (
        "At-Risk Premium Customer"
    ):

        return {

            "recommended_action":
                "Reactivate",

            "business_strategy":
                (
                    "Launch proactive retention "
                    "campaigns, personalised outreach, "
                    "relationship management support, "
                    "and premium recovery incentives."
                )
        }

    # =====================================
    # ACTIVE EVERYDAY CUSTOMER
    # =====================================
    elif cluster_name == (
        "Active Everyday Customer"
    ):

        return {

            "recommended_action":
                "Cross-Sell",

            "business_strategy":
                (
                    "Promote savings plans, credit "
                    "products, digital banking adoption, "
                    "and customer loyalty programs."
                )
        }

    # =====================================
    # LOW ENGAGEMENT CUSTOMER
    # =====================================
    elif cluster_name == (
        "Low Engagement Customer"
    ):

        return {

            "recommended_action":
                "Engage",

            "business_strategy":
                (
                    "Increase engagement through "
                    "financial education, onboarding "
                    "campaigns, behavioural nudges, "
                    "and customer reactivation programs."
                )
        }

    # =====================================
    # DEFAULT
    # =====================================
    else:

        return {

            "recommended_action":
                "Monitor",

            "business_strategy":
                (
                    "Continue monitoring customer "
                    "behaviour to identify future "
                    "growth and retention opportunities."
                )
        }


# =========================================
# CUSTOMER SEGMENT EXPLAINER
# =========================================
def explain_customer_segment(

    model,
    explainer,
    X,
    feature_names,
    cluster_name=None
):

    try:

        # =====================================
        # MODEL PREDICTION
        # =====================================
        prediction = int(
            model.predict(X)[0]
        )

        logging.info(
            f"Predicted Segment: {cluster_name}"
        )

        # =====================================
        # SHAP VALUES
        # =====================================
        shap_values = (
            explainer.shap_values(X)
        )

        # =====================================
        # HANDLE MULTI-CLASS OUTPUT
        # =====================================
        if isinstance(shap_values, list):

            shap_values = (
                shap_values[prediction][0]
            )

        else:

            if len(shap_values.shape) == 3:

                shap_values = (

                    shap_values[
                        0, :, prediction
                    ]
                )

            else:

                shap_values = (
                    shap_values[0]
                )

        # =====================================
        # CONVERT TO NUMERIC ARRAY
        # =====================================
        shap_values = np.array(
            shap_values,
            dtype=float
        ).flatten()

        # =====================================
        # FEATURE VALUES
        # =====================================
        feature_values = (
            X.iloc[0].values
        )

        # =====================================
        # FEATURE CONTRIBUTIONS
        # =====================================
        contributions = list(

            zip(
                feature_names,
                feature_values,
                shap_values
            )
        )

        # =====================================
        # SORT BY ABSOLUTE IMPACT
        # =====================================
        contributions = sorted(

            contributions,

            key=lambda x:
            abs(float(x[2])),

            reverse=True
        )

        # =====================================
        # REMOVE LOW-IMPACT FEATURES
        # =====================================
        contributions = [

            contribution

            for contribution in contributions

            if abs(float(contribution[2])) > 0.001
        ]

        # =====================================
        # TOP DRIVERS
        # =====================================
        top_drivers = (
            contributions[:5]
        )

        positive_drivers = [

            contribution

            for contribution in contributions

            if float(contribution[2]) > 0

        ][:5]

        negative_drivers = [

            contribution

            for contribution in contributions

            if float(contribution[2]) < 0

        ][:5]

        # =====================================
        # REASON CODES
        # =====================================
        reason_codes = [

            reason_code_map.get(
                feature,
                feature
            )

            for feature, _, _
            in top_drivers
        ]

        # =====================================
        # BUSINESS RECOMMENDATION
        # =====================================
        business_action = (

            generate_business_recommendation(
                cluster_name
            )
        )

        # =====================================
        # STRUCTURED RESPONSE
        # =====================================
        response = {

            "predicted_cluster":
                prediction,

            "cluster_name":
                cluster_name,

            "recommended_action":
                business_action[
                    "recommended_action"
                ],

            "business_strategy":
                business_action[
                    "business_strategy"
                ],

            "reason_codes":
                reason_codes,

            "top_drivers": [

                {

                    "feature":
                        feature,

                    "value":
                        round(float(value), 2),

                    "impact":
                        round(
                            float(impact),
                            4
                        )
                }

                for feature, value, impact
                in positive_drivers
            ],

            "protective_factors": [

                {

                    "feature":
                        feature,

                    "value":
                        round(float(value), 2),

                    "impact":
                        round(
                            float(impact),
                            4
                        )
                }

                for feature, value, impact
                in negative_drivers
            ]
        }

        # =====================================
        # HUMAN-READABLE REPORT
        # =====================================
        report = f"""
========================================
APEX TRUST CUSTOMER REVIEW
========================================

Predicted Segment:
{cluster_name}

----------------------------------------
KEY BUSINESS DRIVERS
----------------------------------------
"""

        for reason in reason_codes:

            report += (
                f" - {reason}\n"
            )

        report += """

----------------------------------------
TOP POSITIVE DRIVERS
----------------------------------------
"""

        if len(positive_drivers) > 0:

            for feature, value, impact in (
                positive_drivers
            ):

                report += (

                    f" - {feature}: "
                    f"{value:.2f} "
                    f"(Impact: +{impact:.4f})\n"
                )

        else:

            report += (
                " - No major positive "
                "drivers detected.\n"
            )

        report += """

----------------------------------------
TOP NEGATIVE DRIVERS
----------------------------------------
"""

        if len(negative_drivers) > 0:

            for feature, value, impact in (
                negative_drivers
            ):

                report += (

                    f" - {feature}: "
                    f"{value:.2f} "
                    f"(Impact: {impact:.4f})\n"
                )

        else:

            report += (
                " - No major negative "
                "drivers detected.\n"
            )

        report += f"""

----------------------------------------
BUSINESS RECOMMENDATION
----------------------------------------

Recommended Action:
{business_action['recommended_action']}

Business Strategy:
{business_action['business_strategy']}
"""

        logging.info(
            "Customer explanation generated successfully"
        )

        return response, report

    except Exception as e:

        logging.error(
            f"Error generating SHAP explanation: {e}"
        )

        raise