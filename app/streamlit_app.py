# =========================================
# APEX TRUST CUSTOMER INTELLIGENCE APP
# =========================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="ApexTrust Bank Intelligence",
    page_icon="🏦",
    layout="wide"
)

# =========================================
# CUSTOM STYLING
# =========================================
st.markdown("""
<style>

/* =========================================
MAIN BACKGROUND
========================================= */
.main {
    background-color: #f8fafc;
}

/* =========================================
PAGE CONTAINER
========================================= */
.block-container {
    padding-top: 2rem;
}

/* =========================================
HEADINGS
========================================= */
h1, h2, h3 {
    color: #0f172a;
}

/* =========================================
METRIC CARDS
========================================= */
[data-testid="stMetric"] {

    background-color: white;

    padding: 18px;

    border-radius: 12px;

    border: 1px solid #e2e8f0;

    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}

/* =========================================
METRIC LABELS
========================================= */
[data-testid="stMetricLabel"] {

    color: #475569 !important;

    font-size: 15px !important;

    font-weight: 600 !important;
}

/* =========================================
METRIC VALUES
========================================= */
[data-testid="stMetricValue"] {

    color: #0f172a !important;

    font-size: 28px !important;

    font-weight: bold !important;
}

/* =========================================
BUTTON
========================================= */
.stButton > button {

    background-color: #0f172a;

    color: white !important;

    border-radius: 10px;

    border: none;

    padding: 0.7rem 1rem;

    font-weight: 600;

    width: 100%;
}

/* =========================================
BUTTON HOVER
========================================= */
.stButton > button:hover {

    background-color: #1e293b;

    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.title(
    "🏦 ApexTrust Bank Customer Intelligence Platform"
)

st.markdown("""
### AI-Powered Customer Segmentation & Explainable Intelligence System

This platform enables:

- Customer value segmentation
- Explainable AI-driven insights
- Business recommendation intelligence
- SHAP-powered feature interpretation
- Stakeholder-focused customer analytics
""")

st.markdown("---")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title(
    "📥 Customer Input Panel"
)

st.sidebar.markdown("""
Adjust customer behavioural and financial
attributes to generate intelligent customer insights.
""")

# =========================================
# USER INPUTS
# =========================================
recency = st.sidebar.slider(
    "Recency (Days Since Last Transaction)",
    min_value=1,
    max_value=1200,
    value=45
)

frequency = st.sidebar.slider(
    "Transaction Frequency",
    min_value=1,
    max_value=200,
    value=18
)

monetary = st.sidebar.number_input(
    "Total Transaction Value (£)",
    min_value=0.0,
    value=84000.0,
    step=1000.0
)

age = st.sidebar.slider(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=42
)

account_balance = st.sidebar.number_input(
    "Account Balance (£)",
    min_value=0.0,
    value=30000.0,
    step=1000.0
)

# =========================================
# API PAYLOAD
# =========================================
payload = {

    "Recency": recency,

    "Frequency": frequency,

    "Monetary": monetary,

    "Age": age,

    "CustAccountBalance": account_balance
}

# =========================================
# GENERATE BUTTON
# =========================================
if st.sidebar.button(
    "🚀 Generate Customer Intelligence"
):

    with st.spinner(
        "Analyzing customer behaviour and generating intelligence..."
    ):

        try:

            # =====================================
            # API REQUEST
            # =====================================
            response = requests.post(
                "http://127.0.0.1:8000/explain",
                json=payload
            )

            result = response.json()

            explanation = (
                result["structured_output"]
            )

            engineered_features = (
                result["engineered_features"]
            )

            customer_value_level = (
                result["customer_value_level"]
            )

            # =====================================
            # CUSTOMER SUMMARY
            # =====================================
            st.markdown("""
            ## 📌 Customer Intelligence Summary
            """)

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Segment",
                    explanation["cluster_name"]
                )

            with col2:

                st.metric(
                    "Recommended Action",
                    explanation["recommended_action"]
                )

            with col3:

                st.metric(
                    "Customer Value Level",
                    customer_value_level
                )

            st.markdown("---")

            # =====================================
            # ENGINEERED RFM SCORES
            # =====================================
            st.subheader(
                "⚙️ Engineered Customer Scores"
            )

            score_col1, score_col2, score_col3 = (
                st.columns(3)
            )

            with score_col1:

                st.metric(
                    "R Score",
                    engineered_features["R_Score"]
                )

            with score_col2:

                st.metric(
                    "F Score",
                    engineered_features["F_Score"]
                )

            with score_col3:

                st.metric(
                    "M Score",
                    engineered_features["M_Score"]
                )

            st.markdown("---")

            # =====================================
            # CUSTOMER PROFILE OVERVIEW
            # =====================================
            st.subheader(
                "👤 Customer Behaviour Overview"
            )

            overview_col1, overview_col2 = (
                st.columns(2)
            )

            with overview_col1:

                st.info(f"""
                **Recency:** {recency} days

                **Frequency:** {frequency} transactions

                **Monetary Value:** £{monetary:,.2f}
                """)

            with overview_col2:

                st.info(f"""
                **Customer Age:** {age}

                **Account Balance:** £{account_balance:,.2f}

                **Customer Value Level:** {customer_value_level}
                """)

            # =====================================
            # BUSINESS STRATEGY
            # =====================================
            st.subheader(
                "💼 Recommended Business Strategy"
            )

            st.success(
                explanation["business_strategy"]
            )

            # =====================================
            # KEY DRIVERS
            # =====================================
            st.subheader(
                "🧠 Key Business Drivers"
            )

            for reason in explanation[
                "reason_codes"
            ]:

                st.markdown(
                    f"✅ {reason}"
                )

            st.markdown("---")

            # =====================================
            # DRIVER TABLES
            # =====================================
            col1, col2 = st.columns(2)

            # =====================================
            # POSITIVE DRIVERS
            # =====================================
            with col1:

                st.subheader(
                    "📈 Positive Drivers"
                )

                positive_df = pd.DataFrame(
                    explanation["top_drivers"]
                )

                st.dataframe(
                    positive_df,
                    use_container_width=True
                )

            # =====================================
            # NEGATIVE DRIVERS
            # =====================================
            with col2:

                st.subheader(
                    "📉 Negative Drivers"
                )

                negative_df = pd.DataFrame(
                    explanation[
                        "protective_factors"
                    ]
                )

                st.dataframe(
                    negative_df,
                    use_container_width=True
                )

            st.markdown("---")

            # =====================================
            # SHAP VISUALIZATION
            # =====================================
            st.subheader(
                "📊 SHAP Feature Impact Analysis"
            )

            impact_df = pd.concat([

                positive_df,

                negative_df

            ])

            fig = px.bar(

                impact_df,

                x="impact",

                y="feature",

                orientation="h",

                text="impact",

                title=(
                    "Feature Contribution Toward "
                    "Customer Segment Prediction"
                )
            )

            fig.update_layout(

                height=500,

                title_x=0.1
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =====================================
            # CUSTOMER VALUE GAUGE
            # =====================================
            st.subheader(
                "🎯 Customer Value Strength"
            )

            average_score = (

                engineered_features["R_Score"]

                + engineered_features["F_Score"]

                + engineered_features["M_Score"]

            ) / 3

            gauge_fig = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=average_score,

                    title={
                        "text": "Average Customer Value Score"
                    },

                    gauge={

                        "axis": {
                            "range": [1, 5]
                        },

                        "bar": {
                            "color": "darkblue"
                        },

                        "steps": [

                            {
                                "range": [1, 2],
                                "color": "#fecaca"
                            },

                            {
                                "range": [2, 3],
                                "color": "#fde68a"
                            },

                            {
                                "range": [3, 4],
                                "color": "#bfdbfe"
                            },

                            {
                                "range": [4, 5],
                                "color": "#bbf7d0"
                            }
                        ]
                    }
                )
            )

            gauge_fig.update_layout(
                height=350
            )

            st.plotly_chart(
                gauge_fig,
                use_container_width=True
            )

            # =====================================
            # FULL REPORT
            # =====================================
            with st.expander(
                "📄 Full Customer Intelligence Report"
            ):

                st.text(
                    result["customer_review"]
                )

            # =====================================
            # JSON OUTPUT
            # =====================================
            with st.expander(
                "🧾 Structured JSON Output"
            ):

                st.json(
                    result
                )

        except Exception as e:

            st.error(
                f"API Error: {e}"
            )

            st.warning(
                "Ensure the FastAPI server is running on port 8000."
            )

# =========================================
# FOOTER
# =========================================
st.markdown("---")

st.caption(
    "ApexTrust Bank | Explainable AI Customer Intelligence Platform"
)