# =========================================
# STREAMLIT EXECUTIVE DASHBOARD
# =========================================

import logging

import streamlit as st
import pandas as pd
import numpy as np

from src.modelling.segments import (
    SegmentEngine
)

from src.visualization.customer_segment import (
    CustomerSegmentVisualisation
)

from src.visualization.customer_segment_performance import (
    CustomerSegmentPerformanceVisualisation
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# CONFIG
API_BASE = "http://13.62.100.198:8000"


# =========================================
# PAGE CONFIGURATION
# =========================================
st.set_page_config(

    page_title=
    "ApexTrust Customer Segmentation Dashboard",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded"
)


# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():

    logging.info(
        "Loading segmentation data..."
    )

    rfm_df, cluster_profile = (
        SegmentEngine.cluster_grouper()
    )

    return (
        rfm_df,
        cluster_profile
    )


rfm_df, cluster_profile = load_data()


# =========================================
# INITIALISE VISUALISERS
# =========================================
segment_visualizer = (
    CustomerSegmentVisualisation()
)

performance_visualizer = (
    CustomerSegmentPerformanceVisualisation()
)


# =========================================
# EXECUTIVE INSIGHT ENGINE
# =========================================
def generate_executive_insights(
    cluster_profile: pd.DataFrame
):

    insights = {

        "top_revenue_segment":

            cluster_profile[
                "Estimated_Total_Value"
            ].idxmax(),

        "highest_risk_segment":

            cluster_profile[
                "Avg_Recency"
            ].idxmax(),

        "most_active_segment":

            cluster_profile[
                "Avg_Frequency"
            ].idxmax(),

        "largest_segment":

            cluster_profile[
                "Customer_Count"
            ].idxmax()
    }

    return insights


# =========================================
# BUSINESS RECOMMENDATION ENGINE
# =========================================
def generate_segment_strategy(
    segment_name
):

    strategies = {

        "Loyal High-Value Customer":

            (
                "Focus on wealth retention, "
                "premium relationship management, "
                "investment products, and "
                "high-value loyalty programmes."
            ),

        "At-Risk Premium Customer":

            (
                "Deploy proactive reactivation "
                "campaigns, personalised retention "
                "offers, and relationship recovery "
                "strategies."
            ),

        "Active Everyday Customer":

            (
                "Cross-sell digital banking products, "
                "savings accounts, and customer "
                "loyalty initiatives."
            ),

        "Low Engagement Customer":

            (
                "Increase engagement through "
                "behavioural nudges, onboarding "
                "campaigns, and customer education."
            )
    }

    return strategies.get(
        segment_name,
        "Continue behavioural monitoring."
    )


executive_insights = (
    generate_executive_insights(
        cluster_profile
    )
)


# =========================================
# SIDEBAR
# =========================================
st.sidebar.title(
    "Apex Trust Dashboard"
)

st.sidebar.markdown(
    """
    ### Dashboard Sections

    - Executive KPI Overview
    - Customer Segmentation
    - Behaviour Intelligence
    - Revenue Intelligence
    - Retention Intelligence
    - Strategic Recommendations
    """
)

st.sidebar.markdown("---")

selected_segment = (
    st.sidebar.selectbox(

        "Select Segment",

        options=cluster_profile.index
    )
)


# =========================================
# DASHBOARD HEADER
# =========================================
st.title(
    "Apex Trust Executive Customer Segmentation Dashboard"
)

st.markdown(
    """
    Advanced customer intelligence dashboard
    for strategic segmentation, customer
    retention, behavioural analytics,
    and revenue optimisation.
    """
)


# =========================================
# KPI SECTION
# =========================================
st.subheader(
    "Executive KPI Overview"
)

total_customers = (
    cluster_profile[
        "Customer_Count"
    ].sum()
)

total_revenue = (
    cluster_profile[
        "Estimated_Total_Value"
    ].sum()
)

average_rfm_score = (
    cluster_profile[
        "Avg_RFM_Score"
    ].mean()
)

highest_balance = (
    cluster_profile[
        "Avg_Account_Balance"
    ].max()
)

col1, col2, col3, col4 = (
    st.columns(4)
)

col1.metric(
    "Total Customers",
    f"{total_customers:,.0f}"
)

col2.metric(
    "Estimated Revenue",
    f"£{total_revenue:,.0f}"
)

col3.metric(
    "Average RFM Score",
    f"{average_rfm_score:.2f}"
)

col4.metric(
    "Highest Avg Balance",
    f"£{highest_balance:,.0f}"
)




# =========================================
# CUSTOMER SEGMENTATION OVERVIEW
# =========================================
st.header(
    "Customer Segmentation Overview"
)

col5, col6 = st.columns(2)

with col5:

    fig1 = (
        segment_visualizer
        .plot_segment_distribution(
            rfm_df
        )
    )

    st.pyplot(
        fig1,
        use_container_width=True
    )

    st.info(
        """
        Insight:
        The customer distribution chart highlights
        the proportion of customers across behavioural
        segments. Active Everyday Customers represent
        the largest segment, indicating strong day-to-day
        banking engagement opportunities.
        """
    )

with col6:

    fig2 = (
        segment_visualizer
        .plot_segment_size(
            rfm_df
        )
    )

    st.pyplot(
        fig2,
        use_container_width=True
    )

    st.info(
        """
        Insight:
        Segment size analysis shows the relative scale
        of each customer group. Larger segments provide
        significant cross-sell opportunities, while
        smaller premium segments contribute
        disproportionate revenue value.
        """
    )


fig3 = (
    performance_visualizer
    .plot_revenue_distribution(
        cluster_profile
    )
)

st.pyplot(
    fig3,
    use_container_width=True
)

st.info(
    """
    Insight:
    Revenue concentration is heavily driven by
    Loyal High-Value Customers, demonstrating
    the importance of premium customer retention
    and relationship management strategies.
    """
)


# =========================================
# CUSTOMER BEHAVIOUR INTELLIGENCE
# =========================================
st.header(
    "Customer Behaviour Intelligence"
)

col7, col8 = st.columns(2)

with col7:

    fig4 = (
        segment_visualizer
        .plot_rfm_score_distribution(
            rfm_df
        )
    )

    st.pyplot(
        fig4,
        use_container_width=True
    )

    st.info(
        """
        Insight:
        RFM score distribution reveals clear behavioural
        separation between high-value loyal customers
        and low-engagement customers, validating
        effective customer segmentation.
        """
    )

with col8:

    fig5 = (
        segment_visualizer
        .plot_rfm_score_vs_monetary(
            rfm_df
        )
    )

    st.pyplot(
        fig5,
        use_container_width=True
    )

    st.info(
        """
        Insight:
        Customers with higher RFM scores consistently
        generate stronger monetary value, confirming
        that behavioural engagement strongly correlates
        with customer profitability.
        """
    )


fig6 = (
    segment_visualizer
    .plot_engagement_matrix(
        rfm_df
    )
)

st.pyplot(
    fig6,
    use_container_width=True
)

st.info(
    """
    Insight:
    The engagement matrix highlights behavioural
    intensity across customer groups. High-frequency
    and high-monetary customers represent strong
    retention and upselling opportunities.
    """
)


# =========================================
# REVENUE INTELLIGENCE
# =========================================
st.header(
    "Revenue Intelligence"
)

col9, col10 = st.columns(2)

with col9:

    fig7 = (
        performance_visualizer
        .plot_segment_heatmap(
            cluster_profile
        )
    )

    st.pyplot(
        fig7,
        use_container_width=True
    )

    st.info(
        """
        Insight:
        The segment performance heatmap provides
        comparative intelligence across behavioural,
        monetary, and engagement metrics, enabling
        rapid executive-level customer assessment.
        """
    )

with col10:

    fig8 = (
        performance_visualizer
        .plot_recency_heatmap(
            cluster_profile
        )
    )

    st.pyplot(
        fig8,
        use_container_width=True
    )

    st.info(
        """
        Insight:
        Higher recency values indicate increasing
        inactivity risk. At-Risk Premium Customers
        exhibit elevated churn behaviour despite
        historically strong monetary contribution.
        """
    )


# =========================================
# RADAR PERFORMANCE ANALYTICS
# =========================================
st.header(
    "Normalized Segment Performance Radar"
)

radar_fig = (
    performance_visualizer
    .plot_normalized_segment_radar_chart(
        cluster_profile
    )
)

st.plotly_chart(
    radar_fig,
    use_container_width=True
)

st.info(
    """
    Insight:
    The radar chart provides a normalized comparison
    of customer behavioural dimensions including
    recency, frequency, monetary value, and account
    balance. Loyal High-Value Customers demonstrate
    superior overall customer performance.
    """
)


# =========================================
# EXECUTIVE AI INSIGHTS
# =========================================
st.header(
    "Strategic Customer Intelligence Insights"
)

st.success(

    f"""
    Top Revenue Generating Segment:
    {executive_insights['top_revenue_segment']}

    Strategic Impact:
    This segment contributes the highest overall
    customer value and should remain a primary
    retention priority.
    """
)

st.warning(

    f"""
    Highest Retention Risk Segment:
    {executive_insights['highest_risk_segment']}

    Strategic Impact:
    Customers within this segment display elevated
    inactivity behaviour and require proactive
    engagement strategies.
    """
)

st.info(

    f"""
    Most Active Customer Segment:
    {executive_insights['most_active_segment']}

    Strategic Impact:
    This segment demonstrates strong transaction
    frequency and provides ideal cross-selling
    opportunities.
    """
)

st.error(

    f"""
    Largest Customer Segment:
    {executive_insights['largest_segment']}

    Strategic Impact:
    This segment represents the bank's largest
    behavioural customer population and significantly
    influences overall customer engagement trends.
    """
)


# =========================================
# STRATEGIC RECOMMENDATIONS
# =========================================
st.header(
    "Executive Strategic Recommendations"
)

strategy = (
    generate_segment_strategy(
        selected_segment
    )
)

st.markdown(

    f"""
    ### Selected Segment

    **{selected_segment}**

    ### Strategic Recommendation

    {strategy}
    """
)

st.dataframe(

    cluster_profile.loc[
        [selected_segment]
    ],

    use_container_width=True
)


# =========================================
# RAW DATA
# =========================================
st.header(
    "Cluster Profile Dataset"
)

st.dataframe(
    cluster_profile,
    use_container_width=True
)


# =========================================
# FOOTER
# =========================================
st.markdown("---")

st.caption(
    "Apex Trust Customer Segmentation Intelligence Platform"
)