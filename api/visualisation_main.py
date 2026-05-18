from functools import lru_cache

import io
import base64
import logging

import matplotlib.pyplot as plt
import plotly.io as pio

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.modelling.clusters import (
    ClusteringEngine
)

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


app = FastAPI(
    title="Apex Trust Customer Segmentation API"
)


# =========================================
# ROOT ROUTE
# =========================================
@app.get("/")
def root():

    return {

        "message":
        "Apex Trust Customer Segmentation API Running",

        "dashboard":
        "/dashboard",

        "docs":
        "/docs"
    }


# =========================================
# INITIALISE VISUALISATION CLASSES
# =========================================
segment_visualisation = (
    CustomerSegmentVisualisation()
)

segment_performance_visualisation = (
    CustomerSegmentPerformanceVisualisation()
)


# =========================================
# CACHE PIPELINE DATA
# =========================================
@lru_cache(maxsize=1)
def get_pipeline_data():

    logging.info(
        "Loading segmentation pipeline data..."
    )

    segmented_customer_data, customer_segment_profile = (
        SegmentEngine.cluster_grouper()
    )

    return (
        segmented_customer_data,
        customer_segment_profile
    )


# =========================================
# CONVERT MATPLOTLIB FIGURE TO BASE64
# =========================================
def fig_to_base64(fig):

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight"
    )

    buffer.seek(0)

    image_base64 = (
        base64.b64encode(
            buffer.read()
        ).decode("utf-8")
    )

    plt.close(fig)

    return image_base64


# =========================================
# CONVERT PLOTLY FIGURE TO HTML
# =========================================
def plotly_to_html(fig):

    return pio.to_html(
        fig,
        full_html=False
    )


# =========================================
# REFRESH CACHE
# =========================================
@app.post("/refresh")
def refresh_data():

    get_pipeline_data.cache_clear()

    logging.info(
        "Pipeline cache refreshed"
    )

    return {

        "message":
        "Cache cleared successfully"
    }


# =========================================
# GET SEGMENT DATA
# =========================================
@app.get("/segments")
def get_segments():

    segmented_customer_data, customer_segment_profile = (
        get_pipeline_data()
    )

    return {

        "segmented_data":

        segmented_customer_data.to_dict(
            orient="records"
        ),

        "cluster_summary":

        customer_segment_profile.reset_index()
        .to_dict(
            orient="records"
        )
    }


# =========================================
# DASHBOARD ENDPOINT
# =========================================
@app.get(
    "/dashboard",
    response_class=HTMLResponse
)
def dashboard():

    segmented_customer_data, customer_segment_profile = (
        get_pipeline_data()
    )

    logging.info(
        "Generating dashboard visualisations..."
    )

    # =====================================
    # CUSTOMER SEGMENT VISUALISATIONS
    # =====================================
    fig1 = (
        segment_visualisation
        .plot_segment_distribution(
            segmented_customer_data
        )
    )

    segment_distribution_plot = (
        fig_to_base64(fig1)
    )

    fig2 = (
        segment_visualisation
        .plot_segment_size(
            segmented_customer_data
        )
    )

    segment_size_plot = (
        fig_to_base64(fig2)
    )

    fig3 = (
        segment_visualisation
        .plot_rfm_score_distribution(
            segmented_customer_data
        )
    )

    rfm_distribution_plot = (
        fig_to_base64(fig3)
    )

    fig4 = (
        segment_visualisation
        .plot_rfm_score_vs_monetary(
            segmented_customer_data
        )
    )

    rfm_vs_monetary_plot = (
        fig_to_base64(fig4)
    )

    fig5 = (
        segment_visualisation
        .plot_recency_vs_monetary(
            segmented_customer_data
        )
    )

    recency_vs_monetary_plot = (
        fig_to_base64(fig5)
    )

    fig6 = (
        segment_visualisation
        .plot_engagement_matrix(
            segmented_customer_data
        )
    )

    engagement_matrix_plot = (
        fig_to_base64(fig6)
    )

    # =====================================
    # PERFORMANCE VISUALISATIONS
    # =====================================
    fig7 = (
        segment_performance_visualisation
        .plot_revenue_distribution(
            customer_segment_profile
        )
    )

    revenue_distribution_plot = (
        fig_to_base64(fig7)
    )

    fig8 = (
        segment_performance_visualisation
        .plot_recency_heatmap(
            customer_segment_profile
        )
    )

    recency_heatmap_plot = (
        fig_to_base64(fig8)
    )

    fig9 = (
        segment_performance_visualisation
        .plot_segment_heatmap(
            customer_segment_profile
        )
    )

    segment_heatmap_plot = (
        fig_to_base64(fig9)
    )

    # =====================================
    # PLOTLY RADAR CHART
    # =====================================
    radar_fig = (
        segment_performance_visualisation
        .plot_normalized_segment_radar_chart(
            customer_segment_profile
        )
    )

    radar_chart_html = (
        plotly_to_html(
            radar_fig
        )
    )

    # =====================================
    # HTML DASHBOARD
    # =====================================
    return f"""

    <html>

    <head>

        <title>
            Apex Trust Customer Segmentation Dashboard
        </title>

        <style>

            body {{

                font-family: Arial;
                background-color: #f5f5f5;
                margin: 40px;
            }}

            h1 {{

                color: #222222;
            }}

            h2 {{

                margin-top: 50px;
                color: #444444;
            }}

            img {{

                width: 100%;
                border-radius: 12px;
                margin-bottom: 40px;
                background-color: white;
                padding: 10px;
                box-shadow:
                    0px 2px 10px
                    rgba(0,0,0,0.1);
            }}

            .chart-container {{

                margin-bottom: 60px;
            }}

        </style>

    </head>

    <body>

        <h1>
            Apex Trust Customer Segmentation Dashboard
        </h1>

        <div class="chart-container">

            <h2>
                Customer Segment Distribution
            </h2>

            <img src="data:image/png;base64,{segment_distribution_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                Customer Segment Sizes
            </h2>

            <img src="data:image/png;base64,{segment_size_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                RFM Score Distribution
            </h2>

            <img src="data:image/png;base64,{rfm_distribution_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                RFM Score vs Monetary Value
            </h2>

            <img src="data:image/png;base64,{rfm_vs_monetary_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                Recency vs Monetary Distribution
            </h2>

            <img src="data:image/png;base64,{recency_vs_monetary_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                Customer Engagement Matrix
            </h2>

            <img src="data:image/png;base64,{engagement_matrix_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                Revenue Distribution by Segment
            </h2>

            <img src="data:image/png;base64,{revenue_distribution_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                Customer Recency Heatmap
            </h2>

            <img src="data:image/png;base64,{recency_heatmap_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                Customer Segment Performance Heatmap
            </h2>

            <img src="data:image/png;base64,{segment_heatmap_plot}" />

        </div>

        <div class="chart-container">

            <h2>
                Segment Performance Radar Chart
            </h2>

            {radar_chart_html}

        </div>

    </body>

    </html>
    """


# =========================================
# RETRAIN MODEL
# =========================================
@app.get("/retrain")
def retrain_model():

    try:

        logging.info(
            "Retraining clustering model..."
        )

        cluster_engine = (
            ClusteringEngine()
        )

        cluster_engine.train_and_log_model()

        get_pipeline_data.cache_clear()

        logging.info(
            "Model retrained successfully"
        )

        return {

            "message":
            "Model retrained successfully"
        }

    except Exception as e:

        logging.error(
            f"Retraining error: {e}"
        )

        return {

            "error":
            str(e)
        }