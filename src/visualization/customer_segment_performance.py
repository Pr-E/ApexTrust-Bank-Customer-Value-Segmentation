import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomerSegmentPerformanceVisualisation:

    def __init__(self):

        self.palette = {

            "Loyal High-Value Customer":
                "#2ca02c",

            "At-Risk Premium Customer":
                "#d62728",

            "Active Everyday Customer":
                "#17becf",

            "Low Engagement Customer":
                "#ff7f0e"
        }

    # =========================================
    # 1. REVENUE DISTRIBUTION DONUT CHART
    # =========================================
    def plot_revenue_distribution(
        self,
        cluster_profile: pd.DataFrame
    ):

        df = cluster_profile.copy()

        if "Estimated_Total_Value" not in df.columns:

            df["Estimated_Total_Value"] = (

                df["Avg_Monetary"]

                *

                df["Customer_Count"]
            )

        df = df.sort_values(

            by="Estimated_Total_Value",

            ascending=False
        )

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        colors = [

            self.palette.get(
                segment,
                "#333333"
            )

            for segment in df.index
        ]

        explode = [

            0.08 if i < 2 else 0

            for i in range(len(df))
        ]

        wedges, texts, autotexts = ax.pie(

            df["Estimated_Total_Value"],

            labels=None,

            autopct=lambda p:
            f"{p:.1f}%",

            colors=colors,

            explode=explode,

            shadow=True,

            pctdistance=0.85,

            textprops={
                "fontsize": 10,
                "fontweight": "bold"
            }
        )

        centre_circle = plt.Circle(
            (0, 0),
            0.70,
            fc="white"
        )

        ax.add_artist(
            centre_circle
        )

        for autotext in autotexts:

            autotext.set_color(
                "white"
            )

            autotext.set_fontweight(
                "bold"
            )

        total_sum = (

            df["Estimated_Total_Value"]

            .sum()
        )

        legend_labels = [

            f"{segment}: "

            f"£{row['Estimated_Total_Value']:,.0f} "

            f"({row['Estimated_Total_Value']/total_sum*100:.1f}%)"

            for segment, row in df.iterrows()
        ]

        ax.legend(

            wedges,

            legend_labels,

            title="Customer Segments",

            loc="center left",

            bbox_to_anchor=(1, 0.5)
        )

        ax.set_title(

            "Revenue Distribution by Segment",

            fontsize=16,

            fontweight="bold"
        )

        plt.tight_layout()

        return fig

    # =========================================
    # 2. NORMALIZED SEGMENT RADAR CHART
    # =========================================
    def plot_normalized_segment_radar_chart(
        self,
        cluster_profile: pd.DataFrame
    ):

        metrics = [

            "Avg_Recency",

            "Avg_Frequency",

            "Avg_Monetary",

            "Avg_Account_Balance"
        ]

        normalized = (
            cluster_profile[
                metrics
            ].copy()
        )

        for col in metrics:

            normalized[col] = (

                (
                    normalized[col]
                    - normalized[col].min()
                )

                /

                (
                    normalized[col].max()
                    - normalized[col].min()
                )
            )

        fig = go.Figure()

        for segment in normalized.index:

            fig.add_trace(

                go.Scatterpolar(

                    r=normalized.loc[
                        segment,
                        metrics
                    ].values,

                    theta=metrics,

                    fill="toself",

                    name=segment
                )
            )

        fig.update_layout(

            polar=dict(

                radialaxis=dict(

                    visible=True,

                    range=[0, 1]
                )
            ),

            title=(
                "Normalized Segment "
                "Performance Radar"
            ),

            showlegend=True
        )

        return fig


    # =========================================
    # 3. RECENCY HEATMAP
    # =========================================
    def plot_recency_heatmap(
        self,
        cluster_profile: pd.DataFrame
    ):

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.heatmap(

            cluster_profile[
                ["Avg_Recency"]
            ],

            annot=True,

            cmap="Reds",

            linewidths=0.5,

            ax=ax
        )

        ax.set_title(

            "Customer Recency Risk Heatmap",

            fontsize=15,

            fontweight="bold"
        )

        plt.tight_layout()

        return fig


    # =========================================
    # 4. CUSTOMER SEGMENT HEATMAP
    # =========================================
    def plot_segment_heatmap(
        self,
        cluster_profile: pd.DataFrame
    ):

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        heatmap_data = cluster_profile[
            [
                "Avg_Recency",
                "Avg_Frequency",
                "Avg_Monetary",
                "Avg_Account_Balance",
                "Avg_RFM_Score"
            ]
        ]

        sns.heatmap(

            heatmap_data,

            annot=True,

            fmt=".1f",

            cmap="YlGnBu",

            linewidths=0.5,

            ax=ax
        )

        ax.set_title(

            "Customer Segment Performance Heatmap",

            fontsize=15,

            fontweight="bold"
        )

        plt.tight_layout()

        return fig
    
# =========================================
# EXECUTION
# =========================================
if __name__ == "__main__":

    from src.modelling.segments import (
        main
    )

    logging.info(
        "Loading customer segment profile..."
    )

    rfm_df, cluster_profile = main()

    logging.info(
        "Initialising performance visualisation engine..."
    )

    visualizer = (
        CustomerSegmentPerformanceVisualisation()
    )

    logging.info(
        "Generating performance visualisations..."
    )

    fig1 = (
        visualizer
        .plot_revenue_distribution(
            cluster_profile
        )
    )

    fig2 = (
        visualizer
        .plot_recency_heatmap(
            cluster_profile
        )
    )

    fig3 = (
        visualizer
        .plot_segment_heatmap(
            cluster_profile
        )
    )

    radar_fig = (
        visualizer
        .plot_normalized_segment_radar_chart(
            cluster_profile
        )
    )

    logging.info(
        "Displaying performance plots..."
    )

    radar_fig.show()

    plt.show(block=True)