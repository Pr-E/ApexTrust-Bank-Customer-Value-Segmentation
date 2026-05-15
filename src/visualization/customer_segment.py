import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomerSegmentVisualisation:

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
    # 1. SEGMENT DISTRIBUTION PIE CHART
    # =========================================
    def plot_segment_distribution(
        self,
        rfm_df: pd.DataFrame
    ):

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        segment_counts = (
            rfm_df["Cluster_Name"]
            .value_counts()
        )

        colors = [

            self.palette.get(
                segment,
                "#333333"
            )

            for segment in segment_counts.index
        ]

        explode = [
            0.05
        ] * len(segment_counts)

        ax.pie(

            segment_counts.values,

            labels=segment_counts.index,

            autopct=lambda p:
            f"{p:.1f}%\n"
            f"({int(p * sum(segment_counts.values)/100):,})",

            startangle=90,

            explode=explode,

            colors=colors,

            shadow=True
        )

        ax.set_title(

            "Customer Segment Distribution",

            fontsize=16,

            fontweight="bold"
        )

        ax.axis("equal")

        plt.tight_layout()

        return fig

    # =========================================
    # 2. SEGMENT SIZE BAR CHART
    # =========================================
    def plot_segment_size(
        self,
        rfm_df: pd.DataFrame
    ):

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        counts = (

            rfm_df["Cluster_Name"]

            .value_counts()

            .sort_values()
        )

        colors = [

            self.palette.get(
                segment,
                "#333333"
            )

            for segment in counts.index
        ]

        ax.barh(

            counts.index,

            counts.values,

            color=colors
        )

        ax.set_title(
            "Customer Segment Sizes",
            fontsize=15,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Customer Count"
        )

        ax.set_ylabel(
            "Customer Segment"
        )

        ax.grid(alpha=0.3)

        plt.tight_layout()

        return fig

    # =========================================
    # 3. RFM SCORE DISTRIBUTION BY SEGMENT
    # =========================================
    def plot_rfm_score_distribution(
        self,
        rfm_df: pd.DataFrame
    ):

        logging.info(
            "Generating RFM score distribution..."
        )

        fig, ax = plt.subplots(
            figsize=(14, 7)
        )

        sns.boxplot(

            data=rfm_df,

            x="Cluster_Name",

            y="RFM_Score",

            palette=self.palette,

            ax=ax
        )

        ax.set_title(

            "RFM Score Distribution Across Segments",

            fontsize=16,

            fontweight="bold"
        )

        ax.set_xlabel(
            "Customer Segment"
        )

        ax.set_ylabel(
            "RFM Score"
        )

        ax.tick_params(
            axis="x",
            rotation=15
        )

        ax.grid(alpha=0.3)

        plt.tight_layout()

        return fig

    # =========================================
    # 4. RFM SCORE VS MONETARY VALUE
    # =========================================
    def plot_rfm_score_vs_monetary(
        self,
        rfm_df: pd.DataFrame
    ):

        logging.info(
            "Generating RFM score vs monetary plot..."
        )

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        sns.scatterplot(

            data=rfm_df,

            x="RFM_Score",

            y="Monetary",

            hue="Cluster_Name",

            palette=self.palette,

            alpha=0.5,

            ax=ax
        )

        ax.set_yscale(
            "log"
        )

        ax.set_title(

            "RFM Score vs Monetary Value",

            fontsize=16,

            fontweight="bold"
        )

        ax.set_xlabel(
            "RFM Score"
        )

        ax.set_ylabel(
            "Monetary Value (Log Scale)"
        )

        ax.grid(alpha=0.3)

        plt.tight_layout()

        return fig

    # =========================================
    # 5. RECENCY VS MONETARY SCATTERPLOT
    # =========================================
    def plot_recency_vs_monetary(
        self,
        rfm_df: pd.DataFrame
    ):

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        sns.scatterplot(

            data=rfm_df,

            x="Recency",

            y="Monetary",

            hue="Cluster_Name",

            palette=self.palette,

            alpha=0.4,

            ax=ax
        )

        ax.set_yscale("log")

        ax.set_title(
            "Recency vs Monetary Distribution",
            fontsize=15,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Recency"
        )

        ax.set_ylabel(
            "Monetary"
        )

        ax.grid(alpha=0.3)

        plt.tight_layout()

        return fig

    # =========================================
    # 6. CUSTOMER ENGAGEMENT MATRIX
    # =========================================
    def plot_engagement_matrix(
        self,
        rfm_df: pd.DataFrame
    ):

        fig, ax = plt.subplots(
            figsize=(13, 8)
        )

        sns.scatterplot(

            data=rfm_df,

            x="Recency",

            y="Frequency",

            size="Monetary",

            hue="Cluster_Name",

            palette=self.palette,

            alpha=0.5,

            ax=ax
        )

        ax.set_title(

            "Customer Engagement Matrix",

            fontsize=16,

            fontweight="bold"
        )

        ax.set_xlabel(
            "Recency"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.grid(alpha=0.3)

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
        "Loading segmented customer data..."
    )

    rfm_df, cluster_profile = main()

    logging.info(
        "Initialising customer visualisation engine..."
    )

    visualizer = (
        CustomerSegmentVisualisation()
    )

    logging.info(
        "Generating customer segmentation plots..."
    )

    fig1 = (
        visualizer
        .plot_segment_distribution(
            rfm_df
        )
    )

    fig2 = (
        visualizer
        .plot_segment_size(
            rfm_df
        )
    )

    fig3 = (
        visualizer
        .plot_rfm_score_distribution(
            rfm_df
        )
    )

    fig4 = (
        visualizer
        .plot_rfm_score_vs_monetary(
            rfm_df
        )
    )

    fig5 = (
        visualizer
        .plot_recency_vs_monetary(
            rfm_df
        )
    )

    fig6 = (
        visualizer
        .plot_engagement_matrix(
            rfm_df
        )
    )

    logging.info(
        "Displaying customer segmentation plots..."
    )

    plt.show(block=True)