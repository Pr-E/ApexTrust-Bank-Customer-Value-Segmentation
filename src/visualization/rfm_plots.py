# =========================================
# APEX TRUST RFM SCORE ANALYSIS
# DESCRIPTIVE STATISTICS + VISUALISATIONS
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from src.modelling.segments import main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================
# LOAD DATA
# =========================================
logging.info(
    "Loading segmented customer data..."
)

rfm_df, cluster_profile = main()

logging.info(
    "Data loaded successfully"
)


# =========================================
# CREATE RFM LEVELS
# =========================================
def assign_rfm_level(score):

    if score >= 13:

        return "Elite Value"

    elif score >= 10:

        return "High Value"

    elif score >= 7:

        return "Moderate Value"

    else:

        return "Low Value"


rfm_df['RFM_Level'] = rfm_df[
    'RFM_Score'
].apply(assign_rfm_level)


# =========================================
# DESCRIPTIVE STATISTICS
# =========================================
logging.info(
    "Generating descriptive statistics..."
)

rfm_summary = rfm_df[[
    'Recency',
    'Frequency',
    'Monetary',
    'R_Score',
    'F_Score',
    'M_Score',
    'RFM_Score'
]].describe().round(2)

print("\n")
print("=" * 60)
print("RFM DESCRIPTIVE STATISTICS")
print("=" * 60)
print(rfm_summary)


# =========================================
# RFM LEVEL DISTRIBUTION
# =========================================
rfm_level_distribution = (
    rfm_df['RFM_Level']
    .value_counts()
    .reset_index()
)

rfm_level_distribution.columns = [
    'RFM_Level',
    'Customer_Count'
]

rfm_level_distribution[
    'Percentage'
] = (

    rfm_level_distribution[
        'Customer_Count'
    ]

    /

    rfm_level_distribution[
        'Customer_Count'
    ].sum()

) * 100

rfm_level_distribution[
    'Percentage'
] = rfm_level_distribution[
    'Percentage'
].round(2)

print("\n")
print("=" * 60)
print("RFM LEVEL DISTRIBUTION")
print("=" * 60)
print(rfm_level_distribution)


# =========================================
# COLOR PALETTE
# =========================================
palette = {

    "Elite Value": "#1f77b4",

    "High Value": "#2ca02c",

    "Moderate Value": "#ff7f0e",

    "Low Value": "#d62728"
}


# =========================================
# 1. RFM SCORE DISTRIBUTION
# =========================================
logging.info(
    "Generating RFM score distribution..."
)

plt.figure(figsize=(12, 6))

sns.histplot(
    rfm_df['RFM_Score'],
    bins=15,
    kde=True,
    color="#1f77b4"
)

plt.title(
    "Distribution of RFM Scores",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("RFM Score")
plt.ylabel("Customer Count")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================
# 2. RFM LEVEL PIE CHART
# =========================================
logging.info(
    "Generating RFM level pie chart..."
)

plt.figure(figsize=(10, 8))

colors = [

    palette[level]

    for level in rfm_level_distribution[
        'RFM_Level'
    ]
]

explode = [0.05] * len(colors)

plt.pie(

    rfm_level_distribution[
        'Customer_Count'
    ],

    labels=rfm_level_distribution[
        'RFM_Level'
    ],

    autopct=lambda p:
        f'{p:.1f}%',

    colors=colors,

    explode=explode,

    shadow=True,

    startangle=90
)

plt.title(
    "Customer Distribution by RFM Level",
    fontsize=16,
    fontweight='bold'
)

plt.axis('equal')

plt.tight_layout()
plt.show()


# =========================================
# 3. RFM LEVEL BAR CHART
# =========================================
logging.info(
    "Generating RFM level bar chart..."
)

plt.figure(figsize=(12, 6))

sns.barplot(

    data=rfm_level_distribution,

    x='RFM_Level',

    y='Customer_Count',

    hue='RFM_Level',

    palette=palette,

    legend=False
)

plt.title(
    "Customer Count by RFM Level",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("RFM Level")
plt.ylabel("Customer Count")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================
# 4. RFM COMPONENT BOXPLOTS
# =========================================
logging.info(
    "Generating RFM component boxplots..."
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

sns.boxplot(
    y=rfm_df['Recency'],
    ax=axes[0],
    color="#1f77b4"
)

axes[0].set_title(
    "Recency Distribution"
)

sns.boxplot(
    y=rfm_df['Frequency'],
    ax=axes[1],
    color="#2ca02c"
)

axes[1].set_title(
    "Frequency Distribution"
)

sns.boxplot(
    y=rfm_df['Monetary'],
    ax=axes[2],
    color="#ff7f0e"
)

axes[2].set_title(
    "Monetary Distribution"
)

axes[2].set_yscale('log')

plt.tight_layout()
plt.show()


# =========================================
# 5. RFM HEATMAP
# =========================================
logging.info(
    "Generating RFM correlation heatmap..."
)

plt.figure(figsize=(10, 7))

corr_matrix = rfm_df[[
    'Recency',
    'Frequency',
    'Monetary',
    'R_Score',
    'F_Score',
    'M_Score',
    'RFM_Score'
]].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title(
    "RFM Feature Correlation Heatmap",
    fontsize=16,
    fontweight='bold'
)

plt.tight_layout()
plt.show()


# =========================================
# 6. RFM SCORE VS MONETARY
# =========================================
logging.info(
    "Generating RFM score vs monetary plot..."
)

plt.figure(figsize=(12, 7))

sns.scatterplot(

    data=rfm_df,

    x='RFM_Score',

    y='Monetary',

    hue='RFM_Level',

    palette=palette,

    alpha=0.5
)

plt.yscale('log')

plt.title(
    "RFM Score vs Monetary Value",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("RFM Score")
plt.ylabel("Monetary Value (Log Scale)")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================
# 7. AVERAGE METRICS BY RFM LEVEL
# =========================================
logging.info(
    "Generating average metrics by RFM level..."
)

rfm_grouped = rfm_df.groupby(
    'RFM_Level'
)[
    [
        'Recency',
        'Frequency',
        'Monetary',
        'CustAccountBalance'
    ]
].mean().round(2)

print("\n")
print("=" * 60)
print("AVERAGE METRICS BY RFM LEVEL")
print("=" * 60)
print(rfm_grouped)


# =========================================
# 8. RFM LEVEL REVENUE CONTRIBUTION
# =========================================
logging.info(
    "Generating revenue contribution chart..."
)

revenue_by_level = rfm_df.groupby(
    'RFM_Level'
)['Monetary'].sum().reset_index()

plt.figure(figsize=(12, 6))

sns.barplot(

    data=revenue_by_level,

    x='RFM_Level',

    y='Monetary',

    hue='RFM_Level',

    palette=palette,

    legend=False
)

plt.title(
    "Revenue Contribution by RFM Level",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("RFM Level")
plt.ylabel("Total Monetary Value")

plt.tight_layout()
plt.show()


logging.info(
    "RFM analysis visualisations completed successfully"
)