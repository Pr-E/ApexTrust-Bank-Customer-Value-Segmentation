# 🏦 ApexTrust Bank Customer Intelligence & Segmentation Platform

---

# 📌 Executive Overview

ApexTrust Bank operates in an increasingly data-driven financial ecosystem where customer intelligence, personalization, retention, and behavioral analytics are critical competitive advantages.

This project delivers an end-to-end **Customer Intelligence & Segmentation Platform** that combines:

* RFM Behavioral Analytics
* K-Means Customer Segmentation
* Customer Segment Classification
* Executive Business Intelligence Dashboards
* AI-Generated Strategic Insights
* Real-Time API Deployment
* Interactive Streamlit Applications

The platform transforms raw transactional banking data into actionable strategic intelligence that supports:

* Customer retention
* Revenue optimization
* Cross-selling
* Churn reduction
* Executive decision-making
* Personalized banking strategies

---

# 🎯 Business Objectives

The primary objective of this project is to build a scalable customer intelligence system capable of:

✅ Identifying high-value banking customers

✅ Detecting churn-risk customers early

✅ Understanding customer behavioral patterns

✅ Improving customer retention strategies

✅ Supporting targeted marketing campaigns

✅ Enabling data-driven customer engagement

✅ Delivering stakeholder-ready executive insights

✅ Automating customer segment prediction and explainability

---

# 🧠 Core Intelligence Systems

This project consists of two major machine learning systems:

---

# 1️⃣ Customer Segmentation Engine (Unsupervised Learning)

A behavioral segmentation engine using:

* RFM Analysis
* K-Means Clustering

to automatically discover hidden customer behavioral groups.

## 📊 Generated Customer Segments

| Segment                      | Description                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| 🏆 Loyal High-Value Customer | Highly engaged premium customers with strong monetary value |
| ⚠️ At-Risk Premium Customer  | Previously valuable customers showing churn behavior        |
| 💳 Active Everyday Customer  | Highly active customers with moderate monetary contribution |
| 💤 Low Engagement Customer   | Low-frequency and low-value customers requiring engagement  |

---

# 2️⃣ Customer Segment Classification Engine (Supervised Learning)

A production-ready Random Forest Classification Model was developed to classify customers into their respective customer segments.

## 🔍 Classification Features

The model predicts customer segments using:

* Recency
* Frequency
* Monetary Value
* Customer Age
* Account Balance
* RFM Scores
* Behavioral Features

## ✅ Model Capabilities

* Real-time segment prediction
* Customer value classification
* Automated business recommendations
* SHAP explainability
* Human-readable business reports
* Feature contribution analysis

---

# 🧠 Explainable AI (XAI)

To ensure transparency and business interpretability, the project integrates:

## 🔹 SHAP Explainability

The system automatically explains:

* Why a customer belongs to a segment
* Key behavioral drivers
* Positive contribution factors
* Negative risk indicators
* Recommended business actions

### Example Business Explanations

The platform generates insights such as:

* High transaction frequency increased customer value
* Long inactivity period increased churn risk
* High account balance positively influenced premium classification

---

# 📊 Executive Intelligence Dashboard

A fully interactive executive dashboard was developed using:

* Streamlit
* Matplotlib
* Seaborn
* Plotly

The dashboard delivers stakeholder-ready business intelligence and customer analytics.

---

# 📈 Dashboard Capabilities

## 🔹 Customer Segmentation Overview

* Customer segment distribution
* Segment size analysis
* Segment heatmaps
* Customer population analytics

## 🔹 Customer Behaviour Intelligence

* RFM score analysis
* Engagement matrix
* Recency vs Monetary analysis
* Customer behavior radar charts
* RFM behavioral comparisons

## 🔹 Revenue Intelligence

* Revenue distribution donut charts
* Revenue concentration analysis
* Revenue vs customer percentage analysis
* Profitability intelligence

## 🔹 Retention Intelligence

* Customer retention risk matrix
* Recency heatmaps
* Dormant customer detection
* At-risk premium customer monitoring

## 🔹 AI-Generated Executive Insights

The dashboard automatically generates strategic intelligence such as:

* Highest revenue-generating segment
* Largest customer population
* Highest churn-risk segment
* Most engaged customer segment
* Strategic growth opportunities

---

# ⚙️ Machine Learning Workflow

## 1️⃣ Data Acquisition

* Connected to MongoDB / SQLite databases
* Extracted transactional banking data
* Loaded into Pandas DataFrames

---

## 2️⃣ Data Cleaning & Preprocessing

* Missing value handling
* Duplicate removal
* Date/time standardization
* Feature scaling
* Outlier handling

---

## 3️⃣ Exploratory Data Analysis (EDA)

Performed comprehensive customer analytics including:

* Transaction trends
* Spending behavior
* Customer demographics
* Monetary distributions
* Frequency analysis
* Behavioral outlier detection

---

## 4️⃣ Feature Engineering

Developed advanced customer intelligence features including:

### RFM Features

* Recency
* Frequency
* Monetary

### Additional Features

* Customer age
* Account balance
* Customer lifetime value
* RFM score levels
* Behavioral indicators

---

## 5️⃣ Customer Segmentation

Applied:

* K-Means Clustering
* Elbow Method
* Silhouette Score Analysis

to determine optimal customer clusters.

---

## 6️⃣ Segment Profiling

Generated detailed customer segment profiles including:

* Average recency
* Frequency patterns
* Monetary behavior
* Customer counts
* Revenue contribution
* Account balance intelligence
* Customer value levels

---

## 7️⃣ Customer Classification

Built a supervised learning pipeline using:

* Random Forest Classifier
* Train/Test Split
* Hyperparameter Optimization
* Feature Importance Analysis
* SHAP Explainability

---

# 🚀 API Deployment

A production-ready FastAPI service was developed for:

* Customer segmentation APIs
* Dashboard APIs
* Real-time predictions
* Model retraining
* Data refresh pipelines

## Available API Endpoints

| Endpoint     | Description               |
| ------------ | ------------------------- |
| `/dashboard` | Executive dashboard       |
| `/segments`  | Segmentation output       |
| `/refresh`   | Refresh pipeline cache    |
| `/retrain`   | Retrain clustering model  |
| `/docs`      | Swagger API documentation |

---

# 💻 Streamlit Applications

Two interactive applications were developed:

## 1️⃣ Customer Segment Classifier App

Allows users to:

* Input customer attributes
* Predict customer segment
* View SHAP explanations
* Receive business recommendations

---

## 2️⃣ Executive Intelligence Dashboard

Provides:

* Real-time visual analytics
* Executive KPIs
* Revenue intelligence
* Retention intelligence
* Customer behavior monitoring

---

# 📊 Key Business Insights

The project uncovered several strategic business insights:

✅ A small percentage of customers contributes the majority of revenue

✅ High-value customers exhibit significantly higher transaction frequency and monetary behavior

✅ At-risk premium customers represent major revenue leakage opportunities

✅ Behavioral inactivity strongly correlates with customer churn risk

✅ Active everyday customers provide strong engagement opportunities for cross-selling and digital banking adoption

---

# 🛠️ Technology Stack

## Programming

* Python

## Machine Learning

* Scikit-learn
* SHAP

## Data Processing

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn
* Plotly

## Backend & APIs

* FastAPI

## Dashboarding

* Streamlit

## Database

* MongoDB
* SQLite3

## Deployment

* Docker
* AWS EC2
* Render

## Version Control

* Git & GitHub

---

# 📁 Project Structure

```text
📦 apextrust-customer-segmentation
┣ 📂 api
┃ ┣ 📜 visualisation_main.py
┃ ┗ 📜 prediction_api.py
┃
┣ 📂 dashboard
┃ ┣ 📜 streamlit_dashboard.py
┃ ┗ 📜 customer_classifier_app.py
┃
┣ 📂 models
┃ ┣ 📜 clustering_model.pkl
┃ ┣ 📜 random_forest_classifier.pkl
┃ ┗ 📜 scaler.pkl
┃
┣ 📂 notebooks
┃ ┣ 📜 eda.ipynb
┃ ┗ 📜 experimentation.ipynb
┃
┣ 📂 src
┃ ┣ 📂 modelling
┃ ┃ ┣ 📜 clusters.py
┃ ┃ ┣ 📜 segments.py
┃ ┃ ┗ 📜 classifier.py
┃ ┃
┃ ┣ 📂 visualization
┃ ┃ ┣ 📜 customer_segment.py
┃ ┃ ┗ 📜 customer_segment_performance.py
┃ ┃
┃ ┣ 📂 explainability
┃ ┃ ┗ 📜 customer_segment_explainer.py
┃ ┃
┃ ┗ 📂 preprocessing
┃
┣ 📂 data
┣ 📂 assets
┣ 📜 requirements.txt
┣ 📜 Dockerfile
┣ 📜 README.md
┗ 📜 .gitignore
```

---

# 📈 Strategic Business Value

This platform provides ApexTrust Bank with:

* Enterprise-grade customer intelligence
* Explainable machine learning analytics
* Automated customer segmentation
* Revenue optimization insights
* Retention risk monitoring
* Executive decision intelligence
* Scalable AI-powered customer analytics infrastructure

---

# 🔮 Future Improvements

Potential future enhancements include:

* Real-time streaming analytics
* Customer lifetime value prediction
* Deep learning customer embeddings
* Personalised financial product recommendation
* Churn prediction modeling
* Cloud-native MLOps deployment
* CI/CD automation
* Automated retraining pipelines


---

# 👩‍💻 Author

Developed as an end-to-end customer intelligence and machine learning engineering project focused on:

* Banking analytics
* Customer intelligence
* Explainable AI
* Behavioral segmentation
* Production-grade ML systems
* Executive stakeholder reporting

---

# ⭐ Final Outcome

This project successfully demonstrates how machine learning, behavioral analytics, explainable AI, and executive dashboards can be integrated into a unified customer intelligence platform capable of driving measurable business impact in modern banking environments.
