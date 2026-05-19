# 🏦 ApexTrust Bank Customer Intelligence & Segmentation Platform

---

# 📌 Executive Overview

ApexTrust Bank operates within an increasingly data-driven financial ecosystem where customer intelligence, personalization, retention, and behavioural analytics are critical competitive advantages.

This project delivers an end-to-end **Customer Intelligence & Segmentation Platform** designed to transform raw transactional banking data into actionable behavioural, operational, and strategic business intelligence.

The platform combines:

- RFM Behavioural Analytics
- Advanced Customer Feature Engineering
- K-Means Customer Segmentation
- Random Forest Customer Segment Classification
- SHAP Explainability
- FastAPI Backend Deployment
- Executive Streamlit Dashboards
- AI-Generated Strategic Insights
- Docker Containerisation
- AWS EC2 Cloud Deployment
- CI/CD Automation with GitHub Actions

The solution supports:

- Customer retention
- Revenue optimisation
- Behavioural intelligence
- Churn risk monitoring
- Executive decision-making
- Cross-selling opportunities
- Personalised banking strategies
- Real-time customer analytics

---

# 🎯 Business Objectives

The primary objective of this project is to build a scalable customer intelligence platform capable of:

✅ Identifying high-value banking customers  
✅ Detecting churn-risk customers early  
✅ Understanding behavioural engagement patterns  
✅ Improving customer retention strategies  
✅ Supporting targeted marketing campaigns  
✅ Enabling data-driven customer engagement  
✅ Delivering stakeholder-ready executive intelligence  
✅ Automating customer segment prediction and explainability  
✅ Supporting scalable AI-powered banking analytics  

---

# 🧠 Core Intelligence Systems

The platform consists of two major machine learning systems:

---

# 1️⃣ Customer Segmentation Engine (Unsupervised Learning)

A behavioural customer segmentation engine was developed using:

- RFM Analysis
- K-Means Clustering

to automatically discover hidden customer behavioural groups based on transactional activity and engagement behaviour.

---

## 📊 Generated Customer Segments

| Segment | Description |
|---|---|
| 🏆 Loyal High-Value Customer | Highly engaged premium customers with strong monetary contribution |
| ⚠️ At-Risk Premium Customer | Previously valuable customers displaying churn behaviour |
| 💳 Active Everyday Customer | Highly active customers with moderate monetary contribution |
| 💤 Low Engagement Customer | Low-frequency, low-engagement customers requiring activation |

---

## 📈 Segmentation Intelligence

The segmentation engine enables:

- Behavioural customer profiling
- Revenue concentration analysis
- Customer engagement monitoring
- Retention risk detection
- Segment-level strategic recommendations

---

# 2️⃣ Customer Segment Classification Engine (Supervised Learning)

A production-ready Random Forest Classification Model was developed to automatically classify customers into behavioural customer segments.

---

## 🔍 Classification Features

The classifier predicts customer segments using:

- Recency
- Frequency
- Monetary Value
- Customer Age
- Account Balance
- RFM Scores
- Behavioural Features
- Customer Intelligence Metrics

---

## ✅ Classification Capabilities

The classification system supports:

- Real-time customer segment prediction
- Automated business recommendations
- Feature contribution analysis
- SHAP explainability
- Human-readable customer intelligence reports
- Customer value classification
- Behavioural risk interpretation

---

# 🧠 Explainable AI (XAI)

To improve transparency and business interpretability, the project integrates SHAP Explainability.

---

## 🔹 SHAP Explainability Engine

The explainability system automatically identifies:

- Why a customer belongs to a segment
- Key behavioural drivers
- Positive value indicators
- Negative churn indicators
- Recommended business actions
- Segment-level behavioural influences

---

## 📋 Example AI Explanations

The system generates business-friendly explanations such as:

- High transaction frequency increased customer value
- Increasing inactivity elevated churn risk
- Strong account balance positively influenced premium classification
- Low engagement reduced customer value scoring

---

# 📊 Executive Intelligence Dashboard

A fully interactive executive dashboard was developed using:

- Streamlit
- Matplotlib
- Seaborn
- Plotly

The dashboard provides stakeholder-ready customer intelligence and business analytics.

---

# 📈 Dashboard Capabilities

## 🔹 Customer Segmentation Overview

- Customer segment distribution
- Segment size analysis
- Customer population analytics
- Behavioural segmentation summaries

---

## 🔹 Customer Behaviour Intelligence

- RFM score analysis
- Engagement matrix analytics
- Recency vs Monetary analysis
- Behavioural distribution analysis
- Customer activity intelligence

---

## 🔹 Revenue Intelligence

- Revenue distribution donut charts
- Revenue concentration analysis
- Segment-level profitability analysis
- Revenue contribution intelligence

---

## 🔹 Retention Intelligence

- Customer churn risk monitoring
- Recency heatmaps
- Dormant customer detection
- At-risk premium customer analysis

---

## 🔹 AI-Generated Strategic Insights

The dashboard automatically generates executive intelligence including:

- Highest revenue-generating segment
- Largest customer population
- Highest retention-risk segment
- Most active customer segment
- Strategic growth opportunities
- Behavioural engagement insights

---

# ⚙️ Machine Learning Workflow

---

## 1️⃣ Data Acquisition

- Connected to MongoDB / SQLite databases
- Extracted transactional banking data
- Loaded data into Pandas DataFrames

---

## 2️⃣ Data Cleaning & Preprocessing

- Missing value handling
- Duplicate removal
- Date/time standardisation
- Feature scaling
- Outlier handling
- Data quality validation

---

## 3️⃣ Exploratory Data Analysis (EDA)

Performed comprehensive analytics including:

- Transaction trends
- Customer spending behaviour
- Monetary distributions
- Behavioural outlier analysis
- Engagement analysis
- Customer demographic exploration

---

## 4️⃣ Feature Engineering

Developed advanced behavioural intelligence features including:

### RFM Features

- Recency
- Frequency
- Monetary

### Additional Features

- Customer age
- Account balance
- Customer lifetime value indicators
- RFM score levels
- Behavioural indicators

---

## 5️⃣ Customer Segmentation

Applied:

- K-Means Clustering
- Elbow Method
- Silhouette Score Analysis

to determine optimal customer clusters.

---

## 6️⃣ Segment Profiling

Generated detailed customer intelligence profiles including:

- Average recency
- Transaction frequency
- Monetary behaviour
- Revenue contribution
- Customer counts
- Behavioural engagement
- Account balance intelligence

---

## 7️⃣ Customer Classification

Built a supervised learning pipeline using:

- Random Forest Classifier
- Train/Test Split
- Hyperparameter Optimisation
- Feature Importance Analysis
- SHAP Explainability

---

# 🚀 FastAPI Backend Deployment

A production-ready FastAPI backend service was developed for:

- Customer segmentation APIs
- Dashboard APIs
- Real-time customer predictions
- Model retraining
- Data refresh pipelines
- Executive dashboard delivery

---

## 📡 API Endpoints

| Endpoint | Description |
|---|---|
| `/` | API health route |
| `/dashboard` | Executive intelligence dashboard |
| `/segments` | Segmentation output |
| `/refresh` | Refresh segmentation pipeline |
| `/retrain` | Retrain clustering model |
| `/docs` | Swagger API documentation |

---

# 💻 Streamlit Applications

Two interactive Streamlit applications were developed.

---

## 1️⃣ Customer Segment Classifier App

The classifier application enables users to:

- Input customer attributes
- Predict behavioural customer segments
- View SHAP explainability
- Receive automated business recommendations
- Generate customer intelligence reports

---

## 2️⃣ Executive Intelligence Dashboard

The executive dashboard provides:

- Real-time behavioural analytics
- Executive KPI monitoring
- Revenue intelligence
- Retention intelligence
- Behavioural engagement analysis
- Strategic customer insights

---

# ☁️ Deployment & MLOps

The platform was containerised and deployed using modern MLOps workflows.

---

## 🔹 Deployment Stack

- Docker
- AWS EC2
- GitHub Actions CI/CD
- FastAPI
- Streamlit

---

## 🔹 CI/CD Pipeline

Implemented automated deployment workflows including:

- Docker image builds
- DockerHub image publishing
- EC2 deployment automation
- Container orchestration
- Automated backend refresh workflows

---

# 📊 Key Business Insights

Several strategic customer intelligence insights emerged throughout the project:

✅ A relatively small percentage of customers contributes the majority of estimated revenue  

✅ High-value customers consistently demonstrate stronger behavioural engagement and transaction frequency  

✅ At-risk premium customers represent significant revenue leakage opportunities  

✅ Increasing recency strongly correlates with churn risk behaviour  

✅ Behavioural engagement is often a stronger customer value indicator than account balance alone  

✅ Active everyday customers provide strong cross-selling and digital banking opportunities  

---

# 🛠️ Technology Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- SHAP

## Data Processing

- Pandas
- NumPy

## Data Visualisation

- Matplotlib
- Seaborn
- Plotly

## Backend & APIs

- FastAPI

## Dashboarding

- Streamlit

## Databases

- MongoDB
- SQLite3

## Deployment & MLOps

- Docker
- AWS EC2
- GitHub Actions
- DockerHub

## Version Control

- Git & GitHub

---

# 📁 Project Structure

```text
📦 ApexTrust-Bank-Customer-Value-Segmentation
┣ 📂 api
┃ ┣ 📜 visualisation_main.py
┃ ┗ 📜 main.py
┃
┣ 📂 app
┃ ┣ 📜 streamlit_dashboard.py
┃ ┗ 📜 streamlit_app.py
┃
┣ 📂 src
┃ ┣ 📂 connections
┃ ┣ 📂 data
┃ ┣ 📂 explainability
┃ ┣ 📂 features
┃ ┣ 📂 modelling
┃ ┣ 📂 utils
┃ ┗ 📂 visualization
┃
┣ 📂 dataset
┃
┣ 📂 .github\workflows
┃ ┗ 📜 deploy.yml
┃
┣ 📜 Dockerfile
┣ 📜 requirements.txt
┣ 📜 README.md
┣ 📜 setup.py
┗ 📜 .gitignore
```

---

# 📈 Strategic Business Value

This platform provides ApexTrust Bank with:

- Enterprise-grade customer intelligence
- Explainable machine learning analytics
- Automated behavioural segmentation
- Revenue optimisation intelligence
- Retention risk monitoring
- Executive stakeholder reporting
- AI-powered customer analytics infrastructure
- Scalable cloud-ready ML deployment

---

# 🔮 Future Improvements

Potential future enhancements include:

- Real-time streaming customer analytics
- Customer lifetime value prediction
- Churn prediction models
- Deep learning behavioural embeddings
- Automated retraining pipelines

---

# 👩‍💻 Author

Developed as an end-to-end customer intelligence and machine learning engineering project focused on:

- Banking analytics
- Behavioural intelligence
- Explainable AI
- Production-grade ML systems
- Executive stakeholder reporting
- Customer value optimisation
- Scalable AI deployment systems

---

# ⭐ Final Outcome

This project demonstrates how machine learning, behavioural analytics, explainable AI, cloud deployment, and executive business intelligence can be integrated into a unified customer intelligence platform capable of driving measurable business impact within modern banking environments.