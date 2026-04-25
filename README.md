# 🏦 ApexTrust Bank Customer Value Segmentation

## 📌 Project Overview
ApexTrust Bank operates in a highly competitive financial services environment where customer experience and personalization are key drivers of success. Despite having large volumes of transactional data, the bank lacks behavioral insights into customer activity.

This project develops a **Customer Value Segmentation System** using **RFM (Recency, Frequency, Monetary) analysis** and **machine learning (K-Means clustering)** to transform raw transactional data into actionable business insights.

---

## 🎯 Objectives

- Develop behavior-based customer segmentation
- Identify high-value and at-risk customers
- Enable personalized financial services
- Improve marketing targeting and campaign effectiveness
- Enhance customer retention strategies
- Support data-driven decision-making

---

## 🧠 Key Concepts

### 🔹 RFM Analysis
- **Recency (R):** How recently a customer transacted  
- **Frequency (F):** How often a customer transacts  
- **Monetary (M):** Total value of customer transactions  

### 🔹 Machine Learning
- **K-Means Clustering** for grouping customers based on behavioral similarity  
- **Silhouette Score & Elbow Method** for optimal cluster selection  

---

## 🗂️ Dataset Description

- **Source:** MongoDB / SQLite3  
- **Granularity:** Transaction-level data  

### Key Features:
- `TransactionID` – Unique transaction identifier  
- `CustomerID` – Unique customer identifier  
- `TransactionDate` – Date of transaction  
- `TransactionTime` – Time of transaction  
- `TransactionAmount` – Transaction value  
- `CustomerDOB` – Date of birth  
- `CustGender` – Gender  
- `CustLocation` – Customer location  
- `CustAccountBalance` – Account balance  

---

## ⚙️ Project Workflow

### 1️⃣ Data Acquisition
- Connected to MongoDB using `pymongo`
- Extracted data into Pandas DataFrame

### 2️⃣ Data Cleaning & Preprocessing
- Handled missing values and duplicates  
- Converted date/time formats  
- Standardized features  

### 3️⃣ Exploratory Data Analysis (EDA)
- Transaction trends over time  
- Customer behavior patterns  
- Outlier detection  
- Top customer analysis  

### 4️⃣ Feature Engineering
- Computed RFM metrics  
- Derived features:
  - Customer age  
  - Transaction frequency  
  - Average transaction value  
  - Account balance trends  

### 5️⃣ Customer Segmentation
- Applied **K-Means clustering**  
- Determined optimal clusters using:
  - Elbow Method  
  - Silhouette Score  

### 6️⃣ Segment Profiling
- Identified segments such as:
  - 🏆 Champions  
  - 🤝 Loyal Customers  
  - ⚠️ At-Risk Customers  
  - 💤 Inactive Customers  

### 7️⃣ Deployment
- **FastAPI** for serving model predictions  
- **Streamlit** for interactive dashboards  
- **Docker** for containerization  
- **AWS EC2 / Render** for cloud deployment  

---

## 📊 Key Insights

- A small percentage of customers contributes a large portion of revenue  
- Clear distinction between:
  - High spenders  
  - High-balance customers  
  - Frequent users  
- Customer behavior varies significantly across segments  
- Early signs of churn can be detected through declining activity  

---

## 🛠️ Tech Stack

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn  
- **Database:** MongoDB, SQLite3  
- **Visualization:** Matplotlib, Streamlit  
- **API:** FastAPI  
- **Deployment:** Docker, AWS EC2, Render  
- **Version Control:** Git & GitHub  

---

## 📁 Project Structure

📦 apextrust-customer-segmentation
┣ 📂 data
┣ 📂 notebooks
┣ 📂 src
┣ 📂 models
┣ 📂 dashboard
┣ 📂 api
┣ 📜 README.md
┣ 📜 requirements.txt
┗ 📜 Dockerfile

## 📈 Expected Outcomes

- Clear customer segments based on behavior
- Improved marketing efficiency and targeting
- Enhanced customer retention strategies
- Data-driven decision-making framework
- Scalable analytics pipeline

--- 

## ⚠️ Challenges & Considerations

- Data quality and consistency (e.g., multiple locations per customer)
- Handling large-scale transactional data
- Choosing optimal clustering parameters
- Ensuring model interpretability
