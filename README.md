# 📊 Customer Segmentation using RFM & Clustering
## 📌 Introduction

This project applies RFM (Recency, Frequency, Monetary) analysis combined with clustering algorithms (K-Means, DBSCAN) to segment customers from the Online Retail dataset (2010–2011).

## 🎯 Objectives:

Clean and preprocess retail transaction data.

Engineer meaningful RFM features.

Train clustering models (K-Means, DBSCAN).

Extract insights for marketing strategies.

# ⚙️ Pipeline
## 1. 🔍 Preprocessing

Handle missing values (Description, CustomerID).

Remove invalid values (Quantity < 0, UnitPrice < 0).

Exclude cancelled orders (InvoiceNo starting with "C"/"A").

## 2. 🛠 Feature Engineering

RFM Variables:

Recency → Days since last purchase.

Frequency → Unique purchase days.

Monetary → Total spending.

RFM Score: Quartile-based scoring (1–4).

Outlier Handling: IQR method applied to Frequency & Monetary.

## 3. 📏 Normalization

Standardized features using StandardScaler (mean ~0, std ~1).

MinMaxScaler avoided due to outlier sensitivity.

## 4. 🤖 Model Training
### K-Means

Optimal k ≈ 4 (via Elbow Method & Silhouette Score).

Evaluated with: Silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn Index.

### DBSCAN

Density-based clustering, auto outlier detection.

Parameters: eps, min_samples tuned via k-distance graph.

High-quality clusters but many noise points.

## 5. 📂 Others

Outlier groups separated for analysis (VIPs, Frequent Buyers, Big Spenders).

Visualization with PCA (2D) and 3D scatter plots.

Combined clustering results with RFM Score for interpretable labels.
## 6. Classification

After clustering, we trained an XGBoost Classifier to predict the customer segment for new/unseen customers.

### 🎯 Purpose of classification after clustering:

Automate real-time customer segmentation without re-running clustering each time.

Allow businesses to quickly assign new customers into the right group (e.g., VIP, At Risk, Frequent Buyer).

Support marketing automation (personalized promotions, loyalty programs).

Enable scalability: clustering is expensive, classification is fast.

### ✅ Results:

XGBoost Accuracy: ~98–100% (best among tested models).

Proves that learned clusters are highly distinguishable and can be predicted reliably.

## 7. Streamlit API Deployment

We deployed the model using Streamlit so that users can:

- Upload or input new customer transaction data.

- Automatically preprocess and calculate RFM features.

- Get instant predicted customer segment (based on XGBoost).

- iew visual dashboards (cluster distribution, segment insights, recommendation strategies).

- The app provides an interactive interface where marketing teams can experiment with:

- Checking segment labels for new customers.

- Visualizing customer clusters (2D & 3D plots).

- Viewing personalized strategy suggestions per segment.

## 6. 💡 Insights

### Identified customer segments:

🏆 Valuable Customers (VIPs): High Frequency & Monetary, recent buyers.

🔄 Frequent Buyers: Purchase often, moderate spending/order.

💰 High-Spending Buyers: Fewer orders, but very high value per order.

🤝 Engaged Customers: Consistent and active.

🌱 New/One-time Customers: First-time or low-value purchases.

⚠️ At-Risk Customers: Long inactivity, low spending.

### 📢 Marketing strategies:

VIPs → Loyalty programs, exclusive offers.

Frequent Buyers → Upselling, bundle promotions.

Big Spenders → Premium care, personalized deals.

New Customers → Welcome offers, guided shopping.

At-Risk → Win-back campaigns with strong incentives.

# ✅ Conclusion

K-Means: Provided balanced, clear customer segments.

DBSCAN: Strong outlier detection but produced many noise points.

RFM + Clustering: Helped generate actionable marketing insights and customer strategies.

🚀 Future Work:

Explore advanced clustering (Gaussian Mixture, Spectral Clustering).

Add behavioral/time-series features.

Build recommendation systems for each segment.
