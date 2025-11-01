# Customer Segmentation using RFM & Clustering

## Introduction

This project applies RFM (Recency, Frequency, Monetary) analysis combined with clustering algorithms (K-Means, DBSCAN) to segment customers from the Online Retail dataset (2010–2011). Inspired by the idea of cluster-then-predict approaches in predictive modeling, this project focuses on identifying homogeneous customer groups and then building a classifier to predict the cluster of new customers automatically. The ultimate goal is to enhance marketing decision-making and enable real-time segmentation for business applications.

## Objectives:

- Clean and preprocess retail transaction data.
- Engineer meaningful RFM features.
- Train clustering models (K-Means, DBSCAN).
- Extract insights for marketing strategies.
- Train a classifier (e.g., XGBoost) for real-time segment prediction on new data.

## Pipeline

### 1. Preprocessing

- Handle missing values (Description, CustomerID).
- Remove invalid values (Quantity < 0, UnitPrice < 0).
- Exclude cancelled orders (InvoiceNo starting with "C"/"A").

### 2. Feature Engineering

RFM Variables:
- **Recency** → Days since last purchase.
- **Frequency** → Unique purchase days.
- **Monetary** → Total spending.

RFM Score: Quartile-based scoring (1–4).

Outlier Handling: IQR method applied to Frequency & Monetary.

This feature engineering step draws parallels to dimensionality reduction techniques, where features are combined into three dimensions. In RFM, we engineer features that capture customer behavior patterns—emphasizing Recency for timeliness, Frequency for loyalty, and Monetary for value—similar to how dimensionality reduction methods prioritize essential variables to improve clustering performance.

### 3. Normalization

- Standardized features using StandardScaler (mean ~0, std ~1), which is equivalent to Z-score normalization.
- MinMaxScaler avoided due to outlier sensitivity.

### 4. 🤖 Model Training

#### K-Means
- Optimal k ≈ 4 (via Elbow Method & Silhouette Score).
- Evaluated with: Silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn Index.

#### DBSCAN
- Density-based clustering, auto outlier detection.
- Parameters: eps, min_samples tuned via k-distance graph.
- High-quality clusters but many noise points.

This "cluster-then-predict" structure allows segmentation of customers into interpretable groups such as VIPs, Frequent Buyers, At Risk, and Low Value customers.
Once clusters are formed, these labels serve as the basis for supervised classification.

### 5. Others

- Outlier groups separated for analysis (VIPs, Frequent Buyers, Big Spenders).
- Visualization with PCA (2D) and 3D scatter plots.
- Combined clustering results with RFM Score for interpretable labels.

## 6. Classification

After clustering, we trained an XGBoost Classifier to predict the customer segment for new/unseen customers.

### Purpose of classification after clustering:

- Automate real-time customer segmentation without re-running clustering each time.
- Allow businesses to quickly assign new customers into the right group (e.g., VIP, At Risk, Frequent Buyer).
- Support marketing automation (personalized promotions, loyalty programs).
- Enable scalability: clustering is expensive, classification is fast.

### Results:

XGBoost Accuracy: ~98–100% (best among tested models).

Proves that learned clusters are highly distinguishable and can be predicted reliably.

Below is a table summarizing key evaluation metrics for the clustering and classification stages. Note: XGBoost, although more sophisticated than the transparent Logistic Regression, sometimes underperforms in maintaining interpretability as cluster complexity grows, leading to marginal AUC gains but higher computational overhead.

| Model                   | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-Score | Weighted Avg F1-Score |
|--------------------------|-----------|----------------------|------------------|--------------------|------------------------|
| KNeighborsClassifier     | 0.93      | 0.91                 | 0.86             | 0.87               | 0.93                   |
| LogisticRegression       | 0.90      | 0.91                 | 0.87             | 0.88               | 0.90                   |
| DecisionTreeClassifier   | 0.98      | 0.98                 | 0.98             | 0.98               | 0.98                   |
| RandomForestClassifier   | 0.98      | 0.99                 | 0.98             | 0.98               | 0.98                   |
| XGBClassifier            | 0.99      | 0.99                 | 1.00             | 0.99               | 0.99                   |

## 7. Streamlit API Deployment

We deployed the model using Streamlit so that users can:

  <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/b49ccd8e-1622-4f56-b2d4-791d80633da1" />

- Upload or Randomize input new customer transaction data.

  <img width="247" height="454" alt="image" src="https://github.com/user-attachments/assets/5a4c9a75-53d9-4fc5-b01e-10f39dee2111" />

- Automatically preprocess and calculate RFM features.

<img width="764" height="508" alt="image" src="https://github.com/user-attachments/assets/8ef35e01-e267-4f9c-980c-28cd2a4bba2e" />

- Get instant predicted customer segment (based on XGBoost).

  <img width="727" height="264" alt="image" src="https://github.com/user-attachments/assets/2bbe7d2f-03d7-4627-96d3-4f8660f1f274" />

## 8. Conclusion

This project demonstrates how RFM-based clustering combined with XGBoost classification can create an efficient, interpretable, and scalable customer segmentation system.
It supports real-time predictions, helping businesses better understand customer value and optimize marketing strategies without sacrificing performance or interpretability.


