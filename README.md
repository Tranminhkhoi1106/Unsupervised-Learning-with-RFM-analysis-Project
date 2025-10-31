# Customer Segmentation using RFM & Clustering

## Introduction

This project applies RFM (Recency, Frequency, Monetary) analysis combined with clustering algorithms (K-Means, DBSCAN) to segment customers from the Online Retail dataset (2010–2011). Inspired by recent research in predictive modeling, such as the work by Teng et al. (2023) on bridging accuracy and interpretability in credit scoring via a "Rescaled Cluster-then-Predict" approach, this project adapts similar techniques to customer segmentation. By clustering customers into homogeneous groups and then applying predictive models, we aim to derive actionable insights while balancing performance and explainability.

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

This feature engineering step draws parallels to the rescaling techniques in the paper, where features are adjusted based on their relevance to the target (e.g., default prediction in credit scoring). In RFM, we engineer features that capture customer behavior patterns, emphasizing recency for timeliness, frequency for loyalty, and monetary for value—similar to how the paper rescales features to reflect their impact on the outcome, promoting better clustering by prioritizing essential variables.

### 3. Normalization

- Standardized features using StandardScaler (mean ~0, std ~1), which is equivalent to Z-score normalization.
- MinMaxScaler avoided due to outlier sensitivity.

Feature scaling is crucial for clustering. Z-score normalization ensures features contribute equally without dominance by scale, preventing irrelevant features from skewing results—much like the paper's warning against traditional normalization (e.g., min-max) that doesn't differentiate feature importance. MinMax was tested but discarded here, as it amplified outliers in Monetary values, leading to suboptimal clusters. In contrast, Z-score provided robust handling of imbalanced distributions, aligning with the paper's emphasis on rescaling to mirror feature significance for improved prediction accuracy.

### 4. 🤖 Model Training

#### K-Means
- Optimal k ≈ 4 (via Elbow Method & Silhouette Score).
- Evaluated with: Silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn Index.

#### DBSCAN
- Density-based clustering, auto outlier detection.
- Parameters: eps, min_samples tuned via k-distance graph.
- High-quality clusters but many noise points.

This "cluster-then-predict" framework shares common ground with the paper, where data is segmented into subgroups via clustering, followed by model application (e.g., Logistic Regression or XGBoost) within each group. Both approaches aim to enhance interpretability and performance on imbalanced data—here, imbalanced customer behaviors (e.g., few high-value VIPs vs. many low-frequency buyers). A key similarity is clustering only "positive" cases for efficiency: the paper clusters default (positive) cases to reduce computation, while this project isolates outliers (e.g., VIPs, Big Spenders) as separate groups before full clustering, yielding comparable results with lower resource use.

In the "rescaled cluster-then-predict" technique from the paper, XGBoost plays a dual role: it's used for sophisticated prediction within clusters but often remains unaffected or even dips in AUC with more clusters/polynomial features. Here, XGBoost is adapted post-clustering for classification, leveraging its ensemble strength to predict segments accurately. However, as noted in the paper, XGBoost—although more sophisticated than the transparent Logistic Regression—can sometimes underperform in scenarios requiring high interpretability, such as when clusters increase complexity without proportional gains in AUC.

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
- Upload or Randomize input new customer transaction data.

  <img width="247" height="454" alt="image" src="https://github.com/user-attachments/assets/5a4c9a75-53d9-4fc5-b01e-10f39dee2111" />

- Automatically preprocess and calculate RFM features.

<img width="764" height="508" alt="image" src="https://github.com/user-attachments/assets/8ef35e01-e267-4f9c-980c-28cd2a4bba2e" />

- Get instant predicted customer segment (based on XGBoost).

  <img width="727" height="264" alt="image" src="https://github.com/user-attachments/assets/2bbe7d2f-03d7-4627-96d3-4f8660f1f274" />



