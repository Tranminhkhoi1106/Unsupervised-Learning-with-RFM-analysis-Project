import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Lấy modl XGBoost
with open("xgb_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title(" Customer Segmentation Prediction App")
st.sidebar.header("Input")

# Tạo slider
num_samples = st.sidebar.slider("Number of random samples", min_value=1, max_value=50, value=5)

# Giá trị đầu
recency_default = st.sidebar.number_input("Default Recency (days)", min_value=0, value=50)
frequency_default = st.sidebar.number_input("Default Frequency", min_value=0, value=5)
monetary_default = st.sidebar.number_input("Default Monetary ($)", min_value=0, value=200)

# Hiện RFM của khách hàng mới
st.write("New customer")

# Để default
if "rfm_df" not in st.session_state:
    st.session_state.rfm_df = pd.DataFrame({
        "Recency": [recency_default],
        "Frequency": [frequency_default],
        "Monetary": [monetary_default]
    })

# Nút randomize nhiều dòng
if st.sidebar.button("Randomize Multiple Rows"):
    st.session_state.rfm_df = pd.DataFrame({
        "Recency": np.random.randint(1, 200, num_samples),
        "Frequency": np.random.randint(1, 20, num_samples),
        "Monetary": np.random.randint(10, 10000, num_samples)
    })

# Hiện randomize lên bảng khách hàng mới
rfm_df = st.session_state.rfm_df
st.dataframe(rfm_df)

# Chia IQR của RFM
r_quarters = [0, 25, 50, 100, 200]
f_quarters = [0, 5, 10, 15, 20]
m_quarters = [0, 250, 500, 750, 10000]

rfm_df['r_score'] = pd.cut(rfm_df['Recency'], bins=r_quarters, labels=[4,3,2,1], include_lowest=True, duplicates='drop')
rfm_df['f_score'] = pd.cut(rfm_df['Frequency'], bins=f_quarters, labels=[1,2,3,4], include_lowest=True, duplicates='drop')
rfm_df['m_score'] = pd.cut(rfm_df['Monetary'], bins=m_quarters, labels=[1,2,3,4], include_lowest=True, duplicates='drop')

rfm_df['RFM_Score'] = (
    rfm_df['r_score'].astype(str)
    + rfm_df['f_score'].astype(str)
    + rfm_df['m_score'].astype(str)
)


st.write("### RFM Score Calculation")
st.dataframe(rfm_df[['Recency', 'Frequency', 'Monetary', 'r_score', 'f_score', 'm_score', 'RFM_Score']])


# Predict
if st.button("Predict with XGBoost"):
    # lấy rfm_dc tránh bị shapemismatch vì thiếu cột
    input_data = rfm_df[['Recency', 'Frequency', 'Monetary', 'r_score', 'f_score', 'm_score', 'RFM_Score']].values

    # Standard scale from saved scaler (prevent leak)
    with open("xgb_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    input_data = scaler.transform(input_data)

    # Dự đoán với XGBoost
    model_type = str(type(model))
    if "xgboost.core.Booster" in model_type:
        dmatrix = xgb.DMatrix(input_data)
        prediction = model.predict(dmatrix)
    else:
        prediction = model.predict(input_data)

    # Label collumn
    rfm_df['Prediction'] = prediction
    
    # Đổi tên label
label_mapping = { 0: 'Valuable Customers (Outliers)',
                1: 'Frequent Buyers (Outliers)',
                2: 'High-Spending Buyers (Outliers)',
                3: 'New/Single Purchase Customers',
                4: 'Engaged Customers',
                5: 'At Risk Customers',
                6: 'Moderate Engagement Customers',
                7: 'Low-Value Customers'
}
rfm_df['Customer_Segment'] = rfm_df['Prediction'].map(label_mapping)
    
# Bỏ hiển thị label prediction    
rfm_df.drop(columns=['r_score', 'f_score', 'm_score','Prediction'], inplace=True)

# Hiện nếu thành công
st.success("Prediction completed!")
st.write("### Prediction Results")

# Hiện df với cột dự đoán
st.dataframe(rfm_df)

