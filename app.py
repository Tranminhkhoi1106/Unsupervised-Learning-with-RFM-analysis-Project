import streamlit as st
import joblib
import numpy as np

# The image provided shows the R_score, F_score, and M_score. These scores
# are typically calculated by binning the Recency, Frequency, and Monetary
# values into quartiles. The code below assumes some hardcoded thresholds
# for demonstration. You should replace these with the actual quartile
# values from your training data.
RECENCY_QUARTILES = [1, 18, 51, 142]
FREQUENCY_QUARTILES = [1, 2.000000, 5.000000,209.000000]
MONETARY_QUARTILES = [3.750000, 307.415000, 674.485000, 1661.740000]

def calculate_rfm_scores(recency: float, frequency: float, monetary: float) -> tuple:
    # Reverse scoring for Recency: lower value = higher score
    r_score = 4
    if recency > RECENCY_QUARTILES[2]:
        r_score = 1
    elif recency > RECENCY_QUARTILES[1]:
        r_score = 2
    elif recency > RECENCY_QUARTILES[0]:
        r_score = 3

    # Normal scoring for Frequency and Monetary: higher value = higher score
    f_score = 1
    if frequency > FREQUENCY_QUARTILES[0]:
        f_score = 2
    if frequency > FREQUENCY_QUARTILES[1]:
        f_score = 3
    if frequency > FREQUENCY_QUARTILES[2]:
        f_score = 4

    m_score = 1
    if monetary > MONETARY_QUARTILES[0]:
        m_score = 2
    if monetary > MONETARY_QUARTILES[1]:
        m_score = 3
    if monetary > MONETARY_QUARTILES[2]:
        m_score = 4

    return r_score, f_score, m_score

# --- Main Streamlit Application ---

st.title("Customer Segmentation Prediction App")
st.write("This application uses your trained RFM model to predict a customer's segment. "
    "Please ensure you have the `scaler.joblib` and `model.joblib` files in the same directory.")

# Check for and load the saved model and scaler
try:
    # Load the trained scaler and model
    scaler = joblib.load('xgb_model.joblib')
    model = joblib.load('Standard_scaler.joblib')
    st.sidebar.success("Model and Scaler loaded successfully!")
    model_loaded = True
except FileNotFoundError:
    st.error("Error: `scaler.joblib` or `model.joblib` not found. "
        "Please ensure your trained model files are in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()

st.header("Enter Customer RFM Data")
st.write("Please provide the following customer metrics to get a segment prediction:")

# Create input widgets for user to enter data
col1, col2, col3 = st.columns(3)
with col1:
    recency = st.number_input("Recency (Days since last purchase)", min_value=1, value=90)
with col2:
    frequency = st.number_input("Frequency (Total transactions)", min_value=1, value=5)
with col3:
    monetary = st.number_input("Monetary (Total spend)", min_value=10.0, value=500.0)

# The predict button
if st.button("Predict Segment"):
    # Calculate the RFM scores based on the user's input
    r_score, f_score, m_score = calculate_rfm_scores(recency, frequency, monetary)

    st.subheader("Calculated RFM Scores")
    st.write(f"R-Score: `{r_score}` | F-Score: `{f_score}` | M-Score: `{m_score}`")

    # Create a DataFrame for prediction
    # The model expects a 2D array, so we use reshape(-1, 1) or pass a list of lists.
    # The feature order must be the same as the training data: [r_score, f_score, m_score]
    user_data = np.array([[r_score, f_score, m_score]])

    # Transform the user data using the loaded scaler
    scaled_user_data = scaler.transform(user_data)

    # Make the prediction
    prediction = model.predict(scaled_user_data)

    # Display the result
    st.header("Predicted Customer Segment")
    if prediction[0] == 0:
        st.success("The predicted segment is: **'At Risk Customers'**")
    elif prediction[0] == 1:
        st.success("The predicted segment is: **'Valuable Customers (Outliers)'**")
    elif prediction[0] == 2:
        st.success("The predicted segment is: **'Moderate Engagement Customers'**")
    elif prediction[0] == 3:
        st.success("The predicted segment is: **'Engaged Customers'**")
    elif prediction[0] == 4:
        st.success("The predicted segment is: **'Engaged Customers'**")
    elif prediction[0] == 5:
        st.success("The predicted segment is: **'Engaged Customers'**")
    elif prediction[0] == 6:
        st.success("The predicted segment is: **'Engaged Customers'**")
    else:
        st.warning("Prediction result is outside of known segments.")

    st.write("---")
    st.write("The model made this prediction based on the input RFM scores. "
        "The segment names are based on common RFM analysis categories.")

