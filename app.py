# ===========================
# app.py
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# =====================================================
# 1️⃣ PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telco Churn Prediction By Using TensorFlow ML")
st.markdown(
    """
    Predict customer churn probability using a TensorFlow model trained on Telco customer data.  
    Enter customer details below to get a prediction.  
    The Power BI dashboard provides exploratory analysis and business insights.
    """
)

st.divider()

# =====================================================
# 2️⃣ LOAD MODEL AND ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("Telco_churn_model.keras")
    selected_features = joblib.load("selected_features.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, selected_features, le

model, selected_features, le = load_artifacts()

st.header("🧾 Customer Information")

# =====================================================
# 3️⃣ BUSINESS-FRIENDLY USER INPUTS
# =====================================================

# ---------- Numeric ----------
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges (RM)", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges (RM)", min_value=0.0, value=1000.0)

# ---------- Demographics ----------
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])

# ---------- Services ----------
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

# ---------- Contract & Billing ----------
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Credit card (automatic)", "Bank transfer (automatic)"]
)

# =====================================================
# 4️⃣ MAP INPUTS TO ENCODED FEATURES
# =====================================================
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    
    "gender_Male": int(gender == "Male"),
    "SeniorCitizen_Yes": int(senior == "Yes"),
    "Partner_Yes": int(partner == "Yes"),
    "Dependents_Yes": int(dependents == "Yes"),
    
    "MultipleLines_No phone service": int(multiple_lines == "No phone service"),
    "MultipleLines_Yes": int(multiple_lines == "Yes"),
    
    "InternetService_Fiber optic": int(internet_service == "Fiber optic"),
    "InternetService_No": int(internet_service == "No"),
    
    "OnlineSecurity_No internet service": int(online_security == "No internet service"),
    "OnlineSecurity_Yes": int(online_security == "Yes"),
    
    "OnlineBackup_Yes": int(online_backup == "Yes"),
    
    "DeviceProtection_No internet service": int(device_protection == "No internet service"),
    "DeviceProtection_Yes": int(device_protection == "Yes"),
    
    "TechSupport_No internet service": int(tech_support == "No internet service"),
    "TechSupport_Yes": int(tech_support == "Yes"),
    
    "StreamingTV_No internet service": int(streaming_tv == "No internet service"),
    "StreamingTV_Yes": int(streaming_tv == "Yes"),
    
    "StreamingMovies_No internet service": int(streaming_movies == "No internet service"),
    "StreamingMovies_Yes": int(streaming_movies == "Yes"),
    
    "Contract_One year": int(contract == "One year"),
    "Contract_Two year": int(contract == "Two year"),
    
    "PaperlessBilling_Yes": int(paperless == "Yes"),
    
    "PaymentMethod_Credit card (automatic)": int(payment_method == "Credit card (automatic)"),
    "PaymentMethod_Electronic check": int(payment_method == "Electronic check"),
    "PaymentMethod_Mailed check": int(payment_method == "Mailed check")
}

# Convert to DataFrame with correct column order
X_input = pd.DataFrame([input_data])[selected_features].values

# =====================================================
# 5️⃣ PREDICTION BUTTON
# =====================================================
st.divider()

if st.button("🔍 Predict Churn"):
    prob = model.predict(X_input)[0][0]
    churn_class = int(prob >= 0.5)
    churn_label = le.inverse_transform([churn_class])[0]

    st.subheader("📊 Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")
    st.write(f"**Predicted Churn:** `{churn_label}`")
    st.progress(min(prob, 1.0))

# =====================================================
# 6️⃣ POWER BI DASHBOARD EMBED
# =====================================================
st.divider()
st.header("📈 Power BI Dashboard")

st.markdown(
    """
    The dashboard below provides business insights and customer churn analysis.
    """
)

# Replace this with your actual Power BI embed URL
POWER_BI_EMBED_URL = "https://app.powerbi.com/view?r=eyJrIjoiMmI1ZTQyMWMtMTIwYi00NDY3LTgzYmQtZjkyOThiY2EzNmI1IiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D"

st.components.v1.iframe(
    src=POWER_BI_EMBED_URL,
    width=1200,
    height=700,
    scrolling=True
)

# =====================================================
# 7️⃣ FOOTER
# =====================================================
st.divider()
st.caption(
    "Built with TensorFlow, Scikit-Learn, Power BI & Streamlit | Portfolio Project"
)
