import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it is fraudulent.")

# Load trained model
model = joblib.load("xgb_fraud_model.pkl")

st.subheader("Transaction Features")

# Input fields (same order as training data)
time = st.number_input("Time", value=0.0)

v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", value=0.0)
    v_features.append(v)

amount = st.number_input("Amount", value=0.0)

# Combine inputs into model-ready format
features = np.array([[time] + v_features + [amount]])

if st.button("Predict"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction is Normal")
