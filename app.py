import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Dark mode CSS
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
h1, h2, h3, h4 {
    color: white;
}
label {
    color: white !important;
}
.stNumberInput > div > div > input {
    background-color: #020617;
    color: white;
    border-radius: 8px;
}
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #16a34a, #15803d);
}
.card {
    background-color: #020617;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#cbd5e1;'>Enter transaction details below. Inputs are automatically scaled.</p>", unsafe_allow_html=True)
st.markdown("---")

# Transaction info
st.markdown("<div class='card'><h3>📊 Transaction Information</h3></div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    time = st.number_input("Transaction Time (seconds)", min_value=0.0, step=0.000001, format="%.6f")
with col2:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.000001, format="%.6f")

# PCA features
st.markdown("<div class='card'><h3>🧠 PCA Features (V1 – V28)</h3></div>", unsafe_allow_html=True)
pca_values = []
for row in range(0, 28, 4):
    cols = st.columns(4)
    for j in range(4):
        idx = row + j + 1
        if idx <= 28:
            value = cols[j].number_input(f"PCA Feature V{idx}", value=0.0, step=0.000001, format="%.6f")
            pca_values.append(value)

# Prediction
st.markdown("<div class='card'><h3>🔍 Fraud Prediction</h3></div>", unsafe_allow_html=True)
if st.button("🚀 Predict Transaction"):
    user_input = np.array([[time] + pca_values + [amount]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED\nConfidence: **{probability:.2%}**")
    else:
        st.success(f"✅ Legitimate Transaction\nFraud Probability: **{probability:.2%}**")

st.markdown("<p style='text-align:center;color:#64748b;margin-top:2rem;'>Built with ❤️ using Streamlit & XGBoost</p>", unsafe_allow_html=True)
