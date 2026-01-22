import streamlit as st
import numpy as np
import joblib

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =============================
# Custom CSS (Fintech UI)
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #d1d5db;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

.stButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    font-size: 18px;
    padding: 0.6rem 2.5rem;
    border-radius: 10px;
    border: none;
}

.fraud {
    border-left: 6px solid #ef4444;
    background: rgba(239,68,68,0.15);
    padding: 1.5rem;
    border-radius: 12px;
}

.safe {
    border-left: 6px solid #22c55e;
    background: rgba(34,197,94,0.15);
    padding: 1.5rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Load Model & Scaler
# =============================
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# =============================
# Header
# =============================
st.markdown("<div class='title'>💳 Credit Card Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>ML-powered real-time transaction risk analysis</div>", unsafe_allow_html=True)

# =============================
# Sidebar
# =============================
st.sidebar.title("📊 Model Info")
st.sidebar.write("""
• Algorithm: **XGBoost**  
• Dataset: Credit Card Transactions  
• Scaling: **StandardScaler**  
• Fraud is extremely rare (<0.2%)
""")

st.sidebar.markdown("---")
st.sidebar.write("🧠 Inputs are automatically scaled before prediction")

# =============================
# Input Card
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)

with st.form("fraud_form"):
    st.subheader("🧾 Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        time = st.number_input("⏱ Time", min_value=0.0)
    with col2:
        amount = st.number_input("💰 Amount", min_value=0.0)

    st.markdown("### 🔢 PCA Features (V1 – V28)")

    v_inputs = []

    for i in range(0, 28, 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i + j + 1
            v_inputs.append(
                cols[j].number_input(f"V{idx}", value=0.0)
            )

    submit = st.form_submit_button("🔍 Analyze Transaction")

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction
# =============================
if submit:
    with st.spinner("Analyzing transaction risk..."):
        input_data = np.array([[time] + v_inputs + [amount]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="fraud">
            <h2>🚨 Fraud Detected</h2>
            <p><b>Risk Probability:</b> {probability:.2%}</p>
            <p>This transaction shows strong fraud indicators.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe">
            <h2>✅ Legitimate Transaction</h2>
            <p><b>Fraud Probability:</b> {probability:.2%}</p>
            <p>No suspicious activity detected.</p>
        </div>
        """, unsafe_allow_html=True)
