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
# Custom CSS (Enhanced Readability)
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Titles */
.title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    color: white;
}

.subtitle {
    text-align: center;
    color: #e5e7eb;
    margin-bottom: 2rem;
}

/* Card */
.card {
    background: rgba(255,255,255,0.08);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

/* Input labels */
label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}

/* Button - black default, green on hover */
.stButton > button {
    background-color: black;
    color: white !important;
    font-size: 18px;
    padding: 0.6rem 2.5rem;
    border-radius: 10px;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white !important;
}

/* Result cards */
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
st.markdown("<div class='subtitle'>Machine-Learning Powered Transaction Risk Analysis</div>", unsafe_allow_html=True)

# =============================
# Sidebar
# =============================
st.sidebar.title("📊 Model Information")
st.sidebar.write("""
• Algorithm: **XGBoost**  
• Input Features: **30**  
• Scaling: **StandardScaler**  
• Fraud Rate: **~0.17%**
""")

# =============================
# Input Card
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)

with st.form("fraud_form"):
    st.markdown("<div class='section-header'>🧾 Core Transaction Details</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        time = st.number_input("Transaction Time (seconds since first transaction)", min_value=0.0, format="%.6f")
    with col2:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.6f")

    st.markdown("<div class='section-header'>🔢 PCA Encoded Features</div>", unsafe_allow_html=True)
    st.caption("These features are anonymized representations used by the fraud detection model.")

    v_inputs = []

    for i in range(0, 28, 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i + j + 1
            v_inputs.append(
                cols[j].number_input(
                    f"PCA Feature V{idx}",
                    value=0.0,
                    format="%.6f"
                )
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
            <p><b>Risk Probability:</b> {probability:.6%}</p>
            <p>This transaction exhibits strong fraud patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe">
            <h2>✅ Legitimate Transaction</h2>
            <p><b>Fraud Probability:</b> {probability:.6%}</p>
            <p>No suspicious behavior detected.</p>
        </div>
        """, unsafe_allow_html=True)
