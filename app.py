import streamlit as st
import numpy as np
import joblib

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #d1d5db;
    margin-bottom: 2rem;
}

/* Card */
.card {
    background-color: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.3);
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    font-size: 18px;
    padding: 0.6rem 2rem;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    transform: scale(1.02);
}

/* Result boxes */
.result-safe {
    background-color: rgba(34, 197, 94, 0.15);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 6px solid #22c55e;
}

.result-fraud {
    background-color: rgba(239, 68, 68, 0.15);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 6px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Load Model & Scaler
# =============================
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# =============================
# Sidebar
# =============================
st.sidebar.title("ℹ️ About This App")
st.sidebar.write("""
This application uses an **XGBoost Machine Learning model**
trained on real credit card transaction data to detect **fraudulent transactions**.

✔ Automatic feature scaling  
✔ Real-time prediction  
✔ Beginner-friendly interface
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Built for ML & FinTech learning")

# =============================
# Header
# =============================
st.markdown("<div class='main-title'>💳 Credit Card Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter transaction details to check if it is fraudulent</div>", unsafe_allow_html=True)

# =============================
# Layout
# =============================
left, center, right = st.columns([1, 2, 1])

with center:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    with st.form("fraud_form"):
        col1, col2 = st.columns(2)

        with col1:
            time = st.number_input("🕒 Transaction Time", min_value=0.0)
            v1 = st.number_input("V1")
            v2 = st.number_input("V2")
            v3 = st.number_input("V3")

        with col2:
            v4 = st.number_input("V4")
            v5 = st.number_input("V5")
            amount = st.number_input("💰 Transaction Amount", min_value=0.0)

        submit = st.form_submit_button("🔍 Check Transaction")

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction Logic
# =============================
if submit:
    with st.spinner("Analyzing transaction..."):
        input_data = np.array([[time, v1, v2, v3, v4, v5, amount]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("""
        <div class="result-fraud">
            <h2>🚨 Fraudulent Transaction Detected</h2>
            <p>This transaction shows strong indicators of fraud.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-safe">
            <h2>✅ Transaction is Legitimate</h2>
            <p>No suspicious activity detected.</p>
        </div>
        """, unsafe_allow_html=True)
