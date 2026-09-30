import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinShield | Payment Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


# ============================================================
# LOAD MODEL + SCALER
# ============================================================

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model_and_scaler()


# ============================================================
# SESSION STATE
# ============================================================

if "transaction_time" not in st.session_state:
    st.session_state.transaction_time = 50000.0

if "transaction_amount" not in st.session_state:
    st.session_state.transaction_amount = 100.0

for i in range(1, 29):
    if f"v{i}" not in st.session_state:
        st.session_state[f"v{i}"] = 0.0


# ============================================================
# EXAMPLE LOADERS
# ============================================================

def load_genuine_example():
    st.session_state.transaction_time = 50000.0
    st.session_state.transaction_amount = 100.0

    for i in range(1, 29):
        st.session_state[f"v{i}"] = 0.0


def load_fraud_example():
    st.session_state.transaction_time = 60000.0
    st.session_state.transaction_amount = 150.0

    for i in range(1, 29):
        st.session_state[f"v{i}"] = 0.0

    st.session_state.v10 = -6.2
    st.session_state.v12 = -5.8
    st.session_state.v14 = -7.1
    st.session_state.v17 = -6.4


def reset_values():
    st.session_state.transaction_time = 50000.0
    st.session_state.transaction_amount = 100.0

    for i in range(1, 29):
        st.session_state[f"v{i}"] = 0.0


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .fraud-card {
        background-color: #3b1515;
        border: 1px solid #ef4444;
    }

    .normal-card {
        background-color: #102a1a;
        border: 1px solid #22c55e;
    }

    .probability {
        font-size: 32px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ FinShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Payment Transaction Anomaly Detection System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💳 Transaction Details")

    transaction_time = st.number_input(
        "Transaction Time (seconds)",
        min_value=0.0,
        step=100.0,
        key="transaction_time"
    )

    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        step=10.0,
        key="transaction_amount"
    )

    st.divider()

    st.markdown("### 🧪 Demo Transactions")

    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        st.button(
            "🟢 Genuine",
            use_container_width=True,
            on_click=load_genuine_example
        )

    with demo_col2:
        st.button(
            "🔴 Fraud",
            use_container_width=True,
            on_click=load_fraud_example
        )

    st.divider()

    st.markdown("### 🧬 PCA Features (V1–V28)")

    st.caption(
        "Anonymized PCA features from the Credit Card Fraud Detection dataset."
    )

    v_features = {}

    for i in range(1, 29):

        v_features[f"V{i}"] = st.number_input(
            f"V{i}",
            format="%.4f",
            key=f"v{i}"
        )


# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown("### 🔍 Transaction Analysis")

st.info(
    "Enter transaction details manually or load a demo transaction, "
    "then click **Analyze Transaction**."
)


# ============================================================
# DEMO INFORMATION
# ============================================================

with st.expander("📋 Demo Values"):

    st.markdown("### 🟢 Genuine Example")

    st.code(
        """Time = 50000
Amount = 100
V1–V28 = 0"""
    )

    st.markdown("### 🔴 Fraud Example")

    st.code(
        """Time = 60000
Amount = 150
V10 = -6.2
V12 = -5.8
V14 = -7.1
V17 = -6.4
All other V features = 0"""
    )


# ============================================================
# ACTION BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    analyze_button = st.button(
        "🔍 Analyze Transaction",
        use_container_width=True,
        type="primary"
    )

with col2:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True,
        on_click=reset_values
    )


# ============================================================
# RESET
# ============================================================

if reset_button:
    st.rerun()


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    try:

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if transaction_amount < 0:
            st.error("Transaction amount cannot be negative.")
            st.stop()

        if transaction_time < 0:
            st.error("Transaction time cannot be negative.")
            st.stop()


        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = {
            "Time": transaction_time,
            "Amount": transaction_amount
        }

        input_data.update(v_features)

        input_df = pd.DataFrame([input_data])


        # ----------------------------------------------------
        # Create Hour feature
        # Same feature engineering used during training
        # ----------------------------------------------------

        input_df["Hour"] = (
            (input_df["Time"] // 3600) % 24
        )


        # ----------------------------------------------------
        # Scale Time and Amount
        # ----------------------------------------------------

        input_df[["Time", "Amount"]] = scaler.transform(
            input_df[["Time", "Amount"]]
        )


        # ----------------------------------------------------
        # Match EXACT model feature order
        # ----------------------------------------------------

        expected_columns = model.feature_names_in_

        input_df = input_df[expected_columns]


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        probability_percent = probability * 100


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-card fraud-card">

                <h2>🚨 Fraudulent Transaction Detected</h2>

                <p class="probability">
                Fraud Probability: {probability_percent:.2f}%
                </p>

                <p>
                The model classified this transaction as potentially
                fraudulent/anomalous.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            if probability >= 0.80:
                risk = "HIGH RISK"
            elif probability >= 0.50:
                risk = "MEDIUM RISK"
            else:
                risk = "LOW RISK"

            st.warning(f"Risk Assessment: **{risk}**")


        else:

            st.markdown(
                f"""
                <div class="result-card normal-card">

                <h2>✅ Genuine Transaction</h2>

                <p class="probability">
                Fraud Probability: {probability_percent:.2f}%
                </p>

                <p>
                The model classified this transaction as normal.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            if probability < 0.20:
                risk = "LOW RISK"
            elif probability < 0.50:
                risk = "MEDIUM RISK"
            else:
                risk = "HIGH RISK"

            st.success(f"Risk Assessment: **{risk}**")


        # ----------------------------------------------------
        # Processed data
        # ----------------------------------------------------

        with st.expander("📋 View Processed Transaction Data"):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"An error occurred while processing the transaction: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#888; padding:10px;">

    <b>FinShield — AI-Powered Payment Transaction Anomaly Detection System</b>
    <br><br>
    Educational Machine Learning Prototype | Binary Classification

    </div>
    """,
    unsafe_allow_html=True
)