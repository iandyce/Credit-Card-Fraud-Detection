import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# =========================
# Load Models
# =========================
log_reg = joblib.load("logistic_regression_model.pkl")
rf = joblib.load("random_forest_model.pkl")

# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

st.write("Upload transaction data and predict fraud using Logistic Regression and Random Forest models.")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Drop Class column if present (real data might not have it)
    if "Class" in df.columns:
        df_features = df.drop("Class", axis=1)
    else:
        df_features = df

    # Scale data (important for Logistic Regression)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)

    # =========================
    # Predictions
    # =========================
    st.subheader("Model Predictions")

    df["LogReg_Prob"] = log_reg.predict_proba(df_scaled)[:, 1]
    df["RF_Prob"] = rf.predict_proba(df_scaled)[:, 1]

    df["LogReg_Prediction"] = (df["LogReg_Prob"] > 0.5).astype(int)
    df["RF_Prediction"] = (df["RF_Prob"] > 0.5).astype(int)

    st.write("Prediction Results:")
    st.dataframe(df.head(20))

    # Fraud stats
    fraud_counts = df[["LogReg_Prediction", "RF_Prediction"]].sum()
    st.subheader("🔎 Fraud Detection Summary")
    st.write(f"**Logistic Regression detected:** {fraud_counts['LogReg_Prediction']} fraud cases")
    st.write(f"**Random Forest detected:** {fraud_counts['RF_Prediction']} fraud cases")

    # =========================
    # Download Predictions
    # =========================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("⬆️ Please upload a CSV file to start predictions.")
