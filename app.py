import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load models
with open("logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Scaler (fit once with training data, here simplified by re-fitting on uploaded data)
scaler = StandardScaler()

# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload transaction data and predict fraud using **Logistic Regression** and **Random Forest** models.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocess (scale features, drop target column if exists)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])  # drop label if included

    df_scaled = scaler.fit_transform(df)

    # Predictions
    st.subheader("Prediction Results")
    predictions = rf_model.predict(df_scaled)
    fraud_count = (predictions == 1).sum()
    nonfraud_count = (predictions == 0).sum()

    st.success(f"✅ Legit Transactions: {nonfraud_count}")
    st.error(f"🚨 Fraudulent Transactions: {fraud_count}")

    # Fraud vs Non-Fraud Bar Chart
    st.subheader("Fraud vs Non-Fraud Predictions")
    fig, ax = plt.subplots()
    ax.bar(["Non-Fraud", "Fraud"], [nonfraud_count, fraud_count], color=["green", "red"])
    ax.set_ylabel("Number of Transactions")
    st.pyplot(fig)

    # Fraud Probability Distribution
    st.subheader("Fraud Probability Distribution")
    fraud_probs = rf_model.predict_proba(df_scaled)[:, 1]
    fig, ax = plt.subplots()
    ax.hist(fraud_probs, bins=50, color="orange", alpha=0.7)
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Number of Transactions")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance (Random Forest)")
    importance_df = pd.DataFrame({
        "Feature": df.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature").head(10))

# Sidebar - Single Transaction Check
st.sidebar.header("🔎 Check a Single Transaction")

# Build input fields (for demo we only use Amount + Time, rest set to 0)
time = st.sidebar.number_input("Time (seconds)", min_value=0.0, value=5000.0)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)

# Placeholder for 28 PCA features
extra_features = [0] * 28
single_input = pd.DataFrame([[time, amount] + extra_features], 
                            columns=(["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]))

# Scale and predict
single_scaled = scaler.fit_transform(single_input)  # ⚠️ Here we re-fit; ideally use training scaler
single_pred = rf_model.predict(single_scaled)[0]

st.sidebar.subheader("Prediction Result")
if single_pred == 1:
    st.sidebar.error("🚨 Fraudulent Transaction Detected!")
else:
    st.sidebar.success("✅ Legitimate Transaction")
