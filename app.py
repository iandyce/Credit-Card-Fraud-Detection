import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -------------------------------
# Load Models and Scaler
# -------------------------------
with open("logistic_regression.pkl", "rb") as f:
    log_reg = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    rf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

# Sidebar
st.sidebar.header("🔍 Options")
option = st.sidebar.radio("Choose Action", ["Upload Dataset", "Check Single Transaction", "About"])

# -------------------------------
# Upload Dataset
# -------------------------------
if option == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.write(data.head())

        if "Class" in data.columns:
            st.subheader("⚖️ Fraud vs Non-Fraud Cases")
            fig, ax = plt.subplots()
            sns.countplot(x="Class", data=data, ax=ax, palette="Set2")
            st.pyplot(fig)

        # Predict using Random Forest
        if "Class" in data.columns:
            X = data.drop(columns=["Class"])
        else:
            X = data.copy()

        X_scaled = scaler.transform(X)
        predictions = rf.predict(X_scaled)
        probabilities = rf.predict_proba(X_scaled)[:, 1]

        data["Fraud Prediction"] = predictions
        data["Fraud Probability"] = probabilities

        st.subheader("🔎 Prediction Results")
        st.write(data.head())

        st.subheader("📈 Fraud Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(probabilities, bins=20, kde=True, ax=ax, color="red")
        ax.set_xlabel("Fraud Probability")
        st.pyplot(fig)

        st.subheader("🌳 Random Forest Feature Importance")
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax, palette="viridis")
        st.pyplot(fig)

# -------------------------------
# Single Transaction Checker
# -------------------------------
elif option == "Check Single Transaction":
    st.subheader("📝 Enter Transaction Details")

    # Example input fields (you can expand with more features if needed)
    time = st.number_input("Time", min_value=0, value=0)
    amount = st.number_input("Amount", min_value=0.0, value=100.0)

    # For simplicity: V1–V28 features can be set to 0
    other_features = [0] * 28
    input_data = pd.DataFrame([[time] + other_features + [amount]],
                              columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = rf.predict(input_scaled)[0]
    probability = rf.predict_proba(input_scaled)[0][1]

    st.write("### Prediction Result")
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of Fraud: {probability:.2f})")

# -------------------------------
# About Page
# -------------------------------
else:
    st.subheader("ℹ️ About This Project")
    st.write("""
        This dashboard allows interactive analysis of credit card transactions 
        to detect fraudulent activities. It supports:
        - Uploading datasets for fraud prediction.
        - Visualizing fraud distributions and feature importance.
        - Checking single transaction risk in real-time.
    """)
