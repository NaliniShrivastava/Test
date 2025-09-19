# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Smart Traffic Insights", layout="wide")

st.title("🚦 Smart Traffic Insights for Indian Cities")
st.markdown("Predict and analyze traffic patterns using Machine Learning.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your traffic dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Convert datetime if present
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        df["hour"] = df["date_time"].dt.hour
        df["day"] = df["date_time"].dt.day
        df["month"] = df["date_time"].dt.month

    # Select target & features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = st.selectbox("Select Target Column (y)", options=numeric_cols)
    features = st.multiselect(
        "Select Feature Columns (X)", options=numeric_cols, default=[c for c in numeric_cols if c != target]
    )

    if len(features) > 0:
        X = df[features]
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader("📈 Model Performance")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

        # Plot Actual vs Predicted
        st.subheader("🔍 Prediction Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred, alpha=0.5, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Traffic")
        st.pyplot(fig)

        # Prediction form
        st.subheader("🛠 Try a Custom Prediction")
        input_data = {}
        for col in features:
            val = st.number_input(f"Enter value for {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
            input_data[col] = val

        if st.button("Predict Traffic"):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            st.success(f"Predicted Traffic ({target}): {pred:.2f}")

else:
    st.info("👆 Please upload a CSV file to get started.")
