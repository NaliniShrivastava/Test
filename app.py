import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt

st.title("🚦 Smart Traffic Insights - Mini AI System")

# Upload CSV
uploaded_file = st.file_uploader("Upload Traffic Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Show columns
    st.write("Available columns:", df.columns.tolist())

    # Choose target column
    target = st.selectbox("Select target column", df.columns)

    # Prepare features
    X = df.drop(columns=[target])
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Prepare target
    y_raw = df[target]

    # Try numeric conversion
    y_numeric = pd.to_numeric(y_raw, errors="coerce")

    if y_numeric.notnull().sum() > 0.8 * len(y_raw):
        # Mostly numeric target → regression
        y = y_numeric.fillna(0)
        problem_type = "regression"
    else:
        # Categorical target → classification
        y = y_raw.astype(str)
        problem_type = "classification"

    st.write(f"Detected Problem Type: **{problem_type}**")

    if X.shape[1] == 0:
        st.error("❌ No numeric features available for training!")
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if problem_type == "regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.subheader("✅ Regression Results")
            st.write(f"R² Score: {r2_score(y_test, preds):.2f}")

            # Plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.5)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

        else:  # classification
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.subheader("✅ Classification Results")
            st.write(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

            # Show prediction samples
            st.write("Sample Predictions:", list(zip(y_test[:10], preds[:10])))

        # 🔮 Try prediction on custom input
        st.subheader("🔮 Try Custom Prediction")
        sample_input = {}
        for col in X.columns[:5]:  # limit to first 5 features for UI simplicity
            val = st.number_input(
                f"{col}",
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].mean())
            )
            sample_input[col] = val

        if st.button("Predict Traffic"):
            input_df = pd.DataFrame([sample_input])
            result = model.predict(input_df)[0]
            st.success(f"Predicted Value: {result}")
