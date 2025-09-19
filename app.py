# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1. App Title
# -------------------------
st.title("🚦 Smart Traffic Prediction")

# -------------------------
# 2. Upload Dataset
# -------------------------
uploaded_file = st.file_uploader("Upload your traffic dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)

    st.subheader("📊 Data Preview")
    st.write(df.head())

    # -------------------------
    # 3. Feature Engineering
    # -------------------------
    df['hour'] = df['date_time'].dt.hour
    df['weekday'] = df['date_time'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['date_time'].dt.month
    df['rolling_volume'] = df.groupby('junction')['traffic_volume'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)

    features = ['junction', 'hour', 'weekday', 'is_weekend', 'month', 'rolling_volume']

    # -------------------------
    # 4. Train-Test Split
    # -------------------------
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    y_train = train['traffic_volume']
    X_test = scaler.transform(test[features])
    y_test = test['traffic_volume']

    # -------------------------
    # 5. Model Training
    # -------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.2f}")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test['date_time'], y_test, label='Actual')
    ax.plot(test['date_time'], y_pred, label='Predicted')
    ax.set_title("Traffic Prediction vs Actual")
    ax.legend()
    st.pyplot(fig)

    # -------------------------
    # 6. Inference
    # -------------------------
    st.subheader("🔮 Predict Traffic")

    time_input = st.text_input("Enter time (YYYY-MM-DD HH:MM:SS)", "2025-09-19 08:30:00")
    junction_input = st.number_input("Enter Junction Number", min_value=int(df['junction'].min()), 
                                     max_value=int(df['junction'].max()), value=int(df['junction'].min()))

    if st.button("Predict Traffic"):
        dt = pd.to_datetime(time_input)
        hour = dt.hour
        weekday = dt.weekday()
        is_weekend = 1 if weekday >= 5 else 0
        month = dt.month
        rolling_volume = df[df['junction'] == junction_input]['traffic_volume'].tail(3).mean()

        input_features = np.array([[junction_input, hour, weekday, is_weekend, month, rolling_volume]])
        input_scaled = scaler.transform(input_features)
        pred_volume = model.predict(input_scaled)[0]

        congestion_level = int(pd.qcut(df['traffic_volume'], 5, labels=[1, 2, 3, 4, 5]).iloc[-1])
        peak_hour = df.groupby('hour')['traffic_volume'].mean().idxmax()

        st.success(f"🚗 Predicted Traffic Volume: **{int(pred_volume)}** vehicles")
        st.info(f"📊 Congestion Level (1-5): **{congestion_level}**")
        st.warning(f"⏰ Peak Traffic Hour: **{peak_hour}:00**")
