import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("Smart Traffic Insights")

file = st.file_uploader("Upload CSV", type="csv")
if file:
    df = pd.read_csv(file)
    st.write("Preview:", df.head())

    # Simple setup
    target = 'traffic_volume'
    features = [c for c in df.select_dtypes(include='number').columns if c!=target]

    X = df[features].fillna(0)
    y = df[target].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    st.write("R²:", r2_score(y_test, y_pred))

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    st.pyplot(fig)
