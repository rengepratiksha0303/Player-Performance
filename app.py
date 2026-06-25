import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_model (3).pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("⚽ FIFA Player Performance Predictor")

user_data = {}

for feature in feature_names:
    user_data[feature] = st.number_input(
        feature,
        value=0.0
    )

if st.button("Predict"):

    input_df = pd.DataFrame([user_data])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Performance Score: {prediction[0]:.2f}"
    )
