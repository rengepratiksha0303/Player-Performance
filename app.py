import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("knn_model (3).pkl")
scaler = joblib.load("scaler (2).pkl")

st.set_page_config(
    page_title="FIFA Player Performance Predictor",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ FIFA Player Performance Predictor")
st.write("Predict player performance score using KNN Model")

st.sidebar.header("Player Details")

age = st.sidebar.number_input("Age", 16, 45, 25)
matches_played = st.sidebar.number_input("Matches Played", 0, 100, 20)
goals = st.sidebar.number_input("Goals", 0, 100, 5)
assists = st.sidebar.number_input("Assists", 0, 100, 3)
minutes_played = st.sidebar.number_input("Minutes Played", 0, 10000, 1500)

input_data = pd.DataFrame({
    "age":[age],
    "matches_played":[matches_played],
    "goals":[goals],
    "assists":[assists],
    "minutes_played":[minutes_played]
})

if st.button("Predict Performance"):

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    st.success(
        f"Predicted Performance Score: {prediction[0]:.2f}"
    )
