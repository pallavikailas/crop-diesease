import streamlit as st
import pandas as pd
import joblib
from src.preprocessing import preprocess_data
from src.training import load_model

# Load model
model = load_model("path_to_saved_model.joblib")  # Replace with the actual model path

# Title and description
st.title("Crop Disease Outbreak Prediction")
st.write("Predict the likelihood of a crop disease outbreak based on weather conditions.")

# User inputs
temperature = st.slider("Temperature (°C)", min_value=10, max_value=40, step=1)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)
precipitation = st.slider("Precipitation (mm)", min_value=0, max_value=50, step=1)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, step=1)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[temperature, humidity, precipitation, wind_speed]], 
                              columns=["temperature", "humidity", "precipitation", "wind_speed"])
    processed_data = preprocess_data(input_data)   # Preprocess as per the model's needs
    prediction = model.predict(processed_data)
    st.write("Disease outbreak likelihood:", "High" if prediction[0] == 1 else "Low")
