import streamlit as st
import pandas as pd
from dataloader import load_and_preprocess_data
from model_training import load_model
from feature_engineering import add_features

# Load model
model = load_model("ensemble_model.pkl")  # Ensure the path matches the saved model location

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
    # Create input data for prediction
    input_data = pd.DataFrame([[temperature, humidity, precipitation, wind_speed]], 
                              columns=["temperature", "humidity", "precipitation", "wind_speed"])
    
    # Add features
    input_data = add_features(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write("Disease outbreak likelihood:", "High" if prediction[0] == 1 else "Low")
