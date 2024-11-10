import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("ensemble_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Title and description for the app
st.title("Crop Disease Outbreak Prediction")
st.write("Predict the likelihood and type of crop disease outbreak based on various environmental and crop factors.")

# User inputs for prediction
crop_type = st.selectbox("Crop Type", options=['Wheat', 'Corn', 'Rice', 'Sorghum', 'Barley', 'Oat', 'Lentil', 'Sugarcane','Soybean', 'Sunflower', 'Cotton', 'Peanut', 'Tomato', 'Millet', 'Cassava'])  # Update options with actual crop type codes
season = st.selectbox("Season", options=['Summer', 'Spring', 'Winter', 'Autumn'])     # Update options with actual season codes
temperature = st.slider("Temperature (°C)", min_value=10, max_value=40, step=1)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)
precipitation = st.slider("Precipitation (mm)", min_value=0, max_value=50, step=1)
soil_moisture = st.slider("Soil Moisture (%)", min_value=0, max_value=100, step=1)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, step=1)
sunlight = st.slider("Sunlight (hours)", min_value=0, max_value=24, step=1)

# Prediction button
if st.button("Predict"):
    # Create a DataFrame for the input data based on user inputs
    input_data = pd.DataFrame({
        'Crop Type': [crop_type],
        'Season': [season],
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
        'Precipitation (mm)': [precipitation],
        'Soil Moisture (%)': [soil_moisture],
        'Wind Speed (km/h)': [wind_speed],
        'Sunlight (hours)': [sunlight]
    })

    # Make prediction
    prediction = model.predict(input_data)
    disease_name = label_encoder.inverse_transform(prediction)[0]

    # Display prediction result
    st.write("Predicted Disease Name:", disease_name)

