import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model and label encoder
model = joblib.load("models/ensemble_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Title and description for the app
st.title("Crop Disease Outbreak Prediction")
st.write("Predict the likelihood and type of crop disease outbreak based on various environmental and crop factors.")

# Define a label encoder for categorical inputs if not already fitted
crop_type_encoder = LabelEncoder()
season_encoder = LabelEncoder()

# Fit encoders with known values (replace with actual categories from training data)
crop_type_encoder.classes_ = ['Wheat', 'Corn', 'Rice', 'Sorghum', 'Barley', 'Oat', 'Lentil', 'Sugarcane','Soybean', 'Sunflower', 'Cotton', 'Peanut', 'Tomato', 'Millet', 'Cassava']
season_encoder.classes_ = ['Spring', 'Summer', 'Autumn', 'Winter']

# User inputs
crop_type = st.selectbox("Crop Type", crop_type_encoder.classes_)
season = st.selectbox("Season", season_encoder.classes_)
temperature = st.slider("Temperature (°C)", min_value=10, max_value=40, step=1)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)
precipitation = st.slider("Precipitation (mm)", min_value=0, max_value=50, step=1)
soil_moisture = st.slider("Soil Moisture (%)", min_value=0, max_value=100, step=1)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, step=1)
sunlight = st.slider("Sunlight (hours)", min_value=0, max_value=24, step=1)

if st.button("Predict"):
    # Encode categorical inputs
    crop_type_encoded = crop_type_encoder.transform([crop_type])[0]
    season_encoded = season_encoder.transform([season])[0]
    
    # Create input data as a DataFrame
    input_data = pd.DataFrame({
        'Crop Type': [crop_type_encoded],
        'Season': [season_encoded],
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
        'Precipitation (mm)': [precipitation],
        'Soil Moisture (%)': [soil_moisture],
        'Wind Speed (km/h)': [wind_speed],
        'Sunlight (hours)': [sunlight]
    })
    
    # Predict using the model
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)

    st.write(f"Predicted Crop Disease Outbreak: {prediction[0]}")
