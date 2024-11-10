import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = joblib.load("ensemble_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define a label encoder for categorical inputs if not already fitted
crop_type_encoder = LabelEncoder()
crop_type_encoder.classes_ = np.array(['Rice', 'Maize', 'Wheat', 'Barley', 'Soybean'])  # Example categories

# Title and description for the app
st.title("Crop Disease Outbreak Prediction")
st.write("Predict the likelihood and type of crop disease outbreak based on various environmental and crop factors.")

# Crop Type (Categorical Input)
crop_type = st.selectbox("Crop Type", ['Rice', 'Maize', 'Wheat', 'Barley', 'Soybean'])
growth_stage = st.selectbox("Growth Stage", [1, 2, 3])
temperature = st.slider("Temperature (°C)", min_value=10, max_value=40, step=1)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)
precipitation = st.slider("Precipitation (mm)", min_value=0, max_value=50, step=1)
soil_moisture = st.slider("Soil Moisture (%)", min_value=0, max_value=100, step=1)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, step=1)
sunlight = st.slider("Sunlight (hours)", min_value=0, max_value=24, step=1)

# Soil pH (Continuous Input)
soil_ph = st.slider("Soil pH", min_value=5.0, max_value=8.0, step=0.1)

# Surrounding Crop Diversity (Continuous Input)
surrounding_crop_diversity = st.slider("Surrounding Crop Diversity", min_value=0.0, max_value=5.0, step=0.1)


if st.button("Predict"):
    # Encode categorical inputs using NumPy arrays
    # crop_type_encoded = crop_type_encoder.transform(np.array([crop_type]))[0]
    crop_type_encoded = crop_type_encoder.transform([crop_type])[0]

    # Create input data as a DataFrame
    input_data = pd.DataFrame({
        'Crop_Type': [crop_type_encoded], 
        'Growth_Stage': [growth_stage],
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
        'Precipitation (mm)': [precipitation],
        'Soil_Moisture (%)': [soil_moisture],
        'Wind_Speed (km/h)': [wind_speed],
        'Sunlight (hours)': [sunlight],
        'Soil_pH': [soil_ph],
        'Surrounding_Crop_Diversity': [surrounding_crop_diversity]
    })
    

    # Predict using the model
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)

    st.write(f"Predicted Crop Disease Outbreak: {prediction[0]}")
