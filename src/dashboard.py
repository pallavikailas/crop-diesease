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
crop_type = st.selectbox("Crop_Type", ['Rice', 'Maize', 'Wheat', 'Barley', 'Soybean'])
growth_stage = st.selectbox("Growth Stage", [1, 2, 3])
temperature = st.slider("Temperature", min_value=10, max_value=40, step=1)
humidity = st.slider("Humidity", min_value=0, max_value=110, step=1)
rainfall = st.slider("Rainfall", min_value=-50, max_value=250, step=1)
soil_moisture = st.slider("Soil Moisture", min_value=5, max_value=57, step=1)
wind_speed = st.slider("Wind_Speed", min_value=-1, max_value=11, step=1)
sunlight_hours = st.slider("Sunlight_Hours", min_value=1, max_value=15, step=1)
soil_ph = st.slider("Soil_pH", min_value=4.7, max_value=8.2, step=0.1)
surrounding_crop_diversity = st.slider("Surrounding_Crop_Diversity", min_value=-0.3, max_value=6.9, step=0.1)

treatment_recommendations = {
    'Root Rot': "Ensure proper drainage and avoid overwatering. Use fungicides if necessary.",
    'Leaf Spot': "Remove infected leaves and apply fungicides. Ensure good air circulation around plants.",
    'Fungal Wilt': "Rotate crops and use disease-resistant varieties. Apply fungicides as a preventative measure.",
    'Stem Rot': "Improve soil drainage, avoid injuries to stems, and apply fungicides as needed.",
    'Rust': "Use resistant plant varieties and apply fungicides. Avoid overhead irrigation.",
    'Spot': "Prune affected areas and apply appropriate fungicides.",
    'Bacterial Blight': "Use disease-free seeds and avoid overhead watering. Apply copper-based bactericides.",
    'Anthracnose': "Remove infected plants and apply fungicides. Practice crop rotation.",
    'Blight': "Remove infected plants, apply fungicides, and ensure proper spacing for air circulation.",
    'Mildew': "Apply sulfur or potassium bicarbonate sprays. Ensure good ventilation.",
    'Powdery Mildew': "Apply fungicides, avoid overhead watering, and ensure adequate spacing.",
    'Downy Mildew': "Use resistant varieties, remove infected leaves, and apply fungicides as needed.",
    'Wilt': "Improve soil drainage and consider crop rotation. Remove infected plants promptly."
}

if st.button("Predict"):
    crop_type_encoded = crop_type_encoder.transform([crop_type])[0]

    # Create input data as a DataFrame with adjusted column names
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Rainfall': [rainfall],
        'Soil_Moisture': [soil_moisture],
        'Wind_Speed': [wind_speed],
        'Sunlight_Hours': [sunlight_hours],
        'Soil_pH': [soil_ph],
        'Growth_Stage': [growth_stage],
        'Surrounding_Crop_Diversity': [surrounding_crop_diversity],
        'Crop_Type': [crop_type_encoded]
    })

    # Predict using the model
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)

    st.write(f"Predicted Crop Disease Outbreak: {prediction[0]}")
    st.write("**Treatment Recommendation:**")
    st.write(treatment_recommendations.get(prediction[0], "No specific treatment available."))
