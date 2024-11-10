import streamlit as st
import pandas as pd
import joblib
from dataloader import load_and_preprocess_data
from visualisation import plot_correlation_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load("models/ensemble_model.pkl")

# Load the original dataset to help with LabelEncoder and StandardScaler initialization
processed_data, _ = load_and_preprocess_data('data/crop-disease.csv')

# Initialize the label encoder and scaler (using training data's fitted label encoder and scaler)
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Fit the LabelEncoder on the unique values in the training data for 'Crop Type' and 'Season'
label_encoder.fit(processed_data['Crop Type'])
season_encoder = LabelEncoder()
season_encoder.fit(processed_data['Season'])

# Fit the StandardScaler on the numerical columns from the training data
numerical_columns = ['Temperature (°C)', 'Humidity (%)', 'Precipitation (mm)', 
                     'Soil Moisture (%)', 'Wind Speed (km/h)', 'Sunlight (hours)']
scaler.fit(processed_data[numerical_columns])

# Title and description for the app
st.title("Crop Disease Outbreak Prediction")
st.write("Predict the likelihood and type of crop disease outbreak based on weather conditions.")

# User inputs for prediction
temperature = st.slider("Temperature (°C)", min_value=10, max_value=40, step=1)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)
precipitation = st.slider("Precipitation (mm)", min_value=0, max_value=50, step=1)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, step=1)

# Show the correlation matrix as a visualization option
if st.checkbox('Show Correlation Matrix'):
    data = pd.DataFrame({'Temperature (°C)': [temperature], 'Humidity (%)': [humidity],
                         'Precipitation (mm)': [precipitation], 'Wind Speed (km/h)': [wind_speed]})
    plot_correlation_matrix(data)

# Prediction button
if st.button("Predict"):
    # Create a DataFrame for the input data based on user inputs
    input_data = pd.DataFrame({
        'Crop Type': ['Wheat'],  # Default crop type for prediction (can be changed by user)
        'Season': ['Summer'],  # Default season (can be changed by user)
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
        'Precipitation (mm)': [precipitation],
        'Soil Moisture (%)': [50],  # Assuming a mid-range value for soil moisture
        'Wind Speed (km/h)': [wind_speed],
        'Sunlight (hours)': [6]  # Assuming 6 hours of sunlight
    })
    
    # Apply label encoding to categorical columns
    input_data['Crop Type'] = label_encoder.transform(input_data['Crop Type'])
    input_data['Season'] = season_encoder.transform(input_data['Season'])
    
    # Scale the numerical columns (same scaling used in the training data)
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
    
    # Get the prediction from the model
    prediction = model.predict(input_data)
    
    # Mapping the predicted output (Disease Name)
    disease_name = prediction[0]  # The model predicts the index of the disease name
    
    # Display prediction result
    st.write(f"Predicted Disease Name: {disease_name}")
    st.write("Disease outbreak likelihood:", "High" if prediction[0] == 1 else "Low")
