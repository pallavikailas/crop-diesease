import streamlit as st
import pandas as pd
import joblib
from dataloader import load_and_preprocess_data  
from visualisation import plot_correlation_matrix

# Load the trained ensemble model
model = joblib.load("ensemble_model.pkl")

# Title and description for the app
st.title("Crop Disease Outbreak Prediction")
st.write("Predict the likelihood of a crop disease outbreak based on weather conditions.")

# User inputs for prediction
temperature = st.slider("Temperature (°C)", min_value=10, max_value=40, step=1)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)
precipitation = st.slider("Precipitation (mm)", min_value=0, max_value=50, step=1)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, step=1)

# Show the correlation matrix as a visualization option
if st.checkbox('Show Correlation Matrix'):
    data = pd.DataFrame({'temperature': [temperature], 'humidity': [humidity],
                         'precipitation': [precipitation], 'wind_speed': [wind_speed]})
    plot_correlation_matrix(data)

# Prediction button
if st.button("Predict"):
    # Prepare the user input data as a DataFrame
    input_data = pd.DataFrame([[temperature, humidity, precipitation, wind_speed]],
                              columns=["temperature", "humidity", "precipitation", "wind_speed"])
    
    # Preprocess the input data (use the preprocessing from your training dataset)
    processed_data, _ = load_and_preprocess_data(input_data)

    # Get the prediction from the model
    prediction = model.predict(processed_data)

    # Display prediction result
    st.write("Disease outbreak likelihood:", "High" if prediction[0] == 1 else "Low")
