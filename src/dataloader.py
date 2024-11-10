import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path):
    # Load the dataset from the provided CSV file path
    df = pd.read_csv(file_path)
    
    # Initialize the LabelEncoder to convert categorical columns to numerical values
    label_encoder = LabelEncoder()
    df['Crop_Type'] = label_encoder.fit_transform(df['Crop_Type'])
    
    # Scale the numerical columns to standardize them
    scaler = StandardScaler()
    numerical_columns = ['Temperature', 'Humidity', 'Rainfall', 
                         'Soil_Moisture', 'Wind_Speed', 'Sunlight_Hours', 
                         'Soil_pH', 'Growth_Stage', 'Surrounding_Crop_Diversity']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Split the dataset into features (X) and the target variable (y)
    X = df.drop(columns=["Disease_Type"])  # Features: everything except "Disease Name"
    y = df["Disease_type"]  # Target: "Disease Name"
    
    return X, y
