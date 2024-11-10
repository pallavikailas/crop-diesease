import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop or fill missing values
    df = df.dropna()
    
    # Encode categorical columns
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders
