import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(df):
    # Preprocessing steps such as handling missing values, encoding, scaling, etc.
    df = df.dropna()
    # Implement more preprocessing steps as needed
    return df
