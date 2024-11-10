import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(df=None, file_path=None):
    """
    Loads data either from a CSV file or directly from a DataFrame, cleans it by handling missing values,
    and encodes categorical columns.
    
    Args:
    - df (pd.DataFrame): The data to be processed (optional).
    - file_path (str): Path to the CSV file (optional).
    
    Returns:
    - df (pd.DataFrame): Preprocessed DataFrame.
    - label_encoders (dict): Dictionary of LabelEncoders used for encoding categorical columns.
    """
    
    df = pd.read_csv(file_path)
    df = df.dropna()
    
    # Encode categorical columns
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders
