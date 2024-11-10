import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    """
    Loads data from a CSV file, cleans it by handling missing values,
    and encodes categorical columns.
    
    Args:
    - file_path (str): Path to the CSV file.
    
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
