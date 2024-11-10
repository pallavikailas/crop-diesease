def add_features(df):
    """
    Adds new features to the DataFrame to enhance model performance.
    
    Args:
    - df (pd.DataFrame): The DataFrame to add features to.
    
    Returns:
    - pd.DataFrame: The DataFrame with added features.
    """
    df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1)
    df["precip_wind_product"] = df["precipitation"] * df["wind_speed"]
    return df
