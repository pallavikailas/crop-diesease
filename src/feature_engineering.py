def add_features(df):
    df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1)
    df["precip_wind_product"] = df["precipitation"] * df["wind_speed"]
    return df
