import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe:
    - Replace inf/-inf with NaN
    - Impute NaN values (mean for numeric, mode for categorical)
    - Encode categorical features with LabelEncoder
    - Scale numeric features
    """
    print("🧹 Cleaning data (mean/mode imputation + encoding)...")

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # --- Handle numeric columns ---
    for col in numeric_cols:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)

    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # --- Handle categorical columns ---
    for col in categorical_cols:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else "missing"
        df[col] = df[col].fillna(mode_val)

        # Apply LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print(f"✅ Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
