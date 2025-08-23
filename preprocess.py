import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Paths
SIMARGL_PATH = "Data/simargl2022"
PROCESSED_PATH = "Data/processed/processed_data.csv"

def preprocess_data():
    print("📂 Searching for SIMARGL dataset...")

    # find first CSV in simargl2022
    files = [f for f in os.listdir(SIMARGL_PATH) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("❌ No CSV file found inside Data/simargl2022/")
    
    file_path = os.path.join(SIMARGL_PATH, files[0])
    print(f"✅ Found dataset: {file_path}")

    # load data
    df = pd.read_csv(file_path)
    print(f"📊 Raw data shape: {df.shape}")

    # separate categorical & numerical
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = df.select_dtypes(exclude=["object"]).columns

    print(f"📝 Handling {len(categorical_cols)} categorical and {len(numeric_cols)} numeric columns...")

    # impute numeric with mean
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy="mean")
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

    # impute categorical with most_frequent
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

        # encode categorical with LabelEncoder
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # save processed data
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Preprocessing complete. Saved to {PROCESSED_PATH}")
    print(f"📊 Processed data shape: {df.shape}")

if __name__ == "__main__":
    preprocess_data()
