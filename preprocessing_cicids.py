import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
RAW_DIR = Path("data/cicids")          # Folder with raw CSVs
PROCESSED_DIR = Path("data/processed") # Where output will go
TEST_SIZE = 0.2                        # 20% of data for testing
RANDOM_STATE = 42                      # For reproducibility

# --- Main Preprocessing Function ---
def preprocess():
    """
    Loads, cleans, scales, and splits the CICIDS dataset into training and testing sets.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("🚀 Starting preprocessing pipeline...")

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in '{RAW_DIR}'. Exiting.")
        return

    # --- 1. Create a Global, Consistent Label Mapping ---
    print("🗺️ Scanning all files to create a consistent label map...")
    all_labels = set()
    for file in csv_files:
        try:
            # Read just the 'Label' column to save memory
            df_labels = pd.read_csv(file, usecols=[' Label'], engine='python', encoding='ISO-8859-1')
            df_labels.columns = [col.strip() for col in df_labels.columns]
            all_labels.update(df_labels["Label"].astype(str).str.strip().unique())
        except Exception as e:
            print(f"⚠️ Could not read labels from {file}: {e}")
            continue

    # Sort labels for consistent mapping across runs
    label_map = {label: idx for idx, label in enumerate(sorted(list(all_labels)))}
    print(f"✅ Found {len(label_map)} unique labels across all files.")

    # --- 2. Load and Clean Data ---
    all_dfs = []
    print("\n📂 Loading and cleaning individual CSV files...")
    for file in csv_files:
        print(f"   -> Processing {file.name}")
        try:
            # Read with fallback encoding
            try:
                df = pd.read_csv(file, engine="python")
            except UnicodeDecodeError:
                df = pd.read_csv(file, engine="python", encoding="ISO-8859-1")

            # Clean column names
            df.columns = [col.strip() for col in df.columns]

            if "Label" not in df.columns:
                print(f"   ⚠️ Skipping {file.name}: 'Label' column not found.")
                continue
            
            # Apply the global label map
            df["Label"] = df["Label"].astype(str).str.strip()
            df["LabelMapped"] = df["Label"].map(label_map)
            df = df.drop(columns=["Label"]) # Drop original string label

            all_dfs.append(df)

        except Exception as e:
            print(f"   ⚠️ Skipping {file.name}: An error occurred: {e}")
            continue

    if not all_dfs:
        print("❌ No data was loaded. Exiting.")
        return

    # --- 3. Combine, Handle Infinite Values and Impute ---
    print("\n🧩 Combining all dataframes...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"   -> Combined shape: {full_df.shape}")

    # Drop any potential duplicates
    full_df.drop_duplicates(inplace=True)
    print(f"   -> Shape after dropping duplicates: {full_df.shape}")

    # Handle infinite values which are common in network data
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Identify feature columns (everything except the label)
    feature_cols = [col for col in full_df.columns if col != 'LabelMapped']

    # Convert all feature columns to numeric, coercing errors
    for col in feature_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    
    # Impute missing values with the median (more robust to outliers than mean)
    # This is done before splitting, which is acceptable for simple imputation.
    # For more advanced techniques, imputation should be done post-split.
    imputation_values = full_df[feature_cols].median()
    full_df[feature_cols] = full_df[feature_cols].fillna(imputation_values)

    print("✅ Data cleaning and imputation complete.")

    # --- 4. Split into Training and Testing Sets ---
    print("\n🔪 Splitting data into training and testing sets...")
    X = full_df.drop("LabelMapped", axis=1)
    y = full_df["LabelMapped"]

    # Stratify ensures the class distribution is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   -> Training set size: {X_train.shape[0]} samples")
    print(f"   -> Testing set size:  {X_test.shape[0]} samples")

    # --- 5. Scale Numerical Features ---
    print("\n⚖️ Scaling numerical features...")
    scaler = StandardScaler()

    # Fit the scaler ONLY on the training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test data using the scaler fitted on the training data
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df["LabelMapped"] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["LabelMapped"] = y_test.values
    print("✅ Scaling complete.")

    # --- 6. Save Processed Data and Artifacts ---
    print("\n💾 Saving processed files...")
    # Save datasets
    train_file = PROCESSED_DIR / "train.parquet"
    test_file = PROCESSED_DIR / "test.parquet"
    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)
    print(f"   -> Saved training data to {train_file}")
    print(f"   -> Saved testing data to {test_file}")

    # Save label map
    map_file = PROCESSED_DIR / "label_map.json"
    with open(map_file, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"   -> Saved label map to {map_file}")
    
    # Save the fitted scaler
    scaler_file = PROCESSED_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_file)
    print(f"   -> Saved fitted scaler to {scaler_file}")
    
    # Save the imputation values
    impute_file = PROCESSED_DIR / "imputation_values.json"
    imputation_values.to_json(impute_file, indent=2)
    print(f"   -> Saved imputation values to {impute_file}")


    print("\n📌 Preprocessing complete. You are now ready to train your model! 🎉")


if __name__ == "__main__":
    preprocess()