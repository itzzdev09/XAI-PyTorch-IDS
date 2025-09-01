# preprocessing_cicids.py
import os
import pandas as pd
import json

RAW_DIR = "data/cicids"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define label mapping according to CICIDS2017 or CICIDS2018 attacks
LABEL_MAPPING = {
    "BENIGN": 0,
    "DoS": 1,
    "DDoS": 2,
    "BruteForce": 3,
    "Bot": 4,
    "PortScan": 5,
    "WebAttack": 6,
    "Infiltration": 7,
    "FTP-Patator": 8,
    "SSH-Patator": 9,
    "Heartbleed": 10,
    # Add more if needed...
}

def detect_label_from_value(val):
    val = str(val).lower()
    for key in LABEL_MAPPING:
        if key.lower() in val:
            return LABEL_MAPPING[key]
    return None

def preprocess():
    print("📂 Loading CICIDS raw data...")
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found in CICIDS folder!")
        return

    all_dfs = []

    for file in csv_files:
        file_path = os.path.join(RAW_DIR, file)
        try:
            df = pd.read_csv(file_path, engine="python")
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue

        if "Label" not in df.columns:
            print(f"⚠️ Skipping {file}: No 'Label' column")
            continue

        # Map labels
        df["LabelMapped"] = df["Label"].apply(detect_label_from_value)
        df = df[df["LabelMapped"].notnull()]

        # Fill missing
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("missing").astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Save individual parquet
        out_file = os.path.join(PROCESSED_DIR, f"processed_{file.split('.')[0]}.parquet")
        df.to_parquet(out_file, index=False)
        print(f"✅ {file} → {out_file} (Classes: {df['LabelMapped'].nunique()})")

        all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        for col in full_df.select_dtypes(include=['object']).columns:
            full_df[col] = full_df[col].fillna("missing").astype(str)

        combined_file = os.path.join(PROCESSED_DIR, "cicids_full.parquet")
        full_df.to_parquet(combined_file, index=False)
        print(f"✅ Combined dataset saved → {combined_file} (LabelMapped={full_df['LabelMapped'].nunique()} classes)")

    # Save label map
    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
        json.dump(LABEL_MAPPING, f)
    print("📌 Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
