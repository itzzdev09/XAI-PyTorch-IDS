# preprocess.py
import os
import pandas as pd
import json

RAW_FOLDER = "Data/simargl2022"
PROCESSED_FOLDER = "Data/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

LABEL_MAP = {
    "normal": 0,
    "dos": 1,
    "malware": 2,
    "portscanning": 3
}

def preprocess_single(file_path):
    filename = os.path.basename(file_path)
    attack_type = None
    for key in LABEL_MAP:
        if key in filename.lower():
            attack_type = key
            break
    if attack_type is None:
        print(f"⚠️ Skipping {filename}: ❌ Cannot detect attack type")
        return None
    
    df = pd.read_csv(file_path)
    df["Label"] = LABEL_MAP[attack_type]
    print(f"✅ {filename} → Label {LABEL_MAP[attack_type]}")
    return df

def preprocess(single_file=None):
    if single_file:
        files = [os.path.join(RAW_FOLDER, single_file)]
    else:
        files = [os.path.join(RAW_FOLDER, f) for f in os.listdir(RAW_FOLDER) if f.endswith(".csv")]

    dfs = []
    for f in files:
        df = preprocess_single(f)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise ValueError("❌ No files to process!")

    full_df = pd.concat(dfs, ignore_index=True)
    output_csv = os.path.join(PROCESSED_FOLDER, "processed_data.csv")
    full_df.to_parquet(output_csv, index=False)  # Using parquet for faster I/O
    with open(os.path.join(PROCESSED_FOLDER, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f)
    print(f"✅ Preprocessing complete. Saved to {output_csv}")
    print(f"📊 Final shape: {full_df.shape}")

if __name__ == "__main__":
    # To preprocess all datasets
    preprocess()

    # OR preprocess single file
    # preprocess(single_file="dos-03-15-2022-15-44-32.csv")
