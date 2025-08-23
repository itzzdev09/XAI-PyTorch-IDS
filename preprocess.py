import os
import glob
import pandas as pd
import json

RAW_DATA_DIR = "Data/simargl2022"
PROCESSED_PATH = "Data/processed/processed_data.csv"
LABEL_MAP_PATH = "Data/processed/label_map.json"

# Define label mapping
LABEL_MAP = {
    "normal": 0,
    "dos": 1,
    "malware": 2,
    "portscanning": 3
}

def assign_label_from_filename(filename: str) -> int:
    """
    Determine label from filename (e.g., dos-03-15-2022.csv -> 'dos').
    """
    fname = os.path.basename(filename).lower()
    for key in LABEL_MAP.keys():
        if key in fname:
            return LABEL_MAP[key]
    raise ValueError(f"❌ Could not assign label for file: {filename}")

def preprocess():
    os.makedirs("Data/processed", exist_ok=True)

    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError("❌ No CSV files found in Data/simargl2022")

    all_dfs = []
    print(f"📂 Found {len(files)} CSV files. Preprocessing...")

    for file in files:
        try:
            df = pd.read_csv(file)

            # Add Label column
            df["Label"] = assign_label_from_filename(file)

            print(f"   ✅ {os.path.basename(file)} → Label {df['Label'].iloc[0]}")
            all_dfs.append(df)

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    # Concatenate all
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Save processed data
    full_df.to_csv(PROCESSED_PATH, index=False)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(LABEL_MAP, f, indent=4)

    print(f"\n✅ Preprocessing complete.")
    print(f"📊 Final shape: {full_df.shape}")
    print(f"📌 Saved processed data → {PROCESSED_PATH}")
    print(f"📌 Saved label mapping → {LABEL_MAP_PATH}")

if __name__ == "__main__":
    preprocess()
