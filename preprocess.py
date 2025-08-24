# preprocess.py
import os
import pandas as pd

RAW_DIR = "data/simargl2022"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

LABEL_MAPPING = {
    "normal": 0,
    "dos": 1,
    "malware": 2,
    "portscanning": 3
}

def preprocess():
    print("📂 Loading raw data...")
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found in raw data folder!")
        return

    for file in csv_files:
        file_path = os.path.join(RAW_DIR, file)
        df = pd.read_csv(file_path)

        # Determine label from filename if Label column not present
        if "Label" not in df.columns:
            for key in LABEL_MAPPING.keys():
                if key in file.lower():
                    df["Label"] = LABEL_MAPPING[key]
                    break
            else:
                print(f"⚠️ Skipping {file}: cannot determine label")
                continue

        # Save per-file parquet
        out_file = os.path.join(PROCESSED_DIR, f"processed_{file.split('.')[0]}.parquet")
        df.to_parquet(out_file, index=False)
        print(f"✅ {file} → {out_file} (Label={df['Label'].nunique()} classes)")

    # Save label mapping for reference
    import json
    with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
        json.dump(LABEL_MAPPING, f)
    print("📌 Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
