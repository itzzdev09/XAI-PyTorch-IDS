import os
import pandas as pd
from utils.preprocessing import clean_data

RAW_DIR = "data/simargl2022"
PROCESSED_DIR = "data/processed"

def preprocess():
    print(f"🚀 Reading raw CSVs from {RAW_DIR}...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for file in os.listdir(RAW_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(RAW_DIR, file)
            print(f"📂 Processing {file_path} ...")
            
            df = pd.read_csv(file_path)

            # Clean
            df = clean_data(df)

            # Save
            out_path = os.path.join(PROCESSED_DIR, f"processed_{file}")
            df.to_csv(out_path, index=False)
            print(f"💾 Saved cleaned file to {out_path}")

if __name__ == "__main__":
    preprocess()
