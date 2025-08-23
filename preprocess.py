import os
import pandas as pd
from glob import glob

def preprocess():
    print("📂 Loading raw data...")

    files = glob("Data/simargl2022/*.csv")
    if not files:
        raise FileNotFoundError("❌ No CSV files found in Data/simargl2022")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'Label' not in df.columns:
                raise ValueError(f"❌ No 'Label' column in {f}")
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {f}: {e}")

    full_df = pd.concat(dfs, ignore_index=True)

    # Encode labels numerically (0,1,2,...)
    attack_names = full_df['Label'].unique().tolist()
    label_map = {name: i for i, name in enumerate(attack_names)}
    full_df['Label'] = full_df['Label'].map(label_map)

    os.makedirs("Data", exist_ok=True)

    # Save both formats
    csv_path = "Data/processed_data.csv"
    parquet_path = "Data/processed_data.parquet"

    full_df.to_csv(csv_path, index=False)
    full_df.to_parquet(parquet_path, engine="pyarrow", index=False)

    print(f"✅ Saved processed data to {csv_path} and {parquet_path}")
    print("📊 Attack mapping:", label_map)

if __name__ == "__main__":
    preprocess()
