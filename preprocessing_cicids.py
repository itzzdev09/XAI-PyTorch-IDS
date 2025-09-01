import os
import pandas as pd
import json

RAW_DIR = "data/cicids"            # Folder with raw CSVs
PROCESSED_DIR = "data/processed"   # Where output will go
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess():
    print("📂 Loading CICIDS raw data...")
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found in raw data folder!")
        return

    all_dfs = []

    for file in csv_files:
        file_path = os.path.join(RAW_DIR, file)
        try:
            # Try utf-8, fallback to ISO-8859-1
            try:
                df = pd.read_csv(file_path, engine="python")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, engine="python", encoding="ISO-8859-1")

            # ✅ Strip all column names
            df.columns = [col.strip() for col in df.columns]

            if "Label" not in df.columns:
                print(f"⚠️ Skipping {file}: 'Label' column not found even after stripping.")
                continue

            df["Label"] = df["Label"].astype(str).str.strip()

            # ✅ Create numeric label column
            label_mapping = {label: idx for idx, label in enumerate(df["Label"].unique())}
            df["LabelMapped"] = df["Label"].map(label_mapping)

            # Fill missing values
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna("missing").astype(str)
                else:
                    df[col] = df[col].fillna(0)

            # Save parquet per file
            out_file = os.path.join(PROCESSED_DIR, f"processed_{file.split('.')[0]}.parquet")
            df.to_parquet(out_file, index=False)
            print(f"✅ {file} → {out_file} ({len(label_mapping)} classes)")

            all_dfs.append(df)

        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue

    # Combine all
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)

        # Ensure string columns are valid
        for col in full_df.select_dtypes(include=["object"]).columns:
            full_df[col] = full_df[col].fillna("missing").astype(str)

        combined_file = os.path.join(PROCESSED_DIR, "cicids_full.parquet")
        full_df.to_parquet(combined_file, index=False)
        print(f"✅ Combined dataset saved → {combined_file} ({full_df['LabelMapped'].nunique()} classes)")

        # Save label mapping
        label_map = {label: int(mapped) for label, mapped in zip(full_df["Label"], full_df["LabelMapped"])}
        with open(os.path.join(PROCESSED_DIR, "label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)

    print("📌 Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
