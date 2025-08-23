import os
import glob
import pandas as pd

RAW_DATA_DIR = "Data/simargl2022"

def inspect_all_datasets():
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError("❌ No CSV files found in Data/simargl2022")

    print(f"📂 Found {len(files)} CSV files in {RAW_DATA_DIR}\n")

    for file in files:
        try:
            df = pd.read_csv(file, nrows=5)  # only peek at first 5 rows
            cols = df.columns.tolist()

            # Check if dataset has any candidate label column
            possible_labels = ["Label", "label", "class", "Class", "attack_type", "malware"]
            label_cols = [c for c in possible_labels if c in cols]

            print(f"📑 {os.path.basename(file)}")
            print(f"   ➡ Columns: {cols[:10]}{' ...' if len(cols) > 10 else ''}")  # print first 10 cols only
            if label_cols:
                for col in label_cols:
                    print(f"   ✅ Found label column: '{col}' → unique values: {df[col].unique().tolist()}")
            else:
                print("   ⚠️ No label column found.")
            print("-" * 80)

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

if __name__ == "__main__":
    inspect_all_datasets()
