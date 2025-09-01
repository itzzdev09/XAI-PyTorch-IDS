import os
import pandas as pd
import numpy as np
import json
import warnings

from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool

# --------------------
DATA_PATH = "data/processed/cicids_full.parquet"
MODEL_PATH = "Runs_TrainTest/YOUR_RUN_FOLDER/model.cbm"  # Replace with actual folder
FEATURES_PATH = "Runs_TrainTest/YOUR_RUN_FOLDER/features.json"  # Replace as needed
USED_DATA_PATH = "Runs_TrainTest/YOUR_RUN_FOLDER/used_indices.csv"  # We will generate this below

RANDOM_STATE = 42
# --------------------

def load_dataset():
    df = pd.read_parquet(DATA_PATH)
    if "LabelMapped" not in df.columns:
        raise ValueError("'LabelMapped' column missing.")
    return df

def infer_and_fix_types(df):
    cat_cols = []
    for col in df.columns:
        if col.lower() in ["label", "labelmapped"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
        else:
            df[col] = df[col].astype("category")
            cat_cols.append(col)
    return df, cat_cols

def main():
    warnings.filterwarnings("ignore")

    print("Loading full dataset...")
    df = load_dataset()
    df, cat_cols = infer_and_fix_types(df)

    # Load features used during training
    with open(FEATURES_PATH, "r") as f:
        feature_list = json.load(f)

    # Drop unused
    drop_cols = ["Flow_ID", "Timestamp", "Label", "LabelMapped", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y_all = df["LabelMapped"].values

    # Assume the first 50% (stratified) was used — simulate by random split with saved seed
    from sklearn.model_selection import train_test_split
    _, X_remain, _, y_remain = train_test_split(
        X_all, y_all, test_size=0.5, stratify=y_all, random_state=RANDOM_STATE
    )

    X_remain = X_remain[feature_list]
    cat_indices = [X_remain.columns.get_loc(c) for c in cat_cols if c in X_remain.columns]

    print(f"Evaluating on holdout: {len(y_remain)} samples")

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    pool_remain = Pool(X_remain, y_remain, cat_features=cat_indices)
    y_pred = model.predict(pool_remain).astype(int)
    report = classification_report(y_remain, y_pred, output_dict=True)
    print("Classification Report on Unseen 50%:")
    print(json.dumps(report, indent=2))

    cm = confusion_matrix(y_remain, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
