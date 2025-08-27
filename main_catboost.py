import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from catboost import CatBoostClassifier, Pool

from sklearn.metrics import classification_report, confusion_matrix

import shap

# ---------------------
# CONFIGURATION
# ---------------------
DATA_PATH = "data/processed/simargl_full.parquet"
RUNS_DIR = "Runs"
FOLD_TO_LOAD = 1  # change if you want to test with another fold's model

# ---------------------
# HELPERS
# ---------------------
def get_latest_run_folder(base_path):
    folders = [os.path.join(base_path, d) for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]
    if not folders:
        raise FileNotFoundError("❌ No run folders found in 'Runs/'")
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"📁 Using latest run folder: {latest_folder}")
    return latest_folder

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at {path}")
    print(f"📂 Loading dataset: {path}")
    df = pd.read_parquet(path)
    print(f"✅ Dataset shape: {df.shape}")
    if "Label" not in df.columns:
        raise ValueError("❌ 'Label' column missing in dataset.")
    print("✅ Class distribution:\n", df["Label"].value_counts())
    return df

def infer_and_fix_types(df):
    cat_cols = []
    for c in df.columns:
        if c.lower() in ["label", "target"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
        else:
            df[c] = df[c].astype("category")
            cat_cols.append(c)
    return df, cat_cols

def compute_shap(model, X_sample, cat_indices, out_dir):
    pool = Pool(X_sample, cat_features=cat_indices)
    shap_values = model.get_feature_importance(data=pool, type="ShapValues")
    shap_values = shap_values[:, 1:]  # drop bias term

    mean_abs_shap = np.abs(shap_values).mean(axis=0).ravel()
    mean_abs_shap = mean_abs_shap[:X_sample.shape[1]]

    shap_df = pd.DataFrame({
        "feature": list(X_sample.columns),
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(os.path.join(out_dir, "shap_values_test.csv"), index=False)

    plt.figure(figsize=(10, 6))
    shap_df.head(20).plot(kind="bar", x="feature", y="mean_abs_shap", legend=False)
    plt.title("Top-20 SHAP Features (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_test.png"))
    plt.close()
    print("📌 SHAP summary saved.")

# ---------------------
# MAIN
# ---------------------
def main():
    warnings.filterwarnings("ignore")

    run_dir = get_latest_run_folder(RUNS_DIR)
    model_path = os.path.join(run_dir, f"model_fold_{FOLD_TO_LOAD}.cbm")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    print(f"📦 Loading model from: {model_path}")
    model = CatBoostClassifier()
    model.load_model(model_path)

    df = load_dataset(DATA_PATH)
    df, cat_cols = infer_and_fix_types(df)

    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ALERT"]
    X = df.drop(columns=["Label"] + [c for c in drop_cols if c in df.columns])
    y = df["Label"].values
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    pool = Pool(X, cat_features=cat_indices)
    y_pred = model.predict(X).astype(int)

    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    print("📊 Classification Report:\n", classification_report(y, y_pred))
    report_path = os.path.join(run_dir, "classification_report_full.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📌 Report saved → {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Full Dataset)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(run_dir, "confusion_matrix_full.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"📌 Confusion matrix saved → {cm_path}")

    # Save predictions
    pred_df = pd.DataFrame({"Actual": y, "Predicted": y_pred.ravel()})
    pred_path = os.path.join(run_dir, "predictions_full.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"📌 Predictions saved → {pred_path}")

    # SHAP
    shap_n = min(2000, len(X))
    X_sample = X.sample(n=shap_n, random_state=42)
    compute_shap(model, X_sample, cat_indices, run_dir)

    print("\n✅ Testing complete.")

if __name__ == "__main__":
    main()
