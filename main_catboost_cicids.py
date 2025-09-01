import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

from catboost import CatBoostClassifier, Pool, CatBoostError
import shap

# --------------------
# Config
# --------------------
DATA_PATH = "data/processed/cicids_full.parquet"
RUNS_DIR = "Runs_TrainTest"
os.makedirs(RUNS_DIR, exist_ok=True)

PER_CLASS_MAX = 20_000
MAX_TOTAL_ROWS = 100_000
TEST_SIZE = 0.3  # 70-30 split

RANDOM_STATE = 42

CB_PARAMS_GPU = {
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "iterations": 300,
    "depth": 5,
    "learning_rate": 0.15,
    "l2_leaf_reg": 3.0,
    "random_seed": RANDOM_STATE,
    "task_type": "GPU",
    "devices": "0",
    "auto_class_weights": "Balanced",
    "verbose": 50,
    "use_best_model": False
}
CB_PARAMS_CPU = {**{k: v for k, v in CB_PARAMS_GPU.items() if k not in ["task_type", "devices"]},
                 "task_type": "CPU", "thread_count": os.cpu_count()}

# --------------------
def start_run_dir():
    d = os.path.join(RUNS_DIR, f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_parquet(path)
    if "LabelMapped" not in df.columns:
        raise ValueError("'LabelMapped' column is missing.")
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

def stratified_downsample(df):
    groups = []
    for label, grp in df.groupby("LabelMapped"):
        if len(grp) > PER_CLASS_MAX:
            grp = resample(grp, replace=False, n_samples=PER_CLASS_MAX, random_state=RANDOM_STATE)
        groups.append(grp)
    out = pd.concat(groups).drop_duplicates().sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    if MAX_TOTAL_ROWS and len(out) > MAX_TOTAL_ROWS:
        out = out.sample(n=MAX_TOTAL_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Downsampled to: {out.shape}, distribution:\n{out['LabelMapped'].value_counts()}")
    return out

def compute_shap(model, X_sample, cat_indices, run_dir):
    pool = Pool(X_sample, cat_features=cat_indices)
    shap_vals = model.get_feature_importance(data=pool, type="ShapValues")[:, 1:]
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(run_dir, "shap_values.csv"), index=False)

    plt.figure(figsize=(10, 5))
    shap_df.head(15).plot(kind="bar", x="feature", y="mean_abs_shap", legend=False)
    plt.title("Top‑15 Features by SHAP")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "shap_summary.png"))
    plt.close()

def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"Output → {run_dir}")

    df = load_dataset(DATA_PATH)
    df, cat_cols = infer_and_fix_types(df)
    df_small = stratified_downsample(df)

    drop_cols = ["Flow_ID", "Timestamp", "Label", "LabelMapped", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    X = df_small.drop(columns=[c for c in drop_cols if c in df_small.columns])
    y = df_small["LabelMapped"].values
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    with open(os.path.join(run_dir, "features.json"), "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    # ✅ 50-50 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training samples: {len(y_train)}, Testing: {len(y_test)}")
    print(f"Train label distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"Test label distribution:\n{pd.Series(y_test).value_counts()}")

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)

    try:
        model = CatBoostClassifier(**CB_PARAMS_GPU)
        model.fit(train_pool, eval_set=test_pool)
    except CatBoostError:
        print("GPU failed — fallback to CPU")
        model = CatBoostClassifier(**CB_PARAMS_CPU)
        model.fit(train_pool, eval_set=test_pool)

    model.save_model(os.path.join(run_dir, "model.cbm"))

    y_pred = model.predict(X_test).astype(int)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(run_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Classification report saved.")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
    plt.close()

    # Save probabilities and true labels for detailed analysis
    proba_df = pd.DataFrame(y_proba, columns=[f"class_{i}" for i in range(y_proba.shape[1])])
    proba_df["true"] = y_test
    proba_df["pred"] = y_pred.flatten()
    proba_df.to_csv(os.path.join(run_dir, "predictions_with_proba.csv"), index=False)

    # SHAP analysis on a sample of test set
    sample_n = min(1000, len(X_test))
    compute_shap(model, X_test.sample(n=sample_n, random_state=RANDOM_STATE), cat_indices, run_dir)

    print("✅ Done.")
