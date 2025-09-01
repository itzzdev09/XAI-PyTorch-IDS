# main_catboost_fast.py

import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

from catboost import CatBoostClassifier, Pool, CatBoostError
import shap

# --------------------
# Config
# --------------------
DATA_PATH = "data/processed/cicids_full.parquet"
RUNS_DIR = "Runs"
os.makedirs(RUNS_DIR, exist_ok=True)

PER_CLASS_MAX = 20_000         # 🔁 Downsample each class to 20k
MAX_TOTAL_ROWS = 100_000       # 🔁 Max total data after downsampling
N_SPLITS = 3                   # Fewer folds = faster

RANDOM_STATE = 42
CHUNK_ITERS = 100              # Iterations per chunk

CB_PARAMS_GPU = {
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "iterations": 300,                  # 🔁 Fast!
    "depth": 5,                         # 🔁 Shallower trees
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
# Helpers
# --------------------
def start_run_dir():
    d = os.path.join(RUNS_DIR, f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at {path}")
    print(f"📂 Loading dataset: {path}")
    df = pd.read_parquet(path)
    if "LabelMapped" not in df.columns:
        raise ValueError("❌ 'LabelMapped' column missing.")
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

def stratified_downsample(df, per_class_max=PER_CLASS_MAX, max_total_rows=MAX_TOTAL_ROWS):
    groups = []
    for label, grp in df.groupby("LabelMapped"):
        if len(grp) > per_class_max:
            grp = resample(grp, replace=False, n_samples=per_class_max, random_state=RANDOM_STATE)
        groups.append(grp)
    out = pd.concat(groups).drop_duplicates().sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    if max_total_rows and len(out) > max_total_rows:
        out = out.sample(n=max_total_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"✅ Downsampled to: {out.shape}, Class distribution:\n{out['LabelMapped'].value_counts()}")
    return out

def train_model(params, train_pool, eval_pool, train_dir):
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=eval_pool, verbose=params.get("verbose", 100))
    return model

def compute_shap(model, X_sample, cat_indices, run_dir):
    pool = Pool(X_sample, cat_features=cat_indices)
    shap_values = model.get_feature_importance(data=pool, type="ShapValues")
    shap_values = shap_values[:, 1:]
    mean_abs_shap = np.abs(shap_values).mean(axis=0).ravel()
    shap_df = pd.DataFrame({
        "feature": list(X_sample.columns),
        "mean_abs_shap": mean_abs_shap[:X_sample.shape[1]]
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(run_dir, "shap_values.csv"), index=False)
    plt.figure(figsize=(10, 5))
    shap_df.head(15).plot(kind="bar", x="feature", y="mean_abs_shap", legend=False)
    plt.title("Top-15 Features by SHAP")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "shap_summary.png"))
    plt.close()

# --------------------
# Main
# --------------------
def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"📁 Run directory: {run_dir}")

    df = load_dataset(DATA_PATH)
    df, cat_cols = infer_and_fix_types(df)
    df_small = stratified_downsample(df, PER_CLASS_MAX, MAX_TOTAL_ROWS)

    drop_cols = ["Flow_ID", "Timestamp", "Label", "LabelMapped", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    X = df_small.drop(columns=[c for c in drop_cols if c in df_small.columns])
    y = df_small["LabelMapped"].values
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    with open(os.path.join(run_dir, "features.json"), "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔹 Fold {fold}/{N_SPLITS}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        eval_pool = Pool(X_test, y_test, cat_features=cat_indices)

        try:
            model = train_model(CB_PARAMS_GPU, train_pool, eval_pool, run_dir)
        except CatBoostError:
            print("⚠️ GPU failed, falling back to CPU.")
            model = train_model(CB_PARAMS_CPU, train_pool, eval_pool, run_dir)

        model.save_model(os.path.join(run_dir, f"model_fold_{fold}.cbm"))
        y_pred = model.predict(X_test).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)
        fold_metrics.append(report)

        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"confusion_matrix_fold_{fold}.png"), dpi=150)
        plt.close()

        if fold == 1:
            X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
            compute_shap(model, X_sample, cat_indices, run_dir)

    with open(os.path.join(run_dir, "metrics_kfold.json"), "w") as f:
        json.dump(fold_metrics, f, indent=2)

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
