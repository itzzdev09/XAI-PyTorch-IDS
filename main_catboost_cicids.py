# main_catboost_cicids.py
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

PER_CLASS_MAX = 500_000
N_SPLITS = 5
RANDOM_STATE = 42

CB_PARAMS_GPU = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 10000,
    "depth": 8,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "random_seed": RANDOM_STATE,
    "task_type": "GPU",
    "devices": "0",
    "auto_class_weights": "SqrtBalanced",
    "rsm": 0.8,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "gpu_ram_part": 0.85,
    "verbose": 100,
    "use_best_model": False
}

CB_PARAMS_CPU = {**{k: v for k, v in CB_PARAMS_GPU.items() if k not in ["task_type", "devices", "gpu_ram_part"]},
                 "task_type": "CPU", "thread_count": os.cpu_count()}

CHUNK_ITERS = 1000

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
    print(f"✅ Dataset shape: {df.shape}")
    if "LabelMapped" not in df.columns:
        raise ValueError("❌ 'LabelMapped' column missing.")
    print("✅ Class distribution:\n", df["LabelMapped"].value_counts())
    return df

def infer_and_fix_types(df):
    cat_cols, num_cols = [], []
    for c in df.columns:
        if c.lower() in ["label", "labelmapped"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
            num_cols.append(c)
        else:
            df[c] = df[c].astype("category")
            cat_cols.append(c)
    print(f"✅ Categorical features: {cat_cols}")
    print(f"✅ Numerical features: {num_cols}")
    return df, cat_cols

def stratified_downsample(df, per_class_max=PER_CLASS_MAX):
    groups = []
    for label, grp in df.groupby("LabelMapped"):
        if len(grp) > per_class_max:
            grp = resample(grp, replace=False, n_samples=per_class_max, random_state=RANDOM_STATE)
        groups.append(grp)
    out = pd.concat(groups).drop_duplicates().sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"✅ After downsample: {out.shape}, class counts:\n{out['LabelMapped'].value_counts()}")
    return out

def train_in_chunks(params, train_pool, eval_pool, total_iters, chunk_iters, train_dir):
    os.makedirs(train_dir, exist_ok=True)
    done, model = 0, None
    t0 = time.time()
    while done < total_iters:
        add = min(chunk_iters, total_iters - done)
        this_params = {**params, "iterations": add, "train_dir": train_dir}
        m = CatBoostClassifier(**this_params)
        m.fit(train_pool, eval_set=eval_pool, init_model=model, verbose=100)
        done += add
        model = m
    return model

def compute_shap(model, X_sample, cat_indices, run_dir):
    pool = Pool(X_sample, cat_features=cat_indices)
    shap_values = model.get_feature_importance(data=pool, type="ShapValues")
    shap_values = shap_values[:, 1:]
    mean_abs_shap = np.abs(shap_values).mean(axis=0).ravel()
    mean_abs_shap = mean_abs_shap[:X_sample.shape[1]]
    shap_df = pd.DataFrame({
        "feature": list(X_sample.columns),
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(run_dir, "shap_values.csv"), index=False)
    plt.figure(figsize=(10, 6))
    shap_df.head(20).plot(kind="bar", x="feature", y="mean_abs_shap", legend=False)
    plt.title("Top-20 Features by SHAP")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "shap_summary.png"))
    plt.close()

def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"📂 Run folder: {run_dir}\n")

    df = load_dataset(DATA_PATH)
    df, cat_cols = infer_and_fix_types(df)
    df_small = stratified_downsample(df)

    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Flow_ID", "Timestamp", "Label", "LabelMapped", "Fwd_Header_Length.1"]
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

        train_dir = os.path.join(run_dir, f"fold_{fold}_train_logs")
        try:
            model = train_in_chunks(CB_PARAMS_GPU, train_pool, eval_pool, CB_PARAMS_GPU["iterations"], CHUNK_ITERS, train_dir)
        except CatBoostError:
            model = train_in_chunks(CB_PARAMS_CPU, train_pool, eval_pool, CB_PARAMS_CPU["iterations"], CHUNK_ITERS, train_dir)

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
            X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
            compute_shap(model, X_sample, cat_indices, run_dir)

    with open(os.path.join(run_dir, "metrics_kfold.json"), "w") as f:
        json.dump(fold_metrics, f, indent=2)

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
