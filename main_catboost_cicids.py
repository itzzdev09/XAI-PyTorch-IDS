import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
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

PER_CLASS_MAX = 500_000
MAX_TOTAL_ROWS = 1000_000
TRAIN_RATIO = 0.5  # Use 50% for training + validation (with KFold), rest 50% for final test

RANDOM_STATE = 42
N_FOLDS = 10

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
CB_PARAMS_CPU = {**{k: v for k, v in CB_PARAMS_GPU.items() if k not in ["task_type", "devices"]},
                 "task_type": "CPU", "thread_count": os.cpu_count()}

# --------------------
def start_run_dir():
    d = os.path.join(RUNS_DIR, f"KFold_Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
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

def fix_categorical_nans(df, cat_cols):
    """Replace NaNs in categorical columns with 'NaN' string (CatBoost requirement)"""
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].cat.add_categories("NaN").fillna("NaN")
    return df

def main():
    warnings.filterwarnings("ignore")
    print("🚀 Starting CatBoost CICIDS pipeline...")

    run_dir = start_run_dir()
    print(f"📁 Output → {run_dir}")

    # Load full dataset
    df = load_dataset(DATA_PATH)
    df, cat_cols = infer_and_fix_types(df)

    # Split full dataset 50-50 (train+val vs final test)
    df_train_val, df_final_test = train_test_split(
        df, test_size=1-TRAIN_RATIO, stratify=df["LabelMapped"], random_state=RANDOM_STATE
    )
    print(f"✔ Reserved 50% for training/validation: {df_train_val.shape}, 50% for final testing: {df_final_test.shape}")

    drop_cols = ["Flow_ID", "Timestamp", "Label", "LabelMapped", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

    # Downsample training+validation set
    df_train_val_small = stratified_downsample(df_train_val)

    X_train_val = df_train_val_small.drop(columns=[c for c in drop_cols if c in df_train_val_small.columns])
    y_train_val = df_train_val_small["LabelMapped"].values
    cat_indices = [X_train_val.columns.get_loc(c) for c in cat_cols if c in X_train_val.columns]

    # Save features for reference
    with open(os.path.join(run_dir, "features.json"), "w") as f:
        json.dump(X_train_val.columns.tolist(), f, indent=2)

    print(f"🔁 Running {N_FOLDS}-Fold Cross-validation on training set...")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = []
    fold_num = 1
    models = []

    for train_index, val_index in skf.split(X_train_val, y_train_val):
        print(f"➡️ Fold {fold_num}/{N_FOLDS} — Training on {len(train_index)} samples, Validating on {len(val_index)} samples")

        X_tr, X_val = X_train_val.iloc[train_index].copy(), X_train_val.iloc[val_index].copy()
        y_tr, y_val = y_train_val[train_index], y_train_val[val_index]

        # Combine X_tr and y_tr for downsampling
        df_fold_tr = pd.concat([X_tr, pd.Series(y_tr, name="LabelMapped")], axis=1)
        df_fold_tr_small = stratified_downsample(df_fold_tr)

        X_tr_small = df_fold_tr_small.drop(columns=["LabelMapped"])
        y_tr_small = df_fold_tr_small["LabelMapped"].values

        # Fix NaNs in categorical columns (CatBoost requirement)
        X_tr_small = fix_categorical_nans(X_tr_small, cat_cols)
        X_val = fix_categorical_nans(X_val, cat_cols)

        train_pool = Pool(X_tr_small, y_tr_small, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        try:
            model = CatBoostClassifier(**CB_PARAMS_GPU)
            model.fit(train_pool, eval_set=val_pool)
        except CatBoostError:
            print("GPU failed — fallback to CPU")
            model = CatBoostClassifier(**CB_PARAMS_CPU)
            model.fit(train_pool, eval_set=val_pool)

        models.append(model)

        # Predict on validation fold
        y_val_pred = model.predict(X_val).astype(int)

        # Save classification report and confusion matrix per fold
        report = classification_report(y_val, y_val_pred, output_dict=True)
        fold_metrics.append(report)

        # Confusion matrix plot
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold_num}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"confusion_matrix_fold_{fold_num}.png"))
        plt.close()

        # Save classification report json
        with open(os.path.join(run_dir, f"classification_report_fold_{fold_num}.json"), "w") as f:
            json.dump(report, f, indent=2)

        fold_num += 1

    # Aggregate fold metrics (average accuracy as example)
    avg_accuracy = np.mean([m["accuracy"] for m in fold_metrics])
    print(f"Average CV accuracy: {avg_accuracy:.4f}")

    # Now train final model on full downsampled train+val set (X_train_val, y_train_val)
    print("▶️ Training final model on full downsampled training set...")

    # Fix NaNs on full train_val set
    X_train_val = fix_categorical_nans(X_train_val, cat_cols)

    full_train_pool = Pool(X_train_val, y_train_val, cat_features=cat_indices)

    try:
        final_model = CatBoostClassifier(**CB_PARAMS_GPU)
        final_model.fit(full_train_pool)
    except CatBoostError:
        print("GPU failed — fallback to CPU")
        final_model = CatBoostClassifier(**CB_PARAMS_CPU)
        final_model.fit(full_train_pool)

    final_model.save_model(os.path.join(run_dir, "final_model.cbm"))

    # Prepare final test set
    df_final_test = df_final_test.reset_index(drop=True)
    X_test = df_final_test.drop(columns=[c for c in drop_cols if c in df_final_test.columns])
    y_test = df_final_test["LabelMapped"].values

    # Fix NaNs in final test set
    X_test = fix_categorical_nans(X_test, cat_cols)

    y_test_pred = final_model.predict(X_test).astype(int)
    y_test_proba = final_model.predict_proba(X_test)

    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    with open(os.path.join(run_dir, "classification_report_final_test.json"), "w") as f:
        json.dump(report_test, f, indent=2)

    print("Final test classification report saved.")

    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_test, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix - Final Test")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix_final_test.png"))
    plt.close()

    # Save probabilities and true labels for detailed analysis on final test
    proba_df = pd.DataFrame(y_test_proba, columns=[f"class_{i}" for i in range(y_test_proba.shape[1])])
    proba_df["true_label"] = y_test
    proba_df.to_csv(os.path.join(run_dir, "final_test_probabilities.csv"), index=False)

    # Compute SHAP values on a sample of training data for explainability
    sample_for_shap = X_train_val.sample(n=1000, random_state=RANDOM_STATE)
    print("Computing SHAP values on training sample...")
    compute_shap(final_model, sample_for_shap, cat_indices, run_dir)

    print(f"✅ Run complete. Results saved to: {run_dir}")

if __name__ == "__main__":
    main()
