# main_rf.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter errors

import os
import warnings
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

import shap
import joblib

from models.rf_model import get_rf_model

# --------------------
# Config
# --------------------
DATA_PATH = "data/processed/simargl_full.parquet"
RUNS_DIR = "Runs_RF"
os.makedirs(RUNS_DIR, exist_ok=True)

N_SPLITS = 5
DOWNSAMPLE_RATIO = 0.02  # 2% downsample per fold, adjust to control speed/dataset size
RANDOM_STATE = 42

DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ALERT"]  # columns to drop

# --------------------
# Helper Functions
# --------------------

def start_run_dir():
    d = os.path.join(RUNS_DIR, f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_parquet(path)
    print(f"✅ Loaded dataset: {df.shape}")
    return df

def stratified_downsample(df, target_col, frac, random_state=42):
    # Stratified downsample: sample frac of each class separately and concat
    dfs = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        sample_size = max(int(len(cls_df)*frac), 1)
        dfs.append(cls_df.sample(sample_size, random_state=random_state))
    df_down = pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"✅ Downsampled dataset: {df_down.shape}")
    return df_down

def encode_categoricals(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype).startswith("category"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"🔤 Encoded column: {col}")
    return df

def compute_shap(model, X_sample, run_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "shap_summary_rf.png"))
    plt.close()
    print("✅ SHAP summary plot saved.")

def plot_confusion_matrix(cm, run_dir, fold):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix Fold {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(run_dir, f"confusion_matrix_fold_{fold}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {cm_path}")

def save_metrics(metrics, run_dir, fold):
    path = os.path.join(run_dir, f"metrics_fold_{fold}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved for Fold {fold}")

# --------------------
# Main
# --------------------
def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"📂 Run folder: {run_dir}\n")

    df = load_data(DATA_PATH)

    # Prepare features and target
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X_all = df.drop(columns=["Label"])
    y_all = df["Label"]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_idx = 1
    for train_idx, test_idx in skf.split(X_all, y_all):
        print(f"\n🔁 Fold {fold_idx}/{N_SPLITS}")

        # Split full data into train/test for this fold
        X_train_full, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train_full, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

        # Combine for downsampling train only (stratified)
        train_df = X_train_full.copy()
        train_df["Label"] = y_train_full

        train_df_down = stratified_downsample(train_df, "Label", frac=DOWNSAMPLE_RATIO, random_state=RANDOM_STATE)

        y_train = train_df_down["Label"].values
        X_train = train_df_down.drop(columns=["Label"])
        X_test_fold = X_test.copy()
        y_test_fold = y_test.values

        # Encode categoricals in train and test (fit only on train)
        X_train = encode_categoricals(X_train)
        # For test, encode categorical columns using same LabelEncoders fitted on train
        # Since encode_categoricals uses fresh LabelEncoders, better to combine train+test for encoding:
        # To keep it simple, encode test with train LabelEncoders column-wise:
        for col in X_train.columns:
            if X_train[col].dtype == 'int32' or X_train[col].dtype == 'int64':
                continue  # encoded already
            if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith("category"):
                # Shouldn't occur since we encoded train, but just in case
                le = LabelEncoder()
                combined = pd.concat([X_train[col], X_test_fold[col].astype(str)], axis=0)
                le.fit(combined)
                X_train[col] = le.transform(X_train[col].astype(str))
                X_test_fold[col] = le.transform(X_test_fold[col].astype(str))

        # In case test has any categorical columns, encode similarly:
        X_test = X_test_fold.copy()
        for col in X_train.columns:
            if X_train[col].dtype == 'int32' or X_train[col].dtype == 'int64':
                continue
            if X_train[col].dtype == 'object' or str(X_train[col].dtype).startswith("category"):
                # Re-encode with train encoder again just to be safe
                le = LabelEncoder()
                combined = pd.concat([X_train[col], X_test[col].astype(str)], axis=0)
                le.fit(combined)
                X_test[col] = le.transform(X_test[col].astype(str))

        print("🚀 Training Random Forest...")
        model = get_rf_model()
        model.fit(X_train, y_train)

        print("✅ Training complete. Evaluating...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test_fold, y_pred),
            "precision": precision_score(y_test_fold, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test_fold, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test_fold, y_pred, average='weighted', zero_division=0),
            "roc_auc": roc_auc_score(y_test_fold, y_proba, multi_class='ovr') if y_proba is not None else None,
            "classification_report": classification_report(y_test_fold, y_pred, zero_division=0)
        }

        save_metrics(metrics, run_dir, fold_idx)

        cm = confusion_matrix(y_test_fold, y_pred)
        fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        plot_confusion_matrix(cm, fold_dir, fold_idx)

        model_path = os.path.join(fold_dir, "rf_model_fold.joblib")
        joblib.dump(model, model_path)
        print(f"💾 Model saved → {model_path}")

        # SHAP summary only for fold 1 (to save time)
        if fold_idx == 1:
            print("📊 Computing SHAP values...")
            sample_size = min(1000, len(X_test))
            sample_X = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
            compute_shap(model, sample_X, fold_dir)

        fold_idx += 1

    print("\n🎉 All folds completed!")

if __name__ == "__main__":
    main()
