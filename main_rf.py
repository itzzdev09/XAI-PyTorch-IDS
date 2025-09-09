import os
import json
import time
import joblib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import shap

from models.rf_model import get_rf_model

# -------------------
# CONFIG
# -------------------
DATA_PATH = "data/processed/simargl_full.parquet"
RUNS_DIR = "Runs_RF"
os.makedirs(RUNS_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
PER_CLASS_MAX = 100_000  # Cap each class to this many samples

DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ALERT"]

# -------------------
# UTILS
# -------------------
def start_run_dir():
    run_dir = os.path.join(RUNS_DIR, f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def stratified_downsample(df, per_class_max):
    print(f"✅ Loaded dataset: {df.shape}")
    grouped = []
    for label, group in df.groupby("Label"):
        if len(group) > per_class_max:
            group = resample(group, replace=False, n_samples=per_class_max, random_state=RANDOM_STATE)
        grouped.append(group)
    df_down = pd.concat(grouped).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
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

def plot_confusion_matrix(cm, run_dir, fold):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    path = os.path.join(run_dir, f"fold_{fold}", f"confusion_matrix_fold_{fold}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {path}")

def compute_shap(model, X_sample, run_dir, fold):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    os.makedirs(os.path.join(run_dir, f"fold_{fold}"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"fold_{fold}", "shap_summary_rf.png"))
    plt.close()
    print("✅ SHAP summary plot saved.")

# -------------------
# MAIN
# -------------------
def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"📂 Run folder: {run_dir}\n")

    df = pd.read_parquet(DATA_PATH)
    metrics_all_folds = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["Label"]), 1):
        print(f"\n🔁 Fold {fold}/{N_SPLITS}")

        df_fold = df.iloc[train_idx].copy()
        df_fold = stratified_downsample(df_fold, PER_CLASS_MAX)

        X = df_fold.drop(columns=["Label"] + [c for c in DROP_COLS if c in df_fold.columns])
        y = df_fold["Label"]

        X = encode_categoricals(X)
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)

        # Split for validation within fold
        val_size = 0.2
        val_count = int(len(X) * val_size)
        X_train, X_test = X[val_count:], X[:val_count]
        y_train, y_test = y_encoded[val_count:], y_encoded[:val_count]

        print("🚀 Training Random Forest...")
        model = get_rf_model()
        model.fit(X_train, y_train)
        print("✅ Training complete. Evaluating...")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        fold_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        }

        metrics_all_folds.append(fold_metrics)

        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(fold_metrics, f, indent=4)
        print(f"✅ Metrics saved for Fold {fold}")

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, run_dir, fold)

        # Save model
        joblib.dump(model, os.path.join(fold_dir, f"rf_model_fold.joblib"))
        print(f"💾 Model saved → {os.path.join(fold_dir, f'rf_model_fold.joblib')}")

        # SHAP only for first fold
        if fold == 1:
            sample_X = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
            compute_shap(model, sample_X, run_dir, fold)

    # -------------------
    # AGGREGATE METRICS
    # -------------------
    print("\n📝 Cross-validation results:")
    agg_metrics = {
        k: round(np.mean([m[k] for m in metrics_all_folds]), 4)
        for k in metrics_all_folds[0].keys()
    }

    for k, v in agg_metrics.items():
        print(f"{k}: {v}")

    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as f:
        json.dump(agg_metrics, f, indent=4)

if __name__ == "__main__":
    main()
