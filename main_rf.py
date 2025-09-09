# main_rf.py

import os
import time
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import shap
import joblib

from models.rf_model import get_rf_model

# --------------------
# Config
# --------------------
DATA_PATH = "data/processed/simargl_full.parquet"
RUNS_DIR = "Runs_RF"
os.makedirs(RUNS_DIR, exist_ok=True)

PER_CLASS_MAX = 200_000  # Max samples per class
N_SPLITS = 5             # K-Fold cross-validation
RANDOM_STATE = 42

# --------------------
# Helpers
# --------------------
def start_run_dir():
    d = os.path.join(RUNS_DIR, f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_parquet(path)
    print(f"✅ Loaded dataset: {df.shape}")
    return df

def stratified_downsample(df, per_class_max):
    groups = []
    for label, grp in df.groupby("Label"):
        if len(grp) > per_class_max:
            grp = resample(grp, replace=False, n_samples=per_class_max, random_state=RANDOM_STATE)
        groups.append(grp)
    df_down = pd.concat(groups, axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
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

def plot_confusion_matrix(cm, run_dir, fold=None):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix" + (f" Fold {fold}" if fold else ""))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(run_dir, f"confusion_matrix_fold_{fold}.png" if fold else "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {cm_path}")

def compute_shap(model, X_sample, run_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "shap_summary_rf.png"))
    plt.close()
    print("✅ SHAP summary plot saved.")

# --------------------
# Main
# --------------------
def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"📂 Run folder: {run_dir}")

    df = load_dataset(DATA_PATH)

    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ALERT"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = stratified_downsample(df, per_class_max=PER_CLASS_MAX)

    X = df.drop(columns=["Label"])
    y = df["Label"].values

    X = encode_categoricals(X)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔁 Fold {fold}/{N_SPLITS}")
        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print("🚀 Training Random Forest...")
        model = get_rf_model()
        model.fit(X_train, y_train)

        print("✅ Training complete. Evaluating...")
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        try:
            y_prob = model.predict_proba(X_test)
            if len(np.unique(y)) > 2:
                metrics["roc_auc_ovr_macro"] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                metrics["roc_auc_ovo_macro"] = roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
            else:
                metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
        except Exception as e:
            print(f"⚠️ ROC AUC computation failed: {e}")

        all_metrics = {"classification_report": report, "metrics": metrics}
        fold_metrics.append(all_metrics)

        # Save metrics
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"✅ Metrics saved for Fold {fold}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, fold_dir, fold=fold)

        # Save model
        model_path = os.path.join(fold_dir, "rf_model_fold.joblib")
        joblib.dump(model, model_path)
        print(f"💾 Model saved → {model_path}")

        # SHAP only for first fold
        if fold == 1:
            sample_X = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
            compute_shap(model, sample_X, fold_dir)

    # Save summary of all fold metrics
    with open(os.path.join(run_dir, "all_folds_metrics.json"), "w") as f:
        json.dump(fold_metrics, f, indent=4)
    print(f"\n📊 All folds metrics saved → {os.path.join(run_dir, 'all_folds_metrics.json')}")

if __name__ == "__main__":
    main()
