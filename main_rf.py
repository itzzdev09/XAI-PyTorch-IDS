# main_rf.py

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import shap
import joblib

from models.rf_model import get_rf_model


# --------------------
# Config
# --------------------
DATA_PATH = "data/processed/simargl_full.parquet"
RUNS_DIR = "Runs_RF"
os.makedirs(RUNS_DIR, exist_ok=True)

SAMPLE_FRAC = 0.05  # Use 5% of data for faster runs
RANDOM_STATE = 42

# --------------------
# Helper Functions
# --------------------
def start_run_dir():
    d = os.path.join(RUNS_DIR, f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def load_sampled_data(path, sample_frac=0.05):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_parquet(path)
    df = df.sample(frac=sample_frac, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"✅ Sampled dataset shape: {df.shape}")
    return df

def compute_shap(model, X_sample, run_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "shap_summary_rf.png"))
    plt.close()
    print("✅ SHAP summary plot saved.")

def plot_confusion_matrix(cm, run_dir):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {cm_path}")

# --------------------
# Main
# --------------------
def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"📂 Run folder: {run_dir}\n")

    df = load_sampled_data(DATA_PATH, sample_frac=SAMPLE_FRAC)

    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ALERT"]
    X = df.drop(columns=["Label"] + [c for c in drop_cols if c in df.columns])
    y = df["Label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    print(f"🚀 Training Random Forest...")
    model = get_rf_model()
    model.fit(X_train, y_train)

    print("✅ Training complete. Generating predictions...")
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = os.path.join(run_dir, "classification_report.json")
    with open(report_path, "w") as f:
        import json
        json.dump(report, f, indent=4)
    print(f"✅ Classification report saved → {report_path}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, run_dir)

    # Save model
    model_path = os.path.join(run_dir, "rf_model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model saved → {model_path}")

    # SHAP on sample
    print("📊 Computing SHAP values...")
    sample_X = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
    compute_shap(model, sample_X, run_dir)

if __name__ == "__main__":
    main()
