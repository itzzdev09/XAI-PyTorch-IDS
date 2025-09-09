import os
import time
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import shap

from your_utils_file import (
    load_dataset,
    infer_and_fix_types,
    stratified_downsample,
    start_run_dir
)

# --------------------
# Config
# --------------------
DATA_PATH = "data/processed/simargl_full.parquet"
RUNS_DIR = "Runs_RF"
os.makedirs(RUNS_DIR, exist_ok=True)

PER_CLASS_MAX = 500_000
N_SPLITS = 10
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 30,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced_subsample"
}


def compute_shap_rf(model, X_sample, run_dir):
    """Compute SHAP values for Random Forest."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "shap_summary_rf.png"))
    plt.close()

    print("✅ SHAP summary plot saved.")


def main():
    warnings.filterwarnings("ignore")
    run_dir = start_run_dir()
    print(f"📂 Run folder: {run_dir}\n")

    df = load_dataset(DATA_PATH)
    df, cat_cols = infer_and_fix_types(df)
    df_small = stratified_downsample(df, per_class_max=PER_CLASS_MAX)

    drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ALERT"]
    X = df_small.drop(columns=["Label"] + [c for c in drop_cols if c in df_small.columns])
    y = df_small["Label"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🔹 Fold {fold}/{N_SPLITS}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        fold_metrics.append(report)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        cm_path = os.path.join(run_dir, f"confusion_matrix_fold_{fold}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"📌 Confusion matrix saved → {cm_path}")

        # Compute SHAP on Fold 1
        if fold == 1:
            sample_idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
            compute_shap_rf(model, X_test.iloc[sample_idx], run_dir)


if __name__ == "__main__":
    main()
