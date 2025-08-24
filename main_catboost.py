
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import shap
from models.catboost_model import get_model

PROCESSED_DIR = "data/processed"
RUNS_DIR = "Runs"
os.makedirs(RUNS_DIR, exist_ok=True)

def train_single_dataset(file_path, run_folder):
    print(f"\n📂 Loading dataset: {file_path}")
    df = pd.read_parquet(file_path)

    if df['Label'].nunique() < 2:
        print(f"⚠️ Skipping {file_path}: only one class present in target.")
        return

    X = df.drop("Label", axis=1)
    y = df["Label"]
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    model = get_model()

    # ETA calculation
    start_time = time.time()
    def verbose_callback(iteration, logs):
        elapsed = time.time() - start_time
        if iteration > 0:
            eta = elapsed / iteration * (model.get_params()["iterations"] - iteration)
            print(f"Iteration {iteration}/{model.get_params()['iterations']} – ETA: {eta:.1f}s")

    # Train with callback
    model.fit(
        X, y,
        cat_features=cat_cols,
        verbose=0,
        callbacks=[verbose_callback]
    )

    # Save model
    model_name = os.path.join(run_folder, os.path.basename(file_path).replace(".parquet", "_catboost.cbm"))
    model.save_model(model_name)
    print(f"📌 Model saved → {model_name}")

    # SHAP summary
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    shap_file = os.path.join(run_folder, os.path.basename(file_path).replace(".parquet", "_shap.png"))
    plt.savefig(shap_file)
    plt.close()
    print(f"📌 SHAP summary saved → {shap_file}")

def main():
    run_folder = os.path.join(RUNS_DIR, f"Run_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"📂 New run folder: {run_folder}")

    parquet_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith(".parquet")])
    if not parquet_files:
        raise FileNotFoundError("❌ No processed parquet files found. Run preprocess.py first!")

    for file in parquet_files:
        file_path = os.path.join(PROCESSED_DIR, file)
        train_single_dataset(file_path, run_folder)

if __name__ == "__main__":
    main()
