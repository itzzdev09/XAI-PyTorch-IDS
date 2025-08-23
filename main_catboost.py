import os
import json
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from catboost import CatBoostClassifier, Pool
from datetime import datetime

# ======================
# Utility Functions
# ======================
def load_data():
    print("📂 Loading processed data...")
    processed_dir = "Data/processed"
    data_path = os.path.join(processed_dir, "processed.csv")
    label_map_path = os.path.join(processed_dir, "label_map.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError("❌ Processed data not found. Run preprocess.py first!")

    df = pd.read_csv(data_path)

    if "Label" not in df.columns:
        raise ValueError("❌ 'Label' column not found in dataset. Make sure preprocessing added it.")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Load label mapping
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        # Flip mapping: {0: "Normal", 1: "DoS", ...}
        label_map = {v: k for k, v in label_map.items()}
    else:
        label_map = {i: str(i) for i in sorted(set(y))}

    return X, y, label_map


def get_run_folder():
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join("Runs", f"Run_{run_time}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"📂 New run folder: {run_folder}")
    return run_folder


def save_metrics(y_true, preds, label_map, run_folder):
    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        preds,
        display_labels=[label_map[i] for i in sorted(label_map.keys())],
        cmap="Blues",
        xticks_rotation=45
    )
    disp.figure_.savefig(os.path.join(run_folder, "confusion_matrix.png"))
    plt.close()

    # Classification Report (Text + Image)
    report_text = classification_report(
        y_true,
        preds,
        target_names=[label_map[i] for i in sorted(label_map.keys())]
    )

    # Save as TXT
    with open(os.path.join(run_folder, "classification_report.txt"), "w") as f:
        f.write(report_text)

    # Save as Image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.text(0, 1, report_text, fontsize=10, va="top", family="monospace")
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "metrics.png"))
    plt.close()
    print("📊 Metrics saved (confusion_matrix.png, classification_report.txt, metrics.png)")


def save_shap(model, X, run_folder, label_map):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # SHAP Summary Plot
    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(run_folder, "shap_summary.png"))
    plt.close()

    # SHAP Bar Plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(os.path.join(run_folder, "shap_bar.png"))
    plt.close()

    print("📈 SHAP plots saved (shap_summary.png, shap_bar.png)")


# ======================
# Main Training Function
# ======================
def main():
    run_folder = get_run_folder()
    X, y, label_map = load_data()

    model_path = os.path.join(run_folder, "catboost_model.cbm")

    # If model exists, load it
    if os.path.exists(model_path):
        print("🔄 Loading existing CatBoost model...")
        model = CatBoostClassifier()
        model.load_model(model_path)
    else:
        print("🚀 Training new CatBoost model...")
        model = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            loss_function="MultiClass",
            random_seed=42,
            task_type="GPU" if "GPU" in CatBoostClassifier()._init_params else "CPU"
        )
        model.fit(X, y, verbose=50)
        model.save_model(model_path)
        print(f"✅ Model saved at {model_path}")

    # Predictions
    preds = model.predict(X)
    preds = preds.astype(int).flatten()

    # Save metrics
    save_metrics(y, preds, label_map, run_folder)

    # Save SHAP values
    save_shap(model, X.sample(min(500, len(X)), random_state=42), run_folder, label_map)


if __name__ == "__main__":
    main()
