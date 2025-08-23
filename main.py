import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

from models.catboost_model import get_model  # your existing catboost model

# ---------------- Config ----------------
DATA_DIR = "./Data/simargl2022"
SELECT_CSVS = [
    "dos-03-18-2022-19-27-05.csv",
    "malware-03-25-2022-17-57-07.csv",
    "normal-03-18-2022-19-17-31.csv",
    "normal-03-18-2022-19-25-48.csv",
    "portscanning-03-16-2022-13-44-50.csv"
]  # subset for efficient processing
MIN_SAMPLES_PER_CLASS = 10
RESULTS_DIR = "results"

# ---------------- Utils ----------------
def load_selected_csvs(data_dir, selected_files):
    dfs = []
    for file in selected_files:
        path = os.path.join(data_dir, file)
        try:
            print(f"Loading {path} ...")
            dfs.append(pd.read_csv(path))
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples loaded: {len(df)}")
    print("Class distribution:\n", df['ALERT'].value_counts())
    return df

def preprocess_dataset(df):
    drop_cols = ["FLOW_ID", "PROTOCOL_MAP", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ANALYSIS_TIMESTAMP"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Filter rare classes
    df = df.groupby("ALERT").filter(lambda x: len(x) >= MIN_SAMPLES_PER_CLASS)
    if df["ALERT"].nunique() < 2:
        raise ValueError("Not enough classes after filtering. Reduce MIN_SAMPLES_PER_CLASS.")

    df["ALERT"] = df["ALERT"].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["ALERT"])

    X = df.select_dtypes(include=["int64", "float64"])
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE oversampling (more realistic than random oversample)
    smote = SMOTE(random_state=42, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    print(f"After SMOTE, class distribution: {np.bincount(y_res)}")

    return X_res, y_res, label_encoder, feature_names

# ---------------- SHAP ----------------
def shap_analysis(model, X_test, feature_names, output_folder):
    subset_size = min(3000, len(X_test))
    X_shap = X_test[:subset_size]

    print("Saving SHAP plots to:", output_folder)
    print("X_shap shape:", X_shap.shape)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # For multi-class, combine shap_values to get average importance per feature
    if isinstance(shap_values, list):
        shap_values_to_plot = np.mean(np.abs(np.array(shap_values)), axis=0)
    else:
        shap_values_to_plot = np.abs(shap_values)

    # Summary plot (dot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_shap, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "shap_summary.png"), dpi=150)
    plt.close()

    # Bar plot (mean importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_shap, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "shap_feature_importance.png"), dpi=150)
    plt.close()

# ---------------- Main ----------------
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    df = load_selected_csvs(DATA_DIR, SELECT_CSVS)
    X_res, y_res, label_encoder, feature_names = preprocess_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    model = get_model(output_dim=len(np.unique(y_res)))
    print("🚀 Training CatBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=50,
        early_stopping_rounds=50,
        use_best_model=True
    )

    # Evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(os.path.join(output_folder, "metrics.csv"), index=True)
    print("📊 Metrics saved.")

    # SHAP
    shap_analysis(model, X_test, feature_names, output_folder)
    print(f"📊 SHAP plots saved in {output_folder}")

    # Save model
    model.save_model(os.path.join(output_folder, "catboost_model.cbm"))
    print("💾 Model saved.")

if __name__ == "__main__":
    main()
