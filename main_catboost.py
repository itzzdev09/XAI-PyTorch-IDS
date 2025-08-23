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
from catboost import CatBoostClassifier
import time

# ---------------- Config ----------------
DATA_DIR = "./Data/simargl2022"
SELECT_CSVS = [
    "dos-03-18-2022-19-27-05.csv",
    "malware-03-25-2022-17-57-07.csv",
    "normal-03-18-2022-19-17-31.csv",
    "normal-03-18-2022-19-25-48.csv",
    "portscanning-03-16-2022-13-44-50.csv"
]
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
    print("Class distribution:\n", df["ALERT"].value_counts())
    return df

def preprocess_dataset(df):
    # Drop irrelevant columns
    drop_cols = ["FLOW_ID", "PROTOCOL_MAP", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "ANALYSIS_TIMESTAMP"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Remove classes with very few samples
    df = df.groupby("ALERT").filter(lambda x: len(x) >= MIN_SAMPLES_PER_CLASS)
    if df["ALERT"].nunique() < 2:
        raise ValueError("Not enough classes after filtering. Reduce MIN_SAMPLES_PER_CLASS.")

    df["ALERT"] = df["ALERT"].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["ALERT"])

    X = df.select_dtypes(include=["int64", "float64"])

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # Convert to float32 for GPU CatBoost
    X_res = X_res.astype(np.float32)

    return X_res, y_res, label_encoder, X.columns.tolist()

def shap_analysis(model, X_test, feature_names, output_folder):
    subset_size = min(3000, len(X_test))
    X_shap = X_test[:subset_size]

    print("Saving SHAP plots to:", output_folder)
    print("X_shap shape:", X_shap.shape)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values_combined = np.mean(np.abs(np.array(shap_values)), axis=0)
        shap_values_to_plot = shap_values_combined
    else:
        shap_values_to_plot = shap_values

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_shap, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "shap_summary.png"), dpi=150)
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_shap, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "shap_feature_importance.png"), dpi=150)
    plt.close()

def get_catboost_model(output_dim):
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        task_type='GPU',      # force GPU
        devices='0',          # GPU device id
        random_seed=42,
        verbose=50
    )
    return model

# ---------------- Main ----------------
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    df = load_selected_csvs(DATA_DIR, SELECT_CSVS)
    X_res, y_res, label_encoder, feature_names = preprocess_dataset(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Create model
    model = get_catboost_model(output_dim=len(np.unique(y_res)))

    # Train with estimated time
    start_time = time.time()
    print("🚀 Training CatBoost model on GPU...")
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=20,
        verbose=50
    )
    elapsed = time.time() - start_time
    print(f"⏱ Total training time: {elapsed:.2f}s (~{elapsed/model.tree_count_:.2f}s per iteration)")

    # Metrics
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
