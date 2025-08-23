import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from utils.io import load_selected_csvs
from utils.preprocessing import preprocess_data


def main():
    print("🚀 Loading data...")

    # ✅ Check current working directory
    print(f"Current working directory: {os.getcwd()}")

    # ✅ Safer path handling
    relative_path = os.path.join("data", "processed")
    absolute_path = r"C:\Users\Work\XAI pytorch\data\processed"  # <-- Change if needed

    if os.path.exists(relative_path):
        folder_path = relative_path
    elif os.path.exists(absolute_path):
        folder_path = absolute_path
    else:
        raise FileNotFoundError(
            f"❌ Could not find 'data/processed'. Checked:\n- {relative_path}\n- {absolute_path}"
        )

    # ✅ Load CSVs
    df = load_selected_csvs(folder_path)

    # ✅ Preprocess
    X, y = preprocess_data(df)

    # ✅ Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ CatBoost model
    model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        verbose=100
    )

    print("🚀 Training CatBoost model on GPU...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # ✅ Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ✅ Reports
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"🔥 AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
