import os
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt
from datetime import datetime

def load_data():
    parquet_path = "Data/processed_data.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError("❌ Processed data not found. Run preprocess.py first!")

    print("📂 Loading processed data (Parquet)...")
    df = pd.read_parquet(parquet_path)

    y = df['Label']
    X = df.drop(columns=['Label'])

    # Let CatBoost auto-detect categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return X, y, categorical_cols

def main():
    run_dir = os.path.join("Runs", f"Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"📂 New run folder: {run_dir}")

    X, y, categorical_cols = load_data()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CatBoost model
    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function='MultiClass',
        task_type="GPU",  # use GPU if available
        eval_metric="Accuracy",
        early_stopping_rounds=50,
        verbose=100
    )

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # Save model
    model.save_model(os.path.join(run_dir, "catboost_model.cbm"))
    print(f"✅ Model saved to {run_dir}/catboost_model.cbm")

    # Evaluation
    y_pred = model.predict(X_test)
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(run_dir, "shap_summary.png"))
    plt.close()

if __name__ == "__main__":
    main()
