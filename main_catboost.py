import os
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Paths
PROCESSED_PATH = "Data/processed/processed_data.csv"
MODEL_PATH = "Data/processed/catboost_model.cbm"

def train_model():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError("❌ Run preprocess.py first to generate processed data.")

    print("📂 Loading processed data...")
    df = pd.read_csv(PROCESSED_PATH)

    # Assuming last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    train_pool = Pool(X, y)

    # If model exists, load it and continue training
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=10)

    if os.path.exists(MODEL_PATH):
        print(f"🔄 Resuming training from {MODEL_PATH}")
        model.load_model(MODEL_PATH)

    # Train model
    model.fit(train_pool, init_model=model if os.path.exists(MODEL_PATH) else None)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
