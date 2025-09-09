from sklearn.ensemble import RandomForestClassifier

def get_rf_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
