# rf_model.py

from sklearn.ensemble import RandomForestClassifier

def get_rf_model(n_estimators=100, max_depth=25, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        class_weight='balanced_subsample',
        random_state=random_state
    )
