import xgboost as xgb

def get_xgb_model(num_classes, random_seed=42):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',  # Change to hist
        device='cuda',       # Use GPU properly
        random_state=random_seed,
        verbosity=1
    )
    return model
