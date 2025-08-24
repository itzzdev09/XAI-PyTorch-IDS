# models/catboost_model.py
from catboost import CatBoostClassifier

def get_model():
    """
    Returns a CatBoostClassifier configured for GPU training
    with balanced speed, accuracy, and GPU usage.
    """
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        task_type="GPU",
        devices="0",
        bootstrap_type="Bernoulli",
        subsample=0.6,
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        verbose=50,
        use_best_model=True
    )
    return model
