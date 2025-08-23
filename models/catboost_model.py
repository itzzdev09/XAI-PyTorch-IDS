# models/catboost_model.py

from catboost import CatBoostClassifier

def get_catboost_model():
    """
    Returns a CatBoostClassifier configured for GPU training
    with balanced speed, accuracy, and GPU usage.
    """

    model = CatBoostClassifier(
        iterations=200,          # fewer iterations for faster training
        learning_rate=0.1,       # moderate learning rate
        depth=6,                 # shallower depth = faster training
        l2_leaf_reg=3,           # regularization
        loss_function="Logloss", # classification loss
        eval_metric="AUC",       # track AUC during training
        random_seed=42,          # reproducibility

        # GPU SETTINGS
        task_type="GPU",         # force GPU usage
        devices="0",             # GPU index (0 = first GPU)

        # Subsampling for speed + less GPU load
        bootstrap_type="Bernoulli",
        subsample=0.6,           # use 60% of data per tree

        # Other efficiency tricks
        grow_policy="SymmetricTree",  # default, balanced
        boosting_type="Plain",        # faster boosting
        verbose=50,                   # log every 50 iterations
        use_best_model=True
    )

    return model
