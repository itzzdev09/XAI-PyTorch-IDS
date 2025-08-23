def get_model(output_dim):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        task_type='GPU',    # force GPU
        devices='0',        # GPU device id
        random_seed=42,
        verbose=50
    )
    return model
