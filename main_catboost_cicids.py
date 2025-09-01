from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main():
    warnings.filterwarnings("ignore")
    run_dir = os.path.join(RUNS_DIR, f"KFold_Run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Output → {run_dir}")

    # Load full dataset
    df = load_dataset(DATA_PATH)
    df, cat_cols = infer_and_fix_types(df)

    # ---------
    # Step 1: Reserve 50% of full dataset (stratified) for training/testing
    # ---------
    df_train_test, _ = train_test_split(
        df,
        test_size=0.5,
        stratify=df["LabelMapped"],
        random_state=RANDOM_STATE
    )
    print(f"✔ Reserved 50% of dataset for training & testing → {df_train_test.shape}")

    # ---------
    # Step 2: Stratified downsampling (per class)
    # ---------
    df_small = stratified_downsample(df_train_test)

    # ---------
    # Step 3: Prepare features and target
    # ---------
    drop_cols = ["Flow_ID", "Timestamp", "Label", "LabelMapped", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    X = df_small.drop(columns=[c for c in drop_cols if c in df_small.columns])
    y = df_small["LabelMapped"].values
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    with open(os.path.join(run_dir, "features.json"), "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    # ---------
    # Step 4: Stratified K-Fold Cross-Validation
    # ---------
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n🔁 Fold {fold} — Training on {len(train_idx)} samples, Validating on {len(val_idx)}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        try:
            model = CatBoostClassifier(**CB_PARAMS_GPU)
            model.fit(train_pool, eval_set=val_pool, verbose=50)
        except CatBoostError:
            print("⚠️ GPU failed — switching to CPU")
            model = CatBoostClassifier(**CB_PARAMS_CPU)
            model.fit(train_pool, eval_set=val_pool)

        # Evaluate
        y_pred = model.predict(X_val).astype(int)

        fold_result = {
            "fold": fold,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision_macro": precision_score(y_val, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_val, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_val, y_pred, average='macro', zero_division=0)
        }
        fold_metrics.append(fold_result)
        print(f"✅ Fold {fold} metrics: {fold_result}")

        # Save fold model
        model.save_model(os.path.join(run_dir, f"model_fold_{fold}.cbm"))

    # ---------
    # Step 5: Save and report average metrics
    # ---------
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(os.path.join(run_dir, "kfold_metrics.csv"), index=False)
    avg_metrics = metrics_df.mean(numeric_only=True).to_dict()
    print(f"\n📊 Average Metrics across 5 folds:\n{json.dumps(avg_metrics, indent=2)}")

    with open(os.path.join(run_dir, "average_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=2)

    # ---------
    # Step 6: Final model on full data → evaluate on 20% holdout from 50%
    # ---------
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    train_pool = Pool(X_train_final, y_train_final, cat_features=cat_indices)
    test_pool = Pool(X_test_final, y_test_final, cat_features=cat_indices)

    final_model = CatBoostClassifier(**CB_PARAMS_GPU)
    try:
        final_model.fit(train_pool, eval_set=test_pool)
    except CatBoostError:
        final_model = CatBoostClassifier(**CB_PARAMS_CPU)
        final_model.fit(train_pool, eval_set=test_pool)

    final_model.save_model(os.path.join(run_dir, "final_model.cbm"))

    y_pred_final = final_model.predict(X_test_final).astype(int)
    y_proba_final = final_model.predict_proba(X_test_final)

    final_report = classification_report(y_test_final, y_pred_final, output_dict=True)
    with open(os.path.join(run_dir, "final_classification_report.json"), "w") as f:
        json.dump(final_report, f, indent=2)
    print("🧾 Final model classification report saved.")

    cm = confusion_matrix(y_test_final, y_pred_final)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "final_confusion_matrix.png"))
    plt.close()

    # SHAP on final test set
    sample_n = min(1000, len(X_test_final))
    compute_shap(final_model, X_test_final.sample(n=sample_n, random_state=RANDOM_STATE), cat_indices, run_dir)

    print("✅ All Done.")
