import os
import numpy as np
import shap
import matplotlib.pyplot as plt

def run_and_save_shap(model, X_test, feature_names, out_dir, max_samples=3000):
    n = min(max_samples, len(X_test))
    X_shap = X_test[:n]
    print("Saving SHAP plots to:", out_dir)
    print("X_shap shape:", X_shap.shape)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # CatBoost multiclass returns list[n_classes] of (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_values_to_plot = np.mean(np.abs(np.array(shap_values)), axis=0)
    else:
        shap_values_to_plot = shap_values

    # Dot summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_shap, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150)
    plt.close()

    # Bar summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X_shap, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_importance.png"), dpi=150)
    plt.close()
