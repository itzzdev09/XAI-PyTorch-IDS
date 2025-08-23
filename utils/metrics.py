import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

def evaluate_and_save_metrics(y_true, y_pred, y_prob, class_names, save_path):
    """
    Saves classification report + macro/micro ROC-AUC (OvR) when possible.
    """
    # classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report).transpose()

    # AUC (macro OvR) – requires >= 2 classes
    auc_macro = np.nan
    try:
        if y_prob is not None and len(np.unique(y_true)) > 1:
            auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        pass
    df_report.loc["macro_auc_ovr", "score"] = auc_macro

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_report.to_csv(save_path, index=True)

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
