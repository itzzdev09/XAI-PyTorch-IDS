import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs[:, 1].cpu().numpy()) if probs.shape[1] > 1 else None

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

    # Handle AUC for binary/multi-class
    try:
        if len(set(y_true)) > 2:
            metrics["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovo")
        else:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        metrics["auc"] = None

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics
