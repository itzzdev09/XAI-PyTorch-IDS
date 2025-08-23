import shap
import torch

def explain_kernel(model, X_sample):
    explainer = shap.KernelExplainer(model.predict, X_sample)
    shap_values = explainer.shap_values(X_sample, nsamples=50)
    return shap_values

def explain_deep(model, X_sample):
    model.eval()
    background = torch.from_numpy(X_sample[:50]).float()
    test_data = torch.from_numpy(X_sample[:200]).float()

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_data)

    return shap_values
