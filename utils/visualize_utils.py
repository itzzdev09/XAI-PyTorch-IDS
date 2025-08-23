import matplotlib.pyplot as plt
import os

def plot_loss(history, model_name):
    os.makedirs(f"output/{model_name}", exist_ok=True)
    
    plt.figure()
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"output/{model_name}/loss_curve.png")
    plt.close()

def plot_metric(values, metric_name, model_name):
    os.makedirs(f"output/{model_name}", exist_ok=True)

    plt.figure()
    plt.plot(values)
    plt.title(metric_name)
    plt.savefig(f"output/{model_name}/{metric_name}.png")
    plt.close()
