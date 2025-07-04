import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pylab import plt

from deep_metabolitics.config import image_log_dir


def draw_curve(epochs, train_values=None, val_values=None, metric_name="Loss"):
    if train_values is not None:
        epochs = len(train_values)
    else:
        epochs = len(val_values)
    epochs_range = range(1, epochs + 1)
    title = ""

    if train_values is not None:
        plt.plot(epochs_range, train_values, label=f"Training {metric_name}")
        title += "Training"

    if val_values is not None:
        if train_values is not None:
            title += " and"
        title += f" Validation {metric_name}"

        plt.plot(epochs_range, val_values, label=f"Validation {metric_name}")

    # Add in a title and axes labels
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)

    # Set the tick locations
    plt.xticks(np.arange(0, epochs + 1, 2))

    # Display the plot
    plt.legend(loc="best")

    dirname = metric_name.split(" ")[0].replace(".", "_").replace("-", "_")
    fpath = image_log_dir / dirname.replace("_", "/")
    os.makedirs(fpath, exist_ok=True)

    fpath = fpath / f"{metric_name.split(' ')[1]}.png"
    plt.savefig(fpath)
    plt.show()


# 50 step ve her biri için 20 test sonucu olacak şekilde rastgele veri oluşturuyoruz


def plot_steps_box_results(data):
    results_per_step = data.shape[1]
    # data = np.random.randn(
    #     steps, results_per_step
    # )  # Örnek veriler, kendi verilerinle değiştir

    # Veriyi uzun formata çeviriyoruz
    df = pd.DataFrame(data)
    df = df.melt(var_name="Test No", value_name="Sonuç")
    df["Step"] = df.index // results_per_step

    # Boxplot çizdirme
    plt.figure(figsize=(15, 6))
    sns.boxplot(x="Step", y="Sonuç", data=df)
    plt.xlabel("Step")
    plt.ylabel("Test Sonuçları")
    plt.title("50 Step İçin Test Sonuçlarının Boxplot Gösterimi")
    plt.show()
