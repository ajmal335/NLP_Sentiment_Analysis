
import os
import json

import matplotlib.pyplot as plt


RESULTS_DIR = "results"


def load_history():

    path = os.path.join(
        RESULTS_DIR,
        "training_history.json"
    )

    with open(path, "r") as f:
        return json.load(f)


def plot_loss(history):

    epochs = history["epochs"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


def plot_accuracy(history):

    epochs = history["epochs"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["validation_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


def main():

    history = load_history()

    plot_loss(history)
    plot_accuracy(history)


if __name__ == "__main__":
    main()
