
import os
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from dataset import (
    load_imdb_dataset,
    tokenize_dataset
)

from config import (
    BATCH_SIZE,
    MODEL_SAVE_PATH,
    NUM_LABELS,
    RESULTS_DIR
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at "
            f"{MODEL_SAVE_PATH}"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print("\nLoading IMDb test dataset...")

    dataset = load_imdb_dataset()
    test_dataset = dataset["test"]

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SAVE_PATH
    )

    print("\nTokenizing test dataset...")

    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer
    )

    test_dataset = test_dataset.remove_columns(
        ["text"]
    )

    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"
        ]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nLoading trained model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH,
        num_labels=NUM_LABELS
    )

    model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    print("\nRunning evaluation...")

    with torch.no_grad():

        for batch in tqdm(
            test_loader,
            desc="Evaluating"
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                batch_predictions.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    # Metrics

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="binary"
        )
    )

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    report = classification_report(
        true_labels,
        predictions,
        target_names=[
            "Negative",
            "Positive"
        ]
    )

    # Print results

    print("\n")
    print("=" * 50)
    print("IMDb Sentiment Analysis Evaluation")
    print("=" * 50)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    print("=" * 50)

    # Save metrics

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "test_samples": len(test_dataset)
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"\nMetrics saved to: {metrics_path}"
    )

    # Save classification report

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(report)

    print(
        f"Classification report saved to: "
        f"{report_path}"
    )

    # Create confusion matrix visualization

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "IMDb Sentiment Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    plt.yticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    confusion_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved to: "
        f"{confusion_path}"
    )


if __name__ == "__main__":
    main()
