import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from dataset import (
    load_imdb_dataset,
    tokenize_dataset
)

from config import (
    BATCH_SIZE,
    MODEL_SAVE_PATH,
    NUM_LABELS
)


def main():

    # --------------------------------------------------
    # 1. Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # 2. Check model
    # --------------------------------------------------

    if not os.path.exists(MODEL_SAVE_PATH):

        raise FileNotFoundError(
            f"Trained model not found at "
            f"{MODEL_SAVE_PATH}"
        )

    # --------------------------------------------------
    # 3. Load test dataset
    # --------------------------------------------------

    print("\nLoading IMDb test dataset...")

    dataset = load_imdb_dataset()

    test_dataset = dataset["test"]

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------
    # 4. Load tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SAVE_PATH
    )

    # --------------------------------------------------
    # 5. Tokenize test dataset
    # --------------------------------------------------

    print("\nTokenizing test dataset...")

    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer
    )

    # --------------------------------------------------
    # 6. Remove raw text
    # --------------------------------------------------

    test_dataset = test_dataset.remove_columns(
        ["text"]
    )

    # --------------------------------------------------
    # 7. Convert to PyTorch
    # --------------------------------------------------

    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "label"
        ]
    )

    # --------------------------------------------------
    # 8. DataLoader
    # --------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------
    # 9. Load trained model
    # --------------------------------------------------

    print("\nLoading trained model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH,
        num_labels=NUM_LABELS
    )

    model.to(device)

    model.eval()

    # --------------------------------------------------
    # 10. Inference
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 11. Calculate metrics
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 12. Print results
    # --------------------------------------------------

    print("\n")
    print("=" * 45)
    print("IMDb Sentiment Analysis Evaluation")
    print("=" * 45)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("=" * 45)


if __name__ == "__main__":
    main()