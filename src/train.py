import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from dataset import (
    load_imdb_dataset,
    get_tokenizer,
    tokenize_dataset
)

from model import create_model

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    RANDOM_SEED,
    MODEL_SAVE_PATH
)


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model.
        dataloader: DataLoader containing evaluation data.
        device: CPU or CUDA device.

    Returns:
        tuple: average_loss, accuracy
    """

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / len(dataloader)

    accuracy = correct / total

    return average_loss, accuracy


def main():

    # --------------------------------------------------
    # 1. Set random seed
    # --------------------------------------------------

    torch.manual_seed(RANDOM_SEED)

    # --------------------------------------------------
    # 2. Select device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

    # --------------------------------------------------
    # 3. Load IMDb dataset
    # --------------------------------------------------

    print("\nLoading IMDb dataset...")

    dataset = load_imdb_dataset()

    print(dataset)

    # --------------------------------------------------
    # 4. Create train/validation split
    # --------------------------------------------------

    train_dataset = dataset["train"]

    split = train_dataset.train_test_split(
        test_size=0.1,
        seed=RANDOM_SEED
    )

    train_dataset = split["train"]
    validation_dataset = split["test"]

    test_dataset = dataset["test"]

    print("\nDataset sizes:")

    print(f"Training:   {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print(f"Test:       {len(test_dataset)}")

    # --------------------------------------------------
    # 5. Load tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = get_tokenizer()

    # --------------------------------------------------
    # 6. Tokenize datasets
    # --------------------------------------------------

    print("\nTokenizing datasets...")

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer
    )

    validation_dataset = tokenize_dataset(
        validation_dataset,
        tokenizer
    )

    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer
    )

    # --------------------------------------------------
    # 7. Remove raw text
    # --------------------------------------------------

    train_dataset = train_dataset.remove_columns(
        ["text"]
    )

    validation_dataset = validation_dataset.remove_columns(
        ["text"]
    )

    test_dataset = test_dataset.remove_columns(
        ["text"]
    )

    # --------------------------------------------------
    # 8. Convert to PyTorch tensors
    # --------------------------------------------------

    columns = [
        "input_ids",
        "attention_mask",
        "label"
    ]

    train_dataset.set_format(
        type="torch",
        columns=columns
    )

    validation_dataset.set_format(
        type="torch",
        columns=columns
    )

    test_dataset.set_format(
        type="torch",
        columns=columns
    )

    # --------------------------------------------------
    # 9. Create DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nDataLoaders created.")

    # --------------------------------------------------
    # 10. Create model
    # --------------------------------------------------

    print("\nLoading DistilBERT model...")

    model = create_model()

    model.to(device)

    # --------------------------------------------------
    # 11. Create optimizer
    # --------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------
    # 12. Training
    # --------------------------------------------------

    best_validation_accuracy = 0.0

    for epoch in range(EPOCHS):

        print(
            f"\n========== Epoch "
            f"{epoch + 1}/{EPOCHS} =========="
        )

        model.train()

        total_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc="Training"
        )

        for batch in progress_bar:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=loss.item()
            )

        # Average training loss
        train_loss = total_loss / len(train_loader)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        validation_loss, validation_accuracy = evaluate_model(
            model,
            validation_loader,
            device
        )

        print(
            f"\nTrain Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            os.makedirs(
                MODEL_SAVE_PATH,
                exist_ok=True
            )

            model.save_pretrained(
                MODEL_SAVE_PATH
            )

            tokenizer.save_pretrained(
                MODEL_SAVE_PATH
            )

            print("\nBest model saved.")

    # --------------------------------------------------
    # 13. Load best model
    # --------------------------------------------------

    print("\nLoading best model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH
    )

    model.to(device)

    # --------------------------------------------------
    # 14. Final test evaluation
    # --------------------------------------------------

    print("\n========== Final Test ==========")

    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        device
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()