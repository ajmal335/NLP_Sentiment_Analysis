import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from dataset import (
    load_imdb_dataset,
    get_tokenizer,
    tokenize_dataset
)


MODEL_NAME = "distilbert-base-uncased"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading IMDb dataset...")

dataset = load_imdb_dataset()

small_dataset = dataset["train"].select(range(5))

print("Dataset loaded.")
print(f"Number of samples: {len(small_dataset)}")


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = get_tokenizer()

print("Tokenizer loaded.")


# --------------------------------------------------
# 3. Tokenize dataset
# --------------------------------------------------

print("\nTokenizing dataset...")

tokenized_dataset = tokenize_dataset(
    small_dataset,
    tokenizer
)

print("Tokenization completed.")


# --------------------------------------------------
# 4. Convert to PyTorch tensors
# --------------------------------------------------

tokenized_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

print("\nDataset converted to PyTorch format.")


# --------------------------------------------------
# 5. Create DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=2,
    shuffle=False
)

print("\nDataLoader created.")


# --------------------------------------------------
# 6. Load DistilBERT model
# --------------------------------------------------

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("Model loaded successfully!")


# --------------------------------------------------
# 7. Create optimizer
# --------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

print("\nOptimizer created.")


# --------------------------------------------------
# 8. Get one batch
# --------------------------------------------------

batch = next(iter(dataloader))

input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]
labels = batch["label"]


print("\nBatch shapes:")

print("input_ids:", input_ids.shape)
print("attention_mask:", attention_mask.shape)
print("labels:", labels.shape)


# --------------------------------------------------
# 9. Training step
# --------------------------------------------------

print("\nRunning training step...")

model.train()

# Clear previous gradients
optimizer.zero_grad()


# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)


# Get loss
loss = outputs.loss


print("\nLoss before backward:")
print(loss.item())


# --------------------------------------------------
# 10. Backpropagation
# --------------------------------------------------

loss.backward()

print("\nBackward pass completed.")


# --------------------------------------------------
# 11. Update model weights
# --------------------------------------------------

optimizer.step()

print("\nOptimizer step completed.")


# --------------------------------------------------
# 12. Predictions
# --------------------------------------------------

predictions = torch.argmax(
    outputs.logits,
    dim=1
)


print("\nLogits:")
print(outputs.logits)

print("\nPredictions:")
print(predictions)

print("\nActual labels:")
print(labels)