from torch.utils.data import DataLoader
from dataset import (
    load_imdb_dataset,
    get_tokenizer,
    tokenize_dataset
)


dataset = load_imdb_dataset()

small_dataset = dataset["train"].select(range(5))

tokenizer = get_tokenizer()

tokenized_dataset = tokenize_dataset(
    small_dataset,
    tokenizer
)


print("Original dataset:")
print(small_dataset)

print("\nTokenized dataset:")
print(tokenized_dataset)

print("\nFirst example:")
print(tokenized_dataset[0])

print("\nData types:")

print("input_ids type:")
print(type(tokenized_dataset[0]["input_ids"]))

print("\nattention_mask type:")
print(type(tokenized_dataset[0]["attention_mask"]))

print("\nlabel type:")
print(type(tokenized_dataset[0]["label"]))


print("\nConverting dataset to PyTorch format...")

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

print("\nData types after conversion:")

print("input_ids:")
print(type(tokenized_dataset[0]["input_ids"]))

print("\nattention_mask:")
print(type(tokenized_dataset[0]["attention_mask"]))

print("\nlabel:")
print(type(tokenized_dataset[0]["label"]))

print("\nCreating DataLoader...")

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=2,
    shuffle=False
)

batch = next(iter(dataloader))

print("\nFirst batch:")

print("input_ids shape:")
print(batch["input_ids"].shape)

print("\nattention_mask shape:")
print(batch["attention_mask"].shape)

print("\nlabel shape:")
print(batch["label"].shape)