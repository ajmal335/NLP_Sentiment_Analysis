from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


text = "I really enjoyed this movie."


encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=10
)


print("Input IDs:")
print(encoded["input_ids"])

print("\nAttention Mask:")
print(encoded["attention_mask"])