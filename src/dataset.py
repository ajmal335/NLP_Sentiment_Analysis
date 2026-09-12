from datasets import load_dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH


def load_imdb_dataset():
    """
    Load the IMDb sentiment dataset.

    Returns:
        DatasetDict: IMDb train, test and unsupervised splits.
    """

    return load_dataset("imdb")


def get_tokenizer():
    """
    Load the pretrained DistilBERT tokenizer.

    Returns:
        AutoTokenizer: DistilBERT tokenizer.
    """

    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize a dataset using the DistilBERT tokenizer.

    Args:
        dataset: Hugging Face dataset.
        tokenizer: DistilBERT tokenizer.

    Returns:
        Tokenized dataset.
    """

    def tokenize_function(examples):

        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    return dataset.map(
        tokenize_function,
        batched=True
    )