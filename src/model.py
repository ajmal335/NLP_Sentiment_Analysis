from transformers import AutoModelForSequenceClassification

from config import MODEL_NAME, NUM_LABELS


def create_model():
    """
    Create DistilBERT for binary sentiment classification.

    Returns:
        DistilBertForSequenceClassification
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    return model