import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import (
    MODEL_SAVE_PATH,
    LABEL_NAMES
)


def predict_sentiment(text):

    # --------------------------------------------------
    # 1. Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------
    # 2. Load tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SAVE_PATH
    )

    # --------------------------------------------------
    # 3. Load model
    # --------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH
    )

    model.to(device)

    model.eval()

    # --------------------------------------------------
    # 4. Tokenize input
    # --------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------
    # 5. Prediction
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # --------------------------------------------------
    # 6. Convert logits to probabilities
    # --------------------------------------------------

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0, prediction
    ].item()

    # --------------------------------------------------
    # 7. Result
    # --------------------------------------------------

    sentiment = LABEL_NAMES[
        prediction
    ]

    return sentiment, confidence


if __name__ == "__main__":

    text = input(
        "\nEnter a movie review: "
    )

    sentiment, confidence = predict_sentiment(
        text
    )

    print(
        f"\nSentiment: {sentiment.upper()}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )