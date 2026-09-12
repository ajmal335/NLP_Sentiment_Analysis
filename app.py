
import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.config import (
    MODEL_SAVE_PATH,
    LABEL_NAMES
)


st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SAVE_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SAVE_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


st.title("🎬 IMDb Sentiment Analyzer")

st.write(
    "Analyze the sentiment of a movie review "
    "using a fine-tuned DistilBERT model."
)

st.info(
    "Model: DistilBERT fine-tuned on the IMDb dataset"
)

review = st.text_area(
    "Enter a movie review:",
    height=180,
    placeholder="Example: I really enjoyed this movie..."
)


if st.button(
    "Analyze Sentiment",
    type="primary"
):

    if not review.strip():

        st.warning(
            "Please enter a movie review."
        )

    else:

        tokenizer, model, device = load_model()

        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

        sentiment = LABEL_NAMES[
            prediction
        ]

        st.divider()

        if sentiment == "positive":

            st.success(
                f"### 😊 Positive\n\n"
                f"Confidence: {confidence:.2%}"
            )

        else:

            st.error(
                f"### 😞 Negative\n\n"
                f"Confidence: {confidence:.2%}"
            )

        st.subheader("Prediction Probabilities")

        st.write(
            f"Negative: "
            f"{probabilities[0, 0].item():.2%}"
        )

        st.progress(
            float(probabilities[0, 0].item())
        )

        st.write(
            f"Positive: "
            f"{probabilities[0, 1].item():.2%}"
        )

        st.progress(
            float(probabilities[0, 1].item())
        )


st.divider()

st.caption(
    "NLP Sentiment Analysis | "
    "DistilBERT + IMDb"
)
