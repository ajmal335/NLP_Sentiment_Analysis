
# 🎬 IMDb Sentiment Analysis with DistilBERT

A professional Natural Language Processing project that fine-tunes
DistilBERT for binary sentiment classification on the IMDb movie
review dataset.

## 🚀 Project Overview

This project demonstrates an end-to-end NLP machine learning pipeline:

- Dataset loading
- Train/validation/test splitting
- Text tokenization
- Transformer-based modeling
- DistilBERT fine-tuning
- GPU training with PyTorch
- Model evaluation
- Confusion matrix analysis
- Sentiment prediction
- Interactive Streamlit application

## 🧠 Model

**Base Model:** `distilbert-base-uncased`

**Task:** Binary sentiment classification

**Classes:**

- `0` → Negative
- `1` → Positive

**Maximum sequence length:** 128

**Batch size:** 16

**Learning rate:** 2e-5

**Epochs:** 2

## 📊 Dataset

IMDb Movie Reviews:

- Training samples: 22,500
- Validation samples: 2,500
- Test samples: 25,000

The original IMDb dataset also contains an unsupervised split,
which is not used for supervised training.

## 📈 Results

| Metric | Score |
|---|---:|
| Accuracy | 87.73% |
| Precision | 88.03% |
| Recall | 87.34% |
| F1 Score | 87.68% |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 11,015 | 1,485 |
| Actual Positive | 1,583 | 10,917 |

## 🏗️ Project Structure

```text
NLP_Sentiment_Analysis/
│
├── data/
│
├── models/
│
├── notebooks/
│
├── results/
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── training_history.json
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── plot_results.py
│   ├── predict.py
│   └── train.py
│
├── app.py
├── .gitignore
├── README.md
└── requirements.txt
⚙️ Technologies
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
DistilBERT
Scikit-learn
Matplotlib
Streamlit
Google Colab GPU
🔄 Pipeline
IMDb Review
     ↓
DistilBERT Tokenizer
     ↓
Input IDs + Attention Mask
     ↓
DistilBERT
     ↓
Classification Head
     ↓
Sentiment Prediction
     ↓
Negative / Positive
💻 Running the Project

Install dependencies:

pip install -r requirements.txt

Run prediction:

python src/predict.py

Run evaluation:

python src/evaluate.py

Run the Streamlit application:

streamlit run app.py
🏆 Final Performance

The fine-tuned DistilBERT model achieved:

87.73% accuracy on 25,000 unseen IMDb test reviews.

This project demonstrates practical experience with transformer-based
NLP, PyTorch training pipelines, model evaluation, and deployment.
