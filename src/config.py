MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

EPOCHS = 2

NUM_LABELS = 2

RANDOM_SEED = 42

LABEL_NAMES = {
    0: "negative",
    1: "positive"
}

MODEL_SAVE_PATH = "models/distilbert-imdb"

RESULTS_DIR = "results"