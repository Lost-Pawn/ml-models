import os

# every character the model is allowed to output
# blank token for CTC is added automatically as the last index
CHARS = list("0123456789+-*/=()xy^.")
NUM_CLASSES = len(CHARS) + 1  # +1 for the CTC blank

CHAR_TO_NUM = {c: i for i, c in enumerate(CHARS)}
NUM_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
BLANK_INDEX = len(CHARS)

IMG_HEIGHT = 64
IMG_WIDTH = 256
MAX_LABEL_LEN = 12  # longest equation string we generate

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_PATHS = [
    os.path.join(FONT_DIR, "Kalam-Regular.ttf"),
    os.path.join(FONT_DIR, "PatrickHand-Regular.ttf"),
    os.path.join(FONT_DIR, "GochiHand-Regular.ttf"),
    os.path.join(FONT_DIR, "Caveat[wght].ttf"),
]

BATCH_SIZE = 32
EPOCHS = 14
TRAIN_SAMPLES = 3200
VAL_SAMPLES = 500
LEARNING_RATE = 1e-3

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "equation_crnn.keras")
