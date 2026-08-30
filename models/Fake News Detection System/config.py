import os

# base folder is wherever this project sits, keeps paths portable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "news_dataset.csv")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")

CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
TOP_WORDS_PATH = os.path.join(OUTPUT_DIR, "top_words.png")

# how many synthetic articles to generate per class, 4000 each gave a good
# balance between training speed and having enough vocabulary to learn from
SAMPLES_PER_CLASS = 4000

# split sizes
TEST_SIZE = 0.2
RANDOM_STATE = 42

# tfidf settings, unigrams and bigrams together worked noticeably better
# than unigrams alone during testing
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

# logistic regression settings
C_VALUE = 1.0
MAX_ITER = 1000

for folder in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)
