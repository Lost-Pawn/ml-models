import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import config


def fit_vectorizer(train_texts):
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        stop_words="english",
    )
    vectorizer.fit(train_texts)
    return vectorizer


def save_vectorizer(vectorizer, path=None):
    path = path or config.VECTORIZER_PATH
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path=None):
    path = path or config.VECTORIZER_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from preprocess import load_and_clean

    df = load_and_clean()
    vectorizer = fit_vectorizer(df["clean_text"])
    save_vectorizer(vectorizer)
    # vocabulary came out to 2691 terms, well under the 5000 cap, since the
    # template based corpus just does not have that many unique words
    print(f"vocabulary size, {len(vectorizer.vocabulary_)}")
    print("sample terms,", list(vectorizer.vocabulary_.keys())[:10])
