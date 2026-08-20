from sklearn.model_selection import train_test_split
import config
from preprocess import load_and_clean
from features import fit_vectorizer, save_vectorizer
from model import train_model, save_model


def run_training():
    df = load_and_clean()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )

    vectorizer = fit_vectorizer(X_train_text)
    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = train_model(X_train, y_train)

    save_vectorizer(vectorizer)
    save_model(model)

    # 6400 rows went into training and 1600 into the held out test set,
    # matches the 80 20 split defined in config
    print(f"train size, {X_train.shape[0]}, test size, {X_test.shape[0]}")

    return model, vectorizer, X_test, y_test


if __name__ == "__main__":
    run_training()
