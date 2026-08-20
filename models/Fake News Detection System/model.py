import pickle
from sklearn.linear_model import LogisticRegression
import config


def train_model(X_train, y_train):
    model = LogisticRegression(C=config.C_VALUE, max_iter=config.MAX_ITER)
    model.fit(X_train, y_train)
    return model


def save_model(model, path=None):
    path = path or config.MODEL_PATH
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path=None):
    path = path or config.MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)
