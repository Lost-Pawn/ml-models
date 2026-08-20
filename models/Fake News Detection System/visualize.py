import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def plot_confusion_matrix(cm, path=None):
    path = path or config.CONFUSION_MATRIX_PATH
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["real", "fake"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("confusion matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top_words(vectorizer, model, top_n=15, path=None):
    path = path or config.TOP_WORDS_PATH
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_fake_idx = np.argsort(coefs)[-top_n:]  # pushes prediction toward fake
    top_real_idx = np.argsort(coefs)[:top_n]  # pushes prediction toward real

    fig, ax = plt.subplots(figsize=(8, 6))
    words = list(feature_names[top_real_idx]) + list(feature_names[top_fake_idx])
    values = list(coefs[top_real_idx]) + list(coefs[top_fake_idx])
    colors = ["#3b82f6"] * top_n + ["#ef4444"] * top_n

    ax.barh(words, values, color=colors)
    ax.set_xlabel("logistic regression coefficient")
    ax.set_title("strongest words pointing to real versus fake")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    from train import run_training
    from evaluate import evaluate_model
    from model import load_model
    from features import load_vectorizer

    _, _, X_test, y_test = run_training()
    model = load_model()
    vectorizer = load_vectorizer()
    results = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(results["confusion_matrix"])
    plot_top_words(vectorizer, model)
    # both files saved fine, the top words plot showed words like cover,
    # furious, trick and finally admits on the fake side and words like
    # according, officials, released and updated guidelines on the real
    # side, which lines up with how the templates were written in the
    # first place
    print("saved confusion_matrix.png and top_words.png to the outputs folder")
