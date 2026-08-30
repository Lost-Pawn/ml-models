import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from data import LABELS


def plot_sample_grid(x, y, rows=10, cols=10):
    """10x10 grid of random images with their labels, just to eyeball the data."""
    fig, axes = plt.subplots(rows, cols, figsize=(17, 17))
    axes = axes.ravel()
    for ax in axes:
        idx = np.random.randint(0, len(x))
        ax.imshow(x[idx])
        ax.set_title(LABELS[int(y[idx])], fontsize=8)
        ax.axis('off')
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def plot_history(history):
    """Loss/accuracy/precision/recall curves, train vs val."""
    metrics = [
        ('loss', 'val_loss', 'Loss'),
        ('accuracy', 'val_accuracy', 'Accuracy'),
        ('precision', 'val_precision', 'Precision'),
        ('recall', 'val_recall', 'Recall'),
    ]
    plt.figure(figsize=(12, 10))
    for i, (train_key, val_key, title) in enumerate(metrics, start=1):
        if train_key not in history.history:
            continue
        plt.subplot(2, 2, i)
        plt.plot(history.history[train_key], label=train_key)
        plt.plot(history.history[val_key], label=val_key)
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_and_report(model, x_test, y_test):
    """Confusion matrix + classification report on the test set."""
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(
        xticks_rotation='vertical', ax=ax, cmap='summer'
    )
    plt.show()

    print(classification_report(y_test, y_pred, target_names=LABELS))
    return y_pred


def plot_predictions_grid(x_test, y_test, y_pred, rows=5, cols=5):
    """Random sample of predictions, titled with the predicted class."""
    fig, axes = plt.subplots(rows, cols, figsize=(17, 17))
    axes = axes.ravel()
    for ax in axes:
        idx = np.random.randint(0, len(x_test))
        ax.imshow(x_test[idx])
        ax.set_title(LABELS[int(y_pred[idx])], fontsize=8)
        ax.axis('off')
    plt.subplots_adjust(hspace=0.4)
    plt.show()
