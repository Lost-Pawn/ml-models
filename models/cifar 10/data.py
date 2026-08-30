import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']


def load_cifar10():
    """Load CIFAR-10, scale pixels to [0, 1], one-hot encode labels.

    Returns (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat).
    The plain y_train/y_test are kept around for the confusion matrix later,
    since sklearn wants integer labels, not one-hot.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)


def class_counts(y):
    """Quick check that classes are balanced before training."""
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(classes, counts))
