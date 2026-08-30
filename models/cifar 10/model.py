from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout,
    BatchNormalization, RandomFlip, RandomTranslation,
)

INPUT_SHAPE = (32, 32, 3)


def build_cnn(input_shape=INPUT_SHAPE, num_classes=10):
    """Small CNN for CIFAR-10.

    One conv layer per block instead of two - with only 50k training
    images two stacked convs per block was overkill and just memorized
    the training set faster than it generalized. Augmentation lives
    inside the model now (RandomFlip/RandomTranslation) so it's part of
    the graph and automatically switches off at inference/eval time.
    """
    model = Sequential()

    model.add(RandomFlip('horizontal', input_shape=input_shape))
    model.add(RandomTranslation(0.1, 0.1))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    # GAP instead of Flatten - fewer params going into the dense head,
    # which was the other big source of overfitting here
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def build_densenet121(input_shape=INPUT_SHAPE, num_classes=10):
    """DenseNet121 baseline, for comparing against the small CNN above."""
    from tensorflow.keras.applications import DenseNet121

    base = DenseNet121(input_shape=input_shape, include_top=False,
                        weights='imagenet', pooling='avg')

    model = Sequential()
    model.add(base)
    model.add(Dense(num_classes, activation='softmax'))

    return model
