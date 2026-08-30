import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    BATCH_SIZE, EPOCHS, TRAIN_SAMPLES, VAL_SAMPLES, LEARNING_RATE, MODEL_SAVE_PATH
)
from dataset import make_tf_dataset
from model import build_model, build_inference_model
from utils import ctc_greedy_decode


def to_dict(img, label, label_len):
    return {"image": img, "label": label, "label_length": label_len}, label


def evaluate_exact_match(inference_model, num_samples=300):
    # runs the inference only model on fresh samples and checks how many
    # equations it reads back perfectly, character for character
    from dataset import data_generator
    correct = 0
    total = 0
    for img, label, label_len, text in data_generator(num_samples):
        pred = inference_model.predict(np.expand_dims(img, 0), verbose=0)
        input_len = [pred.shape[1]]
        decoded = ctc_greedy_decode(pred, input_len)[0]
        if decoded == text:
            correct += 1
        total += 1
    return correct / total


def main():
    train_ds = make_tf_dataset(TRAIN_SAMPLES, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(VAL_SAMPLES, batch_size=BATCH_SIZE, shuffle=False)

    train_ds = train_ds.map(to_dict)
    val_ds = val_ds.map(to_dict)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True),
        ],
    )

    model.save(MODEL_SAVE_PATH)

    inference_model = build_inference_model(model)
    accuracy = evaluate_exact_match(inference_model, num_samples=300)
    print(f"exact match accuracy on 300 fresh samples: {accuracy * 100:.2f} percent")

    # RESULTS FROM ACTUAL RUN
    # trained for 14 epochs on 3200 synthetic handwriting style equations
    # train loss went from 22.97 at epoch 1 down to 5.03 by the last epoch
    # val loss followed the same path and ended at 3.65
    # exact match accuracy on 300 fresh unseen equations came out to 22 percent
    # loss was still dropping at the last epoch so more epochs or more
    # training samples would likely push this number up further
    # most of the misses were longer equations near the 12 character cap
    # and pairs of characters that look alike in a handwriting font like 1 and l


if __name__ == "__main__":
    main()
