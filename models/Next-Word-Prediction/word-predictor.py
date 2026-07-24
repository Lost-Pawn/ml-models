import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")
if not gpus:
    print("no GPU available.")

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# hyperparameters
N_ROWS = 200_000
MAX_VOCAB_SIZE = 20_000
MAX_SEQ_LEN = 60
BATCH_SIZE = 512

# load dataset
with tempfile.TemporaryDirectory() as tmp_dir:
    api = KaggleApi()
    api.authenticate()
    print("Fetching dataset...")
    api.dataset_download_files('nishantsingh96/refined-bookcorpus-dataset', path=tmp_dir, unzip=False)

    file_path = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
    dataset = pd.read_csv(file_path, compression='infer')

# eda
df = dataset.iloc[:N_ROWS].copy()
print(df.head())

df = df.rename(columns={'0': "text"})

print(df.shape)
print(df.info())

data = df["text"].dropna().astype(str).tolist()
print(len(data), type(data))

# data preprocessing
# tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(data)
vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)


# create n-gram sequences
def build_ngram_arrays(sentences, tokenizer, max_sequence_len):
    all_sequences = tokenizer.texts_to_sequences(sentences)

    inputs, targets = [], []
    for seq in all_sequences:
        for i in range(1, len(seq)):
            inputs.append(seq[:i])   # context
            targets.append(seq[i])   # next token

    if not inputs:
        return np.empty((0, max_sequence_len - 1), dtype=np.int32), np.empty((0,), dtype=np.int32)

    X = tf.keras.preprocessing.sequence.pad_sequences(
        inputs,
        maxlen=max_sequence_len - 1,
        padding="post",
        truncating="pre",
    )
    y = np.array(targets, dtype=np.int32)
    return X.astype(np.int32), y


text_train, text_test = train_test_split(data, test_size=0.2, random_state=42)

X_train, y_train = build_ngram_arrays(text_train, tokenizer, MAX_SEQ_LEN)
X_test, y_test = build_ngram_arrays(text_test, tokenizer, MAX_SEQ_LEN)

train_ds = (tf.data.Dataset
            .from_tensor_slices((X_train, y_train))
            .shuffle(10000)
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(tf.data.AUTOTUNE))

test_ds = (tf.data.Dataset
           .from_tensor_slices((X_test, y_test))
           .batch(BATCH_SIZE)
           .prefetch(tf.data.AUTOTUNE))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(vocab_size, activation="softmax", dtype="float32"),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
    jit_compile=True,   # cuDNN optimization for faster training
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=100,
    callbacks=[early_stopping],
)

model.summary()

loss, accuracy = model.evaluate(test_ds)
print(f"Model evaluation completed. Loss: {loss}, Accuracy: {accuracy}")

model.save("next_word_prediction.keras")

# plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# beam search
def beam_search(model, tokenizer, seed_text, beam_width=3, n_words=20, context_len=MAX_SEQ_LEN - 1):
    sequences = [[seed_text, 0.0]]
    for _ in range(n_words):
        all_candidates = []
        for seq, score in sequences:
            token_list = tokenizer.texts_to_sequences([seq])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences(
                [token_list], maxlen=context_len, padding="post", truncating="pre"
            )
            predictions = model(token_list, training=False)
            top_indices = np.argsort(predictions[0])[-beam_width:]
            for index in top_indices:
                if index == 0:
                    continue
                word = tokenizer.index_word.get(index)
                if word is None:
                    continue
                candidate = [seq + " " + word, score - np.log(predictions[0][index] + 1e-9)]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]
    return sequences