import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import tensorflow as tf

N_ROWS = 200_000
MAX_VOCAB_SIZE = 20_000
BATCH_SIZE = 256

with tempfile.TemporaryDirectory() as tmp_dir:
    api = KaggleApi()
    api.authenticate()
    print("Fetching dataset...")
    api.dataset_download_files('nishantsingh96/refined-bookcorpus-dataset', path=tmp_dir, unzip=False)

    file_path = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
    dataset = pd.read_csv(file_path, compression='infer')

df = dataset.iloc[:N_ROWS].copy()
print(df.head())

df = df.rename(columns={'0': "text"})

print(df.shape)
print(df.info())

data = df["text"].dropna().astype(str).tolist() 
print(len(data), type(data))

def data_preprocessing(data):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    input_sequences = []
    for seq in sequences:
        for i in range(1, len(seq)):
            n_gram_sequence = seq[:i + 1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    return X, y, tokenizer

X, y, tokenizer = data_preprocessing(data)
vocab_size = len(tokenizer.word_index) + 1

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(vocab_size, activation="softmax"),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
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

