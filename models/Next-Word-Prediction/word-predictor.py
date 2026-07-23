import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import tensorflow as tf


# hyperparameters
N_ROWS = 200_000
MAX_VOCAB_SIZE = 20_000
MAX_SEQ_LEN = 60
BATCH_SIZE = 256

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

# for global max length of sequences
all_sequences = tokenizer.texts_to_sequences(data)
max_sequence_len = max(len(seq) for seq in all_sequences)
max_sequence_len = min(MAX_SEQ_LEN, max_sequence_len)  

def sequences_generator(sentences, tokenizer, max_sequence_len):
    for sentence in sentences:
        seq = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(seq)):
            n_gram_sequence = seq[:i + 1]
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [n_gram_sequence], 
                maxlen=max_sequence_len, 
                padding="pre",
                # truncating="pre"
            )
            yield padded[0, :-1], padded[0, -1] # all rows except last column as X, last column as y


text_train, text_test = train_test_split(data, 
                                        test_size=0.2, 
                                        random_state=42, 
                                        shuffle=True
                                        )

output_signature = (
    tf.TensorSpec(shape=(max_sequence_len - 1,), dtype=tf.int32),
    tf.TensorSpec(shape=(), dtype=tf.int32),
)
    
train_ds = (tf.data.Dataset
            .from_generator(lambda: sequences_generator(text_train, tokenizer, max_sequence_len),
                            output_signature=output_signature
                            )
                            .shuffle(10000)
                            .batch(BATCH_SIZE)
                            .prefetch(tf.data.AUTOTUNE))

test_ds = (tf.data.Dataset
           .from_generator(lambda: sequences_generator(text_test, tokenizer, max_sequence_len), 
                           output_signature=output_signature
                           )
           .batch(BATCH_SIZE)
           .prefetch(tf.data.AUTOTUNE))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
    tf.keras.layers.LSTM(128, return_sequences=True),
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

# plot training history
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# beam search
def beam_search(model, tokenizer, seed_text, beam_width=3, max_sequence_len=20):
    sequences = [[seed_text, 0.0]]
    for _ in range(max_sequence_len):
        all_candidates = []
        for seq, score in sequences:
            token_list = tokenizer.texts_to_sequences([seq])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences(
                [token_list], maxlen=max_sequence_len - 1, padding="pre"
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


