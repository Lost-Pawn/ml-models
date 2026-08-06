import pandas as pd
import numpy as np
import os
import sys

# librosa for audio loading and MFCC extraction
import librosa
import librosa.display as display

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# to play audio in notebook
import IPython.display as ipd
from IPython.display import Audio

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from kaggle.api.kaggle_api_extended import KaggleApi
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Loading the dataset
Ravdess = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
PATH = "./data"

api = KaggleApi()
api.authenticate()
api.dataset_download_files('uwrfkaggler/ravdess-emotional-speech-audio', path=PATH, unzip=True)
print("Dataset downloaded and extracted successfully.")

ravdess = os.path.join(PATH, "audio_speech_actors_01-24")
ravdess_dict = os.listdir(ravdess)
print(ravdess_dict)  # Check the content of the extracted directory

# Preprocessing the dataset
emotions = []
file_paths = []
for i in ravdess_dict:
    for file in os.listdir(os.path.join(ravdess, i)):
        if file.endswith(".wav"):
            parts = file.split("-")
            emotion = int(parts[2])  # Emotion is the third part of the filename
            emotions.append(emotion)
            file_paths.append(os.path.join(ravdess, i, file))

df = pd.DataFrame({"emotions": emotions, "file_paths": file_paths})
df['emotions'] = df['emotions'].map({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'})

print(df.head())  
print(df.tail())
print(df['emotions'].value_counts())  
print(df.isnull().sum())  
print(df.columns)

data, sr = librosa.load(df['file_paths'][0])
print(f"Audio data shape: {data.shape}, Sample rate: {sr}")

ipd.Audio(data, rate=sr)

# Mel log spectrogram visualization
plt.figure(figsize=(10, 4))
mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=8000)
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=sr, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()

mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()

# Data Augmentation Functions
def add_noise(data):
    noise_amp = np.random.uniform(0.002, 0.008)
    noise = np.random.randn(*data.shape)
    return data + noise_amp * noise

def shift_pitch(data, sr):
    n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def change_speed(data):
    speed = np.random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y=data, rate=speed)

def shift(data):
    shift_range = np.random.randint(-3000, 3000)
    shifted = np.zeros_like(data)
    if shift_range > 0:
        shifted[shift_range:] = data[:-shift_range]
    elif shift_range < 0:
        shifted[:shift_range] = data[-shift_range:]
    else:
        shifted = data.copy()
    return shifted

# Original Audio
print("Original Audio")
plt.figure(figsize=(10, 4))
librosa.display.waveshow(data, sr=sr)
plt.title('Original Audio')
plt.show()
ipd.Audio(data, rate=sr)

# Audio with Noise
x_noise = add_noise(data)
plt.figure(figsize=(10, 4))
librosa.display.waveshow(x_noise, sr=sr)
plt.title('Audio with Noise')
plt.show()
ipd.Audio(x_noise, rate=sr)

# Audio with Shifted Pitch
x_pitch = shift_pitch(data, sr)
plt.figure(figsize=(10, 4))
librosa.display.waveshow(x_pitch, sr=sr)
plt.title('Audio with Shifted Pitch')
plt.show()
ipd.Audio(x_pitch, rate=sr)

# Audio with changed speed
x_speed = change_speed(data)
plt.figure(figsize=(10,4))
librosa.display.waveshow(x_speed, sr=sr)
plt.title("Audio with Changed Speed")
plt.show()
ipd.Audio(x_speed, rate=sr)

# Audio with Shift
x_shift = shift(data)
plt.figure(figsize=(10, 4))
librosa.display.waveshow(x_shift, sr=sr)
plt.title("Audio with Shift")
plt.show()
ipd.Audio(x_shift, rate=sr)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["emotions"]
)

print(f"Training Samples : {len(train_df)}")
print(f"Testing Samples  : {len(test_df)}")

# Feature Extraction
MAX_PAD_LEN = 164
def extract_features(file_path, augment=False):
    data, sr = librosa.load(file_path, sr=None)
    
    if augment:
        augmentation = np.random.choice(["noise", "pitch", "speed", "shift", "none"])
        if augmentation == "noise":
            data = add_noise(data)
        elif augmentation == "pitch":
            data = shift_pitch(data, sr)
        elif augmentation == "speed":
            data = change_speed(data)
        elif augmentation == "shift":
            data = shift(data)

    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)

    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc.T

X_train = []
y_train = []

# Training Set 
for _, row in train_df.iterrows():
    # Original Sample
    X_train.append(extract_features(row["file_paths"], augment=False))
    y_train.append(row["emotions"])

    # Augmented Versions
    for _ in range(4):
        X_train.append(extract_features(row["file_paths"], augment=True))
        y_train.append(row["emotions"])

# Test Set
X_test = []
y_test = []

for _, row in test_df.iterrows():
    X_test.append(extract_features(row["file_paths"], augment=False))
    y_test.append(row["emotions"])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)

# Label Encoding
encoder = LabelEncoder()
encoder.fit(df["emotions"])

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Feature Scaling 
scaler = StandardScaler()
n_train, n_timesteps, n_features = X_train.shape
n_test = X_test.shape[0]

X_train = X_train.reshape(-1, n_features)
X_test = X_test.reshape(-1, n_features)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_scaled = X_train.reshape(n_train, n_timesteps, n_features)
X_test_scaled = X_test.reshape(n_test, n_timesteps, n_features)

print("Train Shape :", X_train_scaled.shape)
print("Test Shape  :", X_test_scaled.shape)
print("Mean :", X_train_scaled.mean(), "Std :", X_train_scaled.std())

# Model Building
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),

    tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2, strides=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2, strides=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(256, kernel_size=2, strides=1, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2, strides=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.3)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(y_train.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# Training
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=12, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100, 
    batch_size=64,
    shuffle=True,
    validation_data=(X_test_scaled, y_test),
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Evaluation
plt.figure(figsize=(12, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
