import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# STEP 1: Load the CIFAR-10 dataset
# 60,000 32x32 color images across 10 classes (50k train / 10k test).
# ---------------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"X_train shape: {X_train.shape}")   # (50000, 32, 32, 3)
print(f"y_train shape: {y_train.shape}")   # (50000, 1)
print(f"X_test shape: {X_test.shape}")     # (10000, 32, 32, 3)
print(f"y_test shape: {y_test.shape}")     # (10000, 1)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------------------------------------------------------------------
# STEP 2: Quick visual sanity check - plot a 10x10 grid of random train images
# ---------------------------------------------------------------------------
W_grid, L_grid = 10, 10
fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
axes = axes.ravel()
n_train = len(X_train)
for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0, n_train)
    axes[i].imshow(X_train[index, 1:])
    axes[i].set_title(labels[int(y_train[index])], fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

# ---------------------------------------------------------------------------
# STEP 3: Check class balance for train and test sets (bar chart)
# ---------------------------------------------------------------------------
classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

classes, counts = np.unique(y_train, return_counts=True)
plt.figure()
plt.barh(classes_name, counts)
plt.title('Class distribution in training set')

classes, counts = np.unique(y_test, return_counts=True)
plt.figure()
plt.barh(classes_name, counts)
plt.title('Class distribution in testing set')
# Result: classes are evenly distributed (6000 train / 1000 test images each) - no need for class balancing.

# ---------------------------------------------------------------------------
# STEP 4: Preprocessing
# Normalize pixel values to [0, 1] and one-hot encode the labels for softmax training.
# ---------------------------------------------------------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# ---------------------------------------------------------------------------
# STEP 5: Build the CNN
# 3 conv blocks (32 -> 64 -> 128 filters), each block = 2x(Conv2D + BatchNorm),
# followed by MaxPool + Dropout for regularization, then a Dense head.
# ---------------------------------------------------------------------------
INPUT_SHAPE = (32, 32, 3)
KERNEL_SIZE = (3, 3)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))  # 10-way classification

METRICS = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)

model.summary()

# Optional early stopping on val_loss (kept but not passed into fit, same as original notebook)
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# ---------------------------------------------------------------------------
# STEP 6: Data augmentation + training
# Shift/flip augmentation to reduce overfitting since CIFAR-10 is a small dataset.
# ---------------------------------------------------------------------------
batch_size = 32
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, y_cat_train, batch_size)
steps_per_epoch = X_train.shape[0] // batch_size

r = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_test, y_cat_test),
)

# ---------------------------------------------------------------------------
# STEP 7: Plot training curves (loss, accuracy, precision, recall)
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 16))

plt.subplot(4, 2, 1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='val_Loss')
plt.title('Loss Function Evolution')
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy Function Evolution')
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(r.history['precision'], label='precision')
plt.plot(r.history['val_precision'], label='val_precision')
plt.title('Precision Function Evolution')
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(r.history['recall'], label='recall')
plt.plot(r.history['val_recall'], label='val_recall')
plt.title('Recall Function Evolution')
plt.legend()

# ---------------------------------------------------------------------------
# STEP 8: Evaluate on the test set + confusion matrix + classification report
# ---------------------------------------------------------------------------
evaluation = model.evaluate(X_test, y_cat_test)
print(f'Test Accuracy : {evaluation[1] * 100:.2f}%')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(10, 10))
disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')
plt.show()

print(classification_report(y_test, y_pred))  # per-class precision/recall/f1

# ---------------------------------------------------------------------------
# STEP 9: Sanity check on a single image (index 100 - a Deer)
# ---------------------------------------------------------------------------
my_image = X_test[100]
plt.imshow(my_image)
print(f" Image 100 is {y_test[100]}")

pred_100 = np.argmax(model.predict(my_image.reshape(1, 32, 32, 3)))
print(f"The model predict that image 100 is {pred_100}")

# ---------------------------------------------------------------------------
# STEP 10: Visualize predictions on a 5x5 grid of random test images
# ---------------------------------------------------------------------------
W_grid, L_grid = 5, 5
fig, axes = plt.subplots(L_grid, W_grid, figsize=(17, 17))
axes = axes.ravel()
n_test = len(X_test)
for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0, n_test)
    axes[i].imshow(X_test[index, 1:])
    axes[i].set_title(labels[int(y_pred[index])], fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

# ---------------------------------------------------------------------------
# STEP 11: Helper plots - image with predicted label, and confidence bar chart
# blue = correct prediction, red = incorrect prediction
# ---------------------------------------------------------------------------
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{labels[int(predicted_label)]} {100*np.max(predictions_array):2.0f}% ({labels[int(true_label)]})",
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, int(true_label[i])
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Render first 40 test images side-by-side with their prediction confidence bars
predictions = model.predict(X_test)
num_rows, num_cols = 8, 5
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], y_test, X_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# STEP 12: Try a DenseNet121 transfer-learning model as a comparison baseline
# ---------------------------------------------------------------------------
from keras.applications.densenet import DenseNet121
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
base_model = DenseNet121(input_shape=(32, 32, 3), include_top=False, weights='imagenet', pooling='avg')
model.add(base_model)
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

r = model.fit(
    train_generator,
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_test, y_cat_test),
)

# ---------------------------------------------------------------------------
# STEP 13: Save the trained model to disk
# ---------------------------------------------------------------------------
from tensorflow.keras.models import load_model

model.save('cnn_20_epochs.h5')
