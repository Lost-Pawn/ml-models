import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi

# Load dataset from Kaggle
api = KaggleApi()
api.authenticate()

dataset_path = "./data/cats-dogs-dataset/PetImages"

api.dataset_download_files(
    dataset='bhavikjikadara/dog-and-cat-classification-dataset',
    path=dataset_path,
    unzip=True
)

# Enable memory growth for GPUs to avoid OOM errors
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Load 80% training split
training_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    image_size=(150, 150),
    seed=123,
    batch_size=32 
)

# Load 20% validation split
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32 
)

class_names = training_dataset.class_names
print(class_names) # ['Cat', 'Dog'] -> 0 = Cat, 1 = Dog

# Shuffle -> prefetch for performance, ignore_errors drops corrupt image batches
training_dataset = (training_dataset
    .cache()
    .shuffle(5000) 
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    .ignore_errors()
)

validation_dataset = (validation_dataset
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE) 
    .ignore_errors()
)

# Augmentation layers 
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# CNN model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255, input_shape=(150, 150, 3)),  # normalize 0-255 to 0-1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),           
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary output: 0 = Cat, 1 = Dog
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=60,
    callbacks=[early_stop],
    verbose=1
) 

# Model stopped learning after 41 epochs, best weights restored from epoch 36

val_loss, val_acc = model.evaluate(validation_dataset)
print(f"Final validation accuracy: {val_acc:.4f} and loss: {val_loss:.4f}")
# accuracy: 0.8828, loss: 0.2735

# Save model
model.save('cat_dog_model.keras')