import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Cropping2D, Input, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

labels, counts = np.unique(y_train, return_counts=True)

# Visualize the distribution of classes in the dataset
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[y_train == i][0], cmap='gray')
    plt.title(f'Class {i}\nCount: {counts[i]}')
    plt.axis('off')
plt.savefig('fashion_mnist_classes.png') # dataset is balanced.

print(f"{x_train.shape[0]} training samples, {x_test.shape[0]} testing samples.")

# Visualize the distribution of pixel values in the dataset
print(x_train.min(), x_train.max(), x_train.mean())
plt.figure(figsize=(10, 4))
plt.hist(x_train.flatten(), bins=50, color='gray')
plt.title('Pixel Value Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('pixel_value_distribution.png')

print(f"Duplicate samples: {len(x_train) - len(np.unique(x_train, axis=0))}")

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

def noise(x, noise_factor=0.3):
    x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    x_noisy = np.clip(x_noisy, 0., 1.)
    return x_noisy

x_train_noisy = noise(x_train)
x_test_noisy = noise(x_test)

# Encoder
input_img = Input(shape=(28, 28, 1))                            # 28 * 28 * 1
x = Conv2D(32, 3, activation='relu', padding='same')(input_img) # 28 * 28 * 32
x = MaxPooling2D(2, padding='same')(x)                          # 14 * 14 * 32
x = Conv2D(16, 3, activation='relu', padding='same')(x)         # 14 * 14 * 16
x = MaxPooling2D(2, padding='same')(x)                          # 7 * 7 * 16
x = Conv2D(8, 3, activation='relu', padding='same')(x)          # 7 * 7 * 8
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)                  # 8 * 8 * 8
x = MaxPooling2D(2, padding='same')(x)                          # 4 * 4 * 8
encoded = Conv2D(4, 3, activation='relu', padding='same')(x)    # 4 * 4 * 4

# Decoder
x = UpSampling2D(2)(encoded)                                    # 8 * 8 * 4
x = Conv2D(8, 3, activation='relu', padding='same')(x)          # 8 * 8 * 8
x = Cropping2D(cropping=((0, 1), (0, 1)))(x)                    # 7 * 7 * 8
x = Conv2D(16, 3, activation='relu', padding='same')(x)         # 7 * 7 * 16
x = UpSampling2D(2)(x)                                          # 14 * 14 * 16
x = Conv2D(32, 3, activation='relu', padding='same')(x)         # 14 * 14 * 32
x = UpSampling2D(2)(x)                                          # 28 * 28 * 32
decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x) # 28 * 28 * 1

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
    )

model = Model(inputs=input_img, outputs=decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
history = model.fit(x_train_noisy, x_train,
                    epochs=80,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[early_stopping],
                    verbose=2
                    )

compression_ratio = 28 * 28 * 1 / (4 * 4 * 4)  
print(f"Compression ratio: {compression_ratio:.2f}")

print(f"History keys: {history.history.keys()}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_loss.png')

denoised_images = model.predict(x_test_noisy)

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(3, 10, i + 11)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title('Noisy')
    plt.axis('off')

    plt.subplot(3, 10, i + 21)
    plt.imshow(denoised_images[i].reshape(28, 28), cmap='gray')
    plt.title('Denoised')
    plt.axis('off')
plt.savefig('denoised_images.png')

# loss: 0.2843 - val_loss: 0.2865 at epoch 72
# compression ratio: 12.25