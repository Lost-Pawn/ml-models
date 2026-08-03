import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(train_images, _), (test_images, _) = fashion_mnist.load_data()

# Forward diffusion process per time step
T = 300
beta = np.linspace(0.0001, 0.02, T)
alphas = 1 - beta
alphas_cumprod = np.cumprod(alphas)

def forward_diffusion(x0, t):
    noise = np.random.normal(size=x0.shape)
    xt = np.sqrt(alphas_cumprod[t]) * x0 + np.sqrt(1 - alphas_cumprod[t]) * noise
    return xt, t, noise

# Visualize the forward diffusion process for a sample image
x0 = train_images[0:1].astype('float32') / 255.0
x0 = x0.reshape(-1, 28, 28, 1)

t_values = [0, 50, 100, 200, 299]

plt.figure(figsize=(15, 4))
for idx, t_val in enumerate(t_values):
    xt, t, noise = forward_diffusion(x0, t_val)
    plt.subplot(1, len(t_values), idx + 1)
    plt.imshow(xt.reshape(28, 28), cmap='gray')
    plt.title(f"t = {t_val}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("forward_diffusion_process.png")
plt.show()

# sinusoidal time embedding function
def get_time_embedding(t, dim=32):
    t = tf.cast(t, tf.float32)
    batch = t.shape[0]
    t = tf.reshape(t, (batch, 1))

    frequencies_arr = tf.range(dim // 2, dtype=tf.float32)
    inv_freq = 1 / (10000 ** (2 * frequencies_arr / dim))

    sin_pos = tf.sin(t * inv_freq)
    cos_pos = tf.cos(t * inv_freq)

    input = tf.keras.layers.Concatenate(axis=-1)([sin_pos, cos_pos])
    x = tf.keras.layers.Dense(dim // 2, activation="swish")(input)
    t_embedding = tf.keras.layers.Dense(dim, activation="swish")(x)
    return t_embedding

# U-Net architecture for the diffusion model

# Input layers
input_img = tf.keras.Input(shape=(28, 28, 1))
input_t_emb = tf.keras.Input(shape=(32,))

# Down block 1
x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
t_proj1 = tf.keras.layers.Dense(32)(input_t_emb)
t_proj1 = tf.reshape(t_proj1, (-1, 1, 1, 32))
x = x + t_proj1
skip1 = x
x = tf.keras.layers.MaxPooling2D(2)(x)

# Down block 2
x = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
t_proj2 = tf.keras.layers.Dense(64)(input_t_emb)
t_proj2 = tf.reshape(t_proj2, (-1, 1, 1, 64))
x = x + t_proj2
skip2 = x
x = tf.keras.layers.MaxPooling2D(2)(x)

# Bottleneck
x = tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
t_proj3 = tf.keras.layers.Dense(128)(input_t_emb)
t_proj3 = tf.reshape(t_proj3, (-1, 1, 1, 128))
x = x + t_proj3

# Up block 1
x = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)
x = tf.keras.layers.Concatenate(axis=-1)([x, skip2])
x = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
t_proj4 = tf.keras.layers.Dense(64)(input_t_emb)
t_proj4 = tf.reshape(t_proj4, (-1, 1, 1, 64))
x = x + t_proj4

# Up block 2
x = tf.keras.layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)
x = tf.keras.layers.Concatenate(axis=-1)([x, skip1])
x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
t_proj5 = tf.keras.layers.Dense(32)(input_t_emb)
t_proj5 = tf.reshape(t_proj5, (-1, 1, 1, 32))
x = x + t_proj5

# Output layer
output = tf.keras.layers.Conv2D(1, (1,1), activation=None, padding="same")(x)

model = tf.keras.Model(inputs=[input_img, input_t_emb], outputs=output)
model.summary()


