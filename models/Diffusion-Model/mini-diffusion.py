import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(train_images, _), (test_images, _) = fashion_mnist.load_data()

# Forward diffusion process per time step
T = 1000
beta = np.linspace(0.0001, 0.02, T)
alphas = 1 - beta
alphas_cumprod = np.cumprod(alphas)

def forward_diffusion(x0, t):
    noise = np.random.normal(size=x0.shape).astype('float32')
    sqrt_alpha = np.sqrt(alphas_cumprod[t]).reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha = np.sqrt(1 - alphas_cumprod[t]).reshape(-1, 1, 1, 1)
    xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    return xt, t, noise

# Visualize the forward diffusion process for a sample image
x0 = train_images[69:70].astype('float32') / 255.0
x0 = x0.reshape(-1, 28, 28, 1)

t_values = [0, 200, 400, 600, 800, 999]  

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

# sinusoidal time embedding model
class SinusoidalTimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim=32, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, t):
        t = tf.cast(t, tf.float32)
        t = tf.reshape(t, (-1, 1))
        frequencies_arr = tf.range(self.dim // 2, dtype=tf.float32)
        inv_freq = 1 / (10000 ** (2 * frequencies_arr / self.dim))
        sin_pos = tf.sin(t * inv_freq)
        cos_pos = tf.cos(t * inv_freq)
        return tf.keras.layers.Concatenate(axis=-1)([sin_pos, cos_pos])
    

# U-Net architecture for the diffusion model
def mini_diffusion_model(dim=32):
    # Input layers
    input_img = tf.keras.Input(shape=(28, 28, 1))
    input_t = tf.keras.Input(shape=(), dtype=tf.int32)

    # time embedding layer
    t_sinu = SinusoidalTimeEmbedding(dim=dim)(input_t)
    x_t = tf.keras.layers.Dense(dim // 2, activation="swish")(t_sinu)
    t_emb = tf.keras.layers.Dense(dim, activation="swish")(x_t)

    # Down block 1
    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
    t_proj1 = tf.keras.layers.Dense(32)(t_emb)
    t_proj1 = tf.keras.layers.Reshape((1, 1, 32))(t_proj1)
    x = x + t_proj1
    skip1 = x
    x = tf.keras.layers.MaxPooling2D(2)(x)

    # Down block 2
    x = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    t_proj2 = tf.keras.layers.Dense(64)(t_emb)
    t_proj2 = tf.keras.layers.Reshape((1, 1, 64))(t_proj2)
    x = x + t_proj2
    skip2 = x
    x = tf.keras.layers.MaxPooling2D(2)(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    t_proj3 = tf.keras.layers.Dense(128)(t_emb)
    t_proj3 = tf.keras.layers.Reshape((1, 1, 128))(t_proj3)
    x = x + t_proj3

    # Up block 1
    x = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip2])
    x = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    t_proj4 = tf.keras.layers.Dense(64)(t_emb)
    t_proj4 = tf.keras.layers.Reshape((1, 1, 64))(t_proj4)
    x = x + t_proj4

    # Up block 2
    x = tf.keras.layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip1])
    x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
    t_proj5 = tf.keras.layers.Dense(32)(t_emb)
    t_proj5 = tf.keras.layers.Reshape((1, 1, 32))(t_proj5)
    x = x + t_proj5

    # Output layer
    output = tf.keras.layers.Conv2D(1, (1,1), activation=None, padding="same")(x)
    return tf.keras.Model(inputs=[input_img, input_t], outputs=output)

model = mini_diffusion_model(dim=32)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()

# Training loop
BATCH_SIZE = 1024
EPOCHS = 30
train_images = train_images.astype('float32') / 255.0
train_images = np.expand_dims(train_images, axis=-1)
best_loss = float('inf')

for epoch in range(EPOCHS):
    epoch_losses = []
    np.random.shuffle(train_images)

    for i in range(0, len(train_images), BATCH_SIZE):
        x0_batch = train_images[i:i+BATCH_SIZE]
        t_batch = np.random.randint(0, T, size=(x0_batch.shape[0],))
        xt_batch, _, noise_batch = forward_diffusion(x0_batch, t_batch)

        with tf.GradientTape() as tape:
            noise_pred = model([xt_batch, t_batch], training=True)
            loss_value = loss(noise_batch, noise_pred)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss_value.numpy()}")
    epoch_losses.append(loss_value.numpy())

    # save the best model based on the lowest loss
    if np.mean(epoch_losses) < best_loss:
        best_loss = np.mean(epoch_losses)
        model.save("mini_diffusion_model.keras")

# evaluation and visualization of the reverse diffusion process
def ddpm_sample(model, T, num_images=1, image_shape=(28, 28, 1), save_intermediate_every=None):
    x = np.random.normal(size=(num_images, *image_shape)).astype('float32')

    intermediates = {}
    for t in reversed(range(T)):
        t_batch = np.full((num_images,), t, dtype=np.int32)
        noise_pred = model.predict([x, t_batch])

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = beta[t]

        coef = beta_t / np.sqrt(1 - alpha_cumprod_t)
        mean = (1/np.sqrt(alpha_t)) * (x - coef * noise_pred)

        if t > 0:
            sigma_t = np.sqrt(beta_t)
            noise = np.random.normal(size=x.shape).astype('float32')
            x = mean + sigma_t * noise
        else:
            x = mean

        if save_intermediate_every is not None and t % save_intermediate_every == 0:
            intermediates[t] = x.copy()
    return x, intermediates

final_samples, intermediates = ddpm_sample(model, T, num_images=1, save_intermediate_every= T // 5)

plt.figure(figsize=(15, 4))
for idx, (t_val, img) in enumerate(intermediates.items()):
    plt.subplot(1, len(intermediates), idx + 1)
    plt.imshow(img[0].reshape(28, 28), cmap='gray')
    plt.title(f"t = {t_val}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("reverse_diffusion_process.png")

final_image = final_samples[0].reshape(28, 28)
plt.figure(figsize=(4, 4))
plt.imshow(final_image, cmap='gray')
plt.title("Final Sampled Image")
plt.axis('off')
plt.savefig("final_sampled_image.png")