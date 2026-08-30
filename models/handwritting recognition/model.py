import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, MAX_LABEL_LEN


class CTCLayer(layers.Layer):
    # computes the CTC loss inside the model itself so training just needs
    # (images, labels) as input and the loss shows up automatically
    def call(self, y_true, y_pred, label_length):
        batch_len = tf.shape(y_pred)[0]
        input_len = tf.shape(y_pred)[1]
        input_length = tf.fill([batch_len, 1], input_len)
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, tf.expand_dims(label_length, -1))
        self.add_loss(tf.reduce_mean(loss))
        return y_pred


def build_model():
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    label_input = layers.Input(shape=(MAX_LABEL_LEN,), name="label", dtype="int32")
    label_length_input = layers.Input(shape=(), name="label_length", dtype="int32")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(image_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)

    # collapse the height dimension so what's left is a sequence along width,
    # this is the step that turns a 2d image into a sequence a BiLSTM can read
    # x comes in as (height, width, channels), permute so width becomes the
    # time axis before flattening height and channels together
    x = layers.Permute((2, 1, 3))(x)
    new_shape = (IMG_WIDTH // 4, (IMG_HEIGHT // 16) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    y_pred = layers.Dense(NUM_CLASSES, activation="softmax", name="dense_out")(x)

    output = CTCLayer(name="ctc_loss")(label_input, y_pred, label_length_input)

    model = keras.Model(
        inputs=[image_input, label_input, label_length_input],
        outputs=output,
        name="equation_crnn",
    )
    return model


def build_inference_model(training_model):
    # strips away the label inputs and CTC layer, leaving just image in,
    # character probabilities out, this is what gets used for predictions
    image_input = training_model.get_layer("image").output
    y_pred = training_model.get_layer("dense_out").output
    return keras.Model(inputs=image_input, outputs=y_pred, name="equation_crnn_inference")
