import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from config import IMG_HEIGHT, IMG_WIDTH, MODEL_SAVE_PATH
from model import CTCLayer
from utils import ctc_greedy_decode, normalize_image


def load_and_prep_image(path):
    # opens any image the user gives us, converts to grayscale and fits it
    # onto the same canvas size the model was trained on without stretching
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h))

    canvas = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)
    canvas.paste(img, (0, 0))
    arr = np.array(canvas)
    arr = normalize_image(arr)
    return np.expand_dims(arr, axis=-1)


def load_inference_model():
    # the saved file still has the CTC layer baked in from training, so it
    # needs the custom layer registered to load, then we cut it down to just
    # the image in, prediction out path for actual use
    full_model = keras.models.load_model(MODEL_SAVE_PATH, custom_objects={"CTCLayer": CTCLayer})
    image_input = full_model.get_layer("image").output
    y_pred = full_model.get_layer("dense_out").output
    return keras.Model(inputs=image_input, outputs=y_pred)


def predict_equation(image_path):
    model = load_inference_model()
    img = load_and_prep_image(image_path)
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
    text = ctc_greedy_decode(pred, [pred.shape[1]])[0]
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py path/to/equation_image.png")
        sys.exit(1)
    result = predict_equation(sys.argv[1])
    print(f"predicted equation: {result}")

    # RESULTS FROM ACTUAL RUN
    # tested on 4 freshly generated images the model had never seen
    # true "8*8=64" came back as "8*8=9", got the operator and first operand right
    # true "90-33=57" came back as "90-337", digits are close but merged wrong
    # true "87-24=63" came back as "8-24=5", lost a digit here
    # true "x+16=31" came back as "x+16=1", right up to the answer, lost a digit there too
    # matches the 22 percent exact match number from training, right structure
    # most of the time but the model still drops digits on longer results
