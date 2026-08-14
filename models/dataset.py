import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tensorflow as tf

from config import (
    CHARS, FONT_PATHS, IMG_HEIGHT, IMG_WIDTH, MAX_LABEL_LEN, BATCH_SIZE
)
from utils import encode_label, normalize_image

random.seed(42)


def random_equation():
    # builds a short arithmetic or simple algebra equation that stays inside
    # the vocabulary and the max length, this stands in for real handwriting
    # samples since a labeled handwritten equation dataset was not available here
    kind = random.choice(["arith", "algebra"])
    if kind == "arith":
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            result = a + b
        elif op == "-":
            a, b = max(a, b), min(a, b)
            result = a - b
        else:
            a = random.randint(1, 12)
            b = random.randint(1, 12)
            result = a * b
        text = f"{a}{op}{b}={result}"
    else:
        var = random.choice(["x", "y"])
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        result = a + b
        text = f"{var}+{a}={result}"
    if len(text) > MAX_LABEL_LEN:
        return random_equation()
    return text


def render_equation_image(text):
    # renders the equation string with a random handwriting style font, then
    # applies rotation, offset jitter and blur so the same string never
    # produces the exact same pixels twice, a cheap stand in for real
    # handwriting variation between people
    font_size = random.randint(34, 44)
    font = ImageFont.truetype(random.choice(FONT_PATHS), font_size)

    canvas = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(canvas)
    x = random.randint(4, 12)
    y = random.randint(2, 10)
    draw.text((x, y), text, font=font, fill=0)

    angle = random.uniform(-4, 4)
    canvas = canvas.rotate(angle, fillcolor=255, expand=False)

    if random.random() < 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    arr = np.array(canvas, dtype=np.float32)
    noise = np.random.normal(0, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    return arr


def data_generator(num_samples):
    for _ in range(num_samples):
        text = random_equation()
        img = render_equation_image(text)
        img = normalize_image(img)
        img = np.expand_dims(img, axis=-1)  # add channel dim
        label, label_len = encode_label(text)
        yield img, label, label_len, text


def make_tf_dataset(num_samples, batch_size=BATCH_SIZE, shuffle=True):
    def gen():
        for img, label, label_len, _ in data_generator(num_samples):
            yield img, label, label_len

    output_signature = (
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(MAX_LABEL_LEN,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=512)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
