import numpy as np
from config import CHAR_TO_NUM, NUM_TO_CHAR, BLANK_INDEX, MAX_LABEL_LEN


def encode_label(text):
    # turns "12+7=19" into a fixed length array of class indices
    # padded with the blank index so every label has the same shape
    encoded = [CHAR_TO_NUM[c] for c in text]
    length = len(encoded)
    padded = encoded + [BLANK_INDEX] * (MAX_LABEL_LEN - length)
    return np.array(padded, dtype=np.int32), length


def ctc_greedy_decode(pred, input_lengths):
    # pred shape is (batch, time_steps, num_classes)
    # collapses repeated characters and drops blanks, the standard CTC decode rule
    decoded_texts = []
    pred_labels = np.argmax(pred, axis=-1)
    for i, seq in enumerate(pred_labels):
        seq = seq[: input_lengths[i]]
        chars = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != BLANK_INDEX:
                chars.append(NUM_TO_CHAR.get(idx, ""))
            prev = idx
        decoded_texts.append("".join(chars))
    return decoded_texts


def normalize_image(img_array):
    # scales pixel values from 0-255 to 0-1 so training is stable
    return img_array.astype(np.float32) / 255.0
