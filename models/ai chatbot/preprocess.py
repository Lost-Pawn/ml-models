import json
import re

# This file only prepares data. It does not train or run anything on its own.
# Keeping it separate means train_model.py and chatbot.py can both reuse the
# exact same cleaning logic, so the model always sees text in the same format
# whether it is learning from it or predicting on it. This is important:
# if training and inference used different cleaning rules, the vectorizer
# would see slightly different vocabulary at prediction time and accuracy
# would silently get worse.


def load_intents(path="intents.json"):
    # intents.json holds every conversation topic the bot understands, along
    # with example phrases (patterns) and possible replies (responses).
    with open(path, "r") as f:
        data = json.load(f)
    return data["intents"]


def clean_text(text):
    """
    Normalize raw user text into a simple, consistent format before it is
    turned into TF-IDF features.

    Steps:
    1. Handle non-string / empty input safely so the chatbot never crashes
       on unusual input (None, numbers, empty strings, whitespace-only).
    2. Lowercase everything so "Hello" and "hello" are treated as the same
       word.
    3. Remove punctuation, but keep apostrophes in contractions (don't,
       can't, isn't) before removing the apostrophe itself, by expanding a
       few common negation contractions first. This matters because words
       like "not", "no", and "never" flip the meaning of a sentence, and we
       don't want cleaning to accidentally destroy that signal (e.g. turning
       "i don't need help" into something that reads like "i need help").
    4. Collapse repeated whitespace into single spaces and strip the ends.
    5. Always return a string, even for empty/garbage input, so the caller
       can safely check `if not cleaned:` without extra type checks.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Expand a handful of common negation contractions BEFORE stripping
    # punctuation, so "don't" becomes "do not" instead of turning into
    # "dont" (which is still fine) or losing meaning some other way. This
    # keeps the negation word itself intact as a separate token.
    negation_expansions = {
        r"\bwon't\b": "will not",
        r"\bcan't\b": "cannot",
        r"\bcannot\b": "cannot",
        r"\bn't\b": " not",
    }
    for pattern, replacement in negation_expansions.items():
        text = re.sub(pattern, replacement, text)

    # Remove anything that isn't a letter, digit, or whitespace. This drops
    # punctuation like "!", "?", "," while leaving words (including "not",
    # "no", "never") completely untouched.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces/tabs/newlines into a single space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_dataset(intents):
    # Turns the nested intents.json structure into three flat pieces:
    # texts (every example sentence), labels (the intent each one belongs to),
    # and responses (a lookup table of possible replies per intent).
    texts = []
    labels = []
    responses = {}

    for intent in intents:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            cleaned = clean_text(pattern)
            if cleaned:  # skip any pattern that cleans down to nothing
                texts.append(cleaned)
                labels.append(tag)

    return texts, labels, responses
