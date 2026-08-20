"""
Simple text cleaning functions for resume text. Both train_model.py and
predict.py import clean_text from here so the exact same cleaning is
applied every time, during training and during prediction. If cleaning
was done differently in each place the model would end up seeing
different kinds of text than what it was trained on, and the accuracy
would drop for no obvious reason.
"""

import re

# A short list of common English stopwords. A full NLTK download isn't
# used here on purpose, to keep this project light and free of extra
# downloads that might not be available in every environment.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "in", "on", "at", "to", "for", "of", "with", "by",
    "this", "that", "these", "those", "it", "its", "as", "from", "such",
    "have", "has", "had", "i", "we", "they", "he", "she", "his", "her"
}


def clean_text(text):
    text = text.lower()

    # remove anything that is not a letter or a space
    text = re.sub(r"[^a-z\s]", " ", text)

    # collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return " ".join(words)


if __name__ == "__main__":
    sample = "Data Scientist with 3 years of experience, skilled in Python and SQL!"
    print("Original: ", sample)
    print("Cleaned:  ", clean_text(sample))

    # Result after running this file directly, the sample line got
    # turned into, data scientist years experience skilled python sql
    # the punctuation, numbers and stopwords like with and of were
    # all removed as expected
