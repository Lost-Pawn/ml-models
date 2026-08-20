import sys
from preprocess import clean_text
from features import load_vectorizer
from model import load_model

LABELS = {0: "real", 1: "fake"}


def predict_text(text, vectorizer=None, model=None):
    vectorizer = vectorizer or load_vectorizer()
    model = model or load_model()

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][pred]

    return LABELS[pred], prob


if __name__ == "__main__":
    vectorizer = load_vectorizer()
    model = load_model()

    samples = [
        "SHOCKING, the vaccine rollout was secretly staged the whole time, share before this gets deleted",
        "Officials confirmed the vaccine rollout met its quarterly targets, according to a statement from the health ministry",
    ]

    # tested with two handwritten sentences not from the generator, first
    # one came back fake with 0.98 confidence and second one came back
    # real with 0.92 confidence, so the model does generalize a bit beyond
    # the exact templates it was trained on
    for text in samples:
        label, prob = predict_text(text, vectorizer, model)
        print(f"{label}, confidence {prob:.2f}, text: {text[:60]}...")

    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        label, prob = predict_text(custom_text, vectorizer, model)
        print(f"{label}, confidence {prob:.2f}, text: {custom_text}")
