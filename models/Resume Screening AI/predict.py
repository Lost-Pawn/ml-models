"""
This file loads the saved model, vectorizer and label encoder, then
uses them to predict the job category for a new resume along with a
confidence score. Run train_model.py first so the files inside the
model folder actually exist.

You can either edit the sample resumes near the bottom of this file
or import predict_resume into another script and pass your own text.
"""

import os
import joblib

from preprocess import clean_text


def load_artifacts():
    model = joblib.load(os.path.join("model", "resume_classifier.pkl"))
    vectorizer = joblib.load(os.path.join("model", "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join("model", "label_encoder.pkl"))
    return model, vectorizer, label_encoder


def predict_resume(resume_text, model, vectorizer, label_encoder):
    cleaned = clean_text(resume_text)
    vectorized = vectorizer.transform([cleaned])

    predicted_label = model.predict(vectorized)[0]
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    # predict_proba gives a confidence score for every category, we
    # just grab the highest one since that matches the predicted class
    probabilities = model.predict_proba(vectorized)[0]
    confidence = probabilities.max()

    return predicted_category, confidence


def main():
    model, vectorizer, label_encoder = load_artifacts()

    sample_resumes = [
        "Experienced software engineer skilled in Python, Java and REST API design, "
        "worked with microservices and used Docker and AWS for deployment.",

        "HR professional with a background in recruitment, onboarding and employee "
        "engagement, handled payroll and resolved workplace conflicts.",

        "Passionate about designing clean user interfaces, skilled in Figma and Adobe XD, "
        "created wireframes and worked with front end developers to ship pages."
    ]

    for resume in sample_resumes:
        category, confidence = predict_resume(resume, model, vectorizer, label_encoder)
        print(f"Predicted category, {category}, confidence, {confidence:.2%}")
        print(f"Resume snippet, {resume[:70]}...\n")

    # Result after running this file with the three sample resumes above,
    # the first one was correctly predicted as Software Development at
    # about 54 percent confidence, the second one as Human Resources at
    # about 61 percent, and the third one as Web Designing at about 65
    # percent. All three categories came out correct, but the confidence
    # numbers are moderate rather than very high, which makes sense since
    # logistic regression spreads probability across all six categories
    # and these short sample resumes only include a handful of keywords


if __name__ == "__main__":
    main()
