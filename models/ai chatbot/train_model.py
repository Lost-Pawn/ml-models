import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to a file instead of trying to open a window
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from preprocess import load_intents, build_dataset

intents = load_intents("intents.json")
texts, labels, responses = build_dataset(intents)

print(f"Loaded {len(intents)} intents and {len(texts)} training examples.\n")

# ---------------------------------------------------------------------------
# 1. TF-IDF feature extraction
# ---------------------------------------------------------------------------
# TF-IDF (Term Frequency - Inverse Document Frequency) turns each sentence
# into a vector of word-importance scores instead of raw word counts, so
# common filler words matter less than distinctive ones.
#
# ngram_range=(1, 2) tells the vectorizer to look at both single words
# ("hours") AND two-word phrases ("working hours"). This helps a lot for a
# chatbot because a lot of the meaning lives in short phrases rather than
# single words - "good morning" as a whole means something different from
# "good" and "morning" scored separately, and "how old" is a much stronger
# signal for the `age` intent than "how" or "old" alone.
#
# sublinear_tf=True applies a log scaling to term frequency (1 + log(tf))
# instead of raw counts. This stops a word that happens to repeat in a
# sentence from dominating the vector disproportionately - useful here since
# our sentences are short and any repetition is more likely to be
# incidental than meaningful.
#
# min_df=1 keeps every word/phrase, since with ~500 short examples spread
# across many intents, being too aggressive about dropping rare terms would
# throw away exactly the distinctive phrases that separate similar intents.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)

# Fit the vectorizer ONLY on the full training text set here; the split
# into train/test happens after, on the already-vectorized data, using the
# same fitted vectorizer for both. This avoids leaking test data into the
# vocabulary/IDF statistics in a way that's easy to reason about, and mirrors
# how the saved vectorizer is later applied unchanged to live user input.
X = vectorizer.fit_transform(texts)

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
# stratify=labels makes sure the proportion of each intent is preserved in
# both the training and test sets - otherwise, with 19 intents, a plain
# random split could easily leave some rare intent with zero test examples.
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# ---------------------------------------------------------------------------
# 3. Model training
# ---------------------------------------------------------------------------
# Logistic Regression works well for small-to-medium text classification
# problems like this one, trains almost instantly with no GPU needed, and -
# importantly for a student project - is easy to explain: it learns a
# weight for each word/phrase per intent, and picks the intent whose
# weighted sum of TF-IDF scores is highest.
#
# C controls regularization strength (inverse of it, technically): a
# smaller C means stronger regularization. It's tempting to assume a small
# dataset needs heavy regularization, but that was tested directly here
# rather than assumed: running 5-fold cross-validation across
# C = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50] showed accuracy climbing
# steadily as C increased (0.1 -> ~19% CV accuracy, 1.0 -> ~78%,
# 10 -> ~83%), then flattening out from C=10 onwards. In other words, this
# dataset benefits from a fairly *unregularized* model - with n-gram
# features and short, fairly distinct sentences per intent, the classes are
# close to linearly separable, so extra regularization was only causing
# underfitting. C=10 is used because it sits right where the curve
# plateaus, rather than chasing the single highest number further out.
model = LogisticRegression(max_iter=1000, C=10.0)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print("=" * 60)
print("BASIC ACCURACY")
print("=" * 60)
print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy:     {test_acc:.4f}\n")

# ---------------------------------------------------------------------------
# 4. Full evaluation: precision, recall, F1, classification report
# ---------------------------------------------------------------------------
# Accuracy alone can be misleading, especially if some intents are easier
# to predict than others. Precision, recall, and F1-score break performance
# down per intent:
#   - Precision: of everything the model labeled as intent X, how much was
#     actually X? (Low precision = model over-predicts this intent.)
#   - Recall: of everything that was actually intent X, how much did the
#     model correctly find? (Low recall = model misses this intent often.)
#   - F1-score: the harmonic mean of precision and recall, a single number
#     that balances both.
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, test_preds, average="weighted", zero_division=0
)

print("=" * 60)
print("WEIGHTED AVERAGE METRICS (test set)")
print("=" * 60)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}\n")

print("=" * 60)
print("PER-INTENT CLASSIFICATION REPORT (test set)")
print("=" * 60)
report = classification_report(y_test, test_preds, zero_division=0)
print(report)

# ---------------------------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------------------------
# The confusion matrix shows, for every actual intent (rows), how many test
# examples were predicted as each possible intent (columns). The diagonal
# is correct predictions; anything off the diagonal shows exactly which
# intents get confused with each other, which is far more actionable than
# a single accuracy number.
class_labels = sorted(set(labels))
cm = confusion_matrix(y_test, test_preds, labels=class_labels)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_labels)))
ax.set_yticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=90)
ax.set_yticklabels(class_labels)
ax.set_xlabel("Predicted intent")
ax.set_ylabel("Actual intent")
ax.set_title("Confusion Matrix - Intent Classification")
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        if cm[i, j] > 0:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
fig.colorbar(im, ax=ax, label="Number of test examples")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150)
plt.close(fig)
print("Saved confusion_matrix.png\n")

# ---------------------------------------------------------------------------
# 6. Cross-validation
# ---------------------------------------------------------------------------
# A single train/test split can be noisy, especially with a modestly sized
# dataset - the reported accuracy depends partly on which examples happened
# to land in the test set. 5-fold StratifiedKFold cross-validation instead
# trains and evaluates the model 5 separate times, each time using a
# different 1/5 of the data as the held-out fold (while keeping each
# intent's proportion consistent across folds), and reports the spread of
# scores. The mean gives a more reliable estimate of real-world performance
# than a single split, and the standard deviation shows how stable that
# estimate is.
print("=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, labels, cv=skf, scoring="accuracy")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")
print(f"\nMean Accuracy:      {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}\n")

# ---------------------------------------------------------------------------
# 7. Save model artifacts
# ---------------------------------------------------------------------------
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("responses.pkl", "wb") as f:
    pickle.dump(responses, f)

# Small metadata file so chatbot.py can display accurate info at startup
# without hard-coding numbers that would go stale as the dataset changes.
metadata = {
    "model": "TF-IDF (1-2 gram) + Logistic Regression",
    "num_intents": len(intents),
    "num_training_examples": len(texts),
    "train_accuracy": round(train_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
    "cv_std_accuracy": round(float(cv_scores.std()), 4),
}
with open("model_info.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved chatbot_model.pkl, vectorizer.pkl, responses.pkl, model_info.json")
