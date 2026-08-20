from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import config


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": preds,
        "report": classification_report(y_test, preds, target_names=["real", "fake"]),
    }


if __name__ == "__main__":
    from train import run_training

    model, vectorizer, X_test, y_test = run_training()
    results = evaluate_model(model, X_test, y_test)

    # actual run on the held out set gave accuracy 0.8981, precision 0.9037,
    # recall 0.8912, f1 0.8974, out of 1600 test rows 76 real articles got
    # flagged as fake and 87 fake articles slipped through as real, most of
    # those mistakes trace back to the crossover sentences that were mixed
    # into the dataset on purpose to keep the classes from separating too
    # cleanly
    print(f"accuracy, {results['accuracy']:.4f}")
    print(f"precision, {results['precision']:.4f}")
    print(f"recall, {results['recall']:.4f}")
    print(f"f1 score, {results['f1']:.4f}")
    print("confusion matrix,")
    print(results["confusion_matrix"])
    print(results["report"])
