import os
import config
from data_generator import generate_dataset
from train import run_training
from evaluate import evaluate_model
from visualize import plot_confusion_matrix, plot_top_words
from model import load_model
from features import load_vectorizer


def main():
    # step 1, build the dataset if it is not already sitting on disk
    if not os.path.exists(config.RAW_DATA_PATH):
        df = generate_dataset()
        df.to_csv(config.RAW_DATA_PATH, index=False)
        print(f"generated dataset with {len(df)} rows")
    else:
        print("dataset already exists, skipping generation")

    # step 2, clean the text, split it, fit tfidf and train logistic regression
    model, vectorizer, X_test, y_test = run_training()

    # step 3, score the model on the held out test set
    results = evaluate_model(model, X_test, y_test)
    print(f"accuracy, {results['accuracy']:.4f}")
    print(f"precision, {results['precision']:.4f}")
    print(f"recall, {results['recall']:.4f}")
    print(f"f1 score, {results['f1']:.4f}")
    print(results["report"])

    # step 4, save the plots for a quick visual check
    plot_confusion_matrix(results["confusion_matrix"])
    plot_top_words(vectorizer, model)
    print(f"plots saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
