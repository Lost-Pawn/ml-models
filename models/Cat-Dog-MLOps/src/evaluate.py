from src.data import load_data
import mlflow.tensorflow

def evaluate_model():
    _, _, test_ds = load_data()
    model = mlflow.tensorflow.load_model("models:/catdog-classifier@production")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate_model()