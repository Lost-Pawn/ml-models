from src.data import load_data
from src.model import create_model
import mlflow
import mlflow.tensorflow

def train_model():
    # Load the data
    train_ds, val_ds, test_ds = load_data()

    model = create_model(num_classes=1)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # MLflow run
    mlflow.set_experiment("catdog-transfer-learning")
    with mlflow.start_run(run_name="mobilenetv2-head-only"):
        mlflow.tensorflow.autolog()  # enable automatic logging of tf metrics and parameters

        history = model.fit(train_ds, validation_data=val_ds, epochs=5)
    return history

if __name__ == "__main__":
    history = train_model()
    print(history.history)