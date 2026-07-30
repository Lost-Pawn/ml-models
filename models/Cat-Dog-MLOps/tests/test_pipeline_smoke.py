from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.model import create_model

def test_training_runs_without_error():
    ds = image_dataset_from_directory(
        "tests/mini_dataset",
        labels="inferred",
        label_mode="int",
        class_names=["cats", "dogs"],
        image_size=(224, 224),
        batch_size=4
    )
    ds = ds.map(lambda x, y: (preprocess_input(x), y))

    model = create_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(ds, epochs=1)

    assert history is not None
    assert "loss" in history.history