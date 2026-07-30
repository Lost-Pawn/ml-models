from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_data():
    train_ds = image_dataset_from_directory(
        "data/train",
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=123,
        class_names=["cats", "dogs"],
        image_size=(224, 224),
        batch_size=32
    )

    validation_ds = image_dataset_from_directory(
        "data/train",
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=123,
        class_names=["cats", "dogs"],
        image_size=(224, 224),
        batch_size=32
    )

    test_ds = image_dataset_from_directory(
        "data/test",
        labels="inferred",
        label_mode="int",
        class_names=["cats", "dogs"],
        image_size=(224, 224),
        batch_size=32
    )


    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    validation_ds = validation_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))
    return train_ds, validation_ds, test_ds

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_data()
    print(train_ds, val_ds, test_ds)