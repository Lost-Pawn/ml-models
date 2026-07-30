from src.data import load_data

def test_load_data_shapes():
    train_ds, val_ds, test_ds = load_data()
    images, labels = next(iter(train_ds))
    assert images.shape == (32, 224, 224, 3)
    assert labels.shape == (32,)