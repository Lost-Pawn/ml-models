from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_model(num_classes=1):
    base_model = MobileNetV2(
        weights="imagenet", 
        include_top=False, 
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()