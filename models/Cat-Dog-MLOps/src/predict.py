import mlflow.tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

def predict_image(image_path):
    model = mlflow.tensorflow.load_model("models:/catdog-classifier@production")

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  

    # Make prediction
    prediction = model.predict(img_array)
    return prediction[0][0]  # return prob of the image being a dog (1) or a cat (0)

if __name__ == "__main__":
    result = predict_image("data/test/cats/cat.12490.jpg")  
    label = "dog" if result > 0.5 else "cat"
    confidence = result if label == "dog" else 1 - result
    print(f"Prediction: {label} (confidence: {confidence:.4f})")