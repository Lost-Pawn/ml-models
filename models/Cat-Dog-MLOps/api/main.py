from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io
import mlflow.tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

loaded_model = mlflow.tensorflow.load_model("models:/catdog-classifier@production")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB").resize((224, 224))
    
    # Preprocess the image
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  

    # Make prediction
    prediction = loaded_model.predict(img_array)
    result = prediction[0][0]  

    label = "dog" if result > 0.5 else "cat"
    confidence = result if label == "dog" else 1 - result

    return {"prediction": label, "confidence": float(confidence)}