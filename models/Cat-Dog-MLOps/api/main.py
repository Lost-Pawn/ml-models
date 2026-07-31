import io
import json
import logging
from datetime import datetime

import mlflow.tensorflow
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

logging.basicConfig(filename='predictions.log', level=logging.INFO)

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

    logging.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "confidence": float(confidence)
    }))
        
    return {"prediction": label, "confidence": float(confidence)}