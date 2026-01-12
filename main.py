from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Brain Tumor CNN API")


MODEL_PATH = "cnn_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)


CLASS_NAMES = ["No Tumor", "Tumor"]

IMG_SIZE = 224  


@app.get("/")
def home():
    return {"message": "Brain Tumor CNN API is running 🧠"}


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2)
    }
