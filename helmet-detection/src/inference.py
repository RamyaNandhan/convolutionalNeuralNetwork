import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "models/helmet_model.h5"
IMG_SIZE = 224

def load_trained_model():
    return load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    model = load_trained_model()
    img = preprocess_image(image_path)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "No Helmet ❌"
    else:
        return "Helmet ✅"

if __name__ == "__main__":
    image_path = "sample.jpg"
    result = predict(image_path)
    print(f"Prediction: {result}")