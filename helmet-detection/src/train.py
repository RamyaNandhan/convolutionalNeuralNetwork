import os
from preprocess import get_data_generators
from model import build_model

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_PATH = "models/helmet_model.h5"

def train():
    train_gen, val_gen = get_data_generators(TRAIN_DIR, VAL_DIR)

    model = build_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()