from tensorflow import keras
import numpy as np

model = keras.models.load_model("models/saved_model.h5")

def predict(sample):
    prediction = model.predict(sample)
    return prediction
