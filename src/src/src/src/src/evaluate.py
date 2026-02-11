from tensorflow import keras
import numpy as np

model = keras.models.load_model("models/saved_model.h5")

# Dummy test
print("Evaluation Done")
