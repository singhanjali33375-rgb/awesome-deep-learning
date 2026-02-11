from model import build_model
from data_loader import load_data
from preprocessing import preprocess_data
import numpy as np

data = load_data("data/raw/sample.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X = preprocess_data(X)

model = build_model(X.shape[1])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=10)

model.save("models/saved_model.h5")

print("Model Trained and Saved")
