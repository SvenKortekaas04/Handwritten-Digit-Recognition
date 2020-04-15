# Author: Sven Kortekaas

from model import Model
import cv2

# Define the model and load weights
model = Model(num_classes=10)
model.load_weights("model.h5")

# Load image
image = cv2.imread("assets/image.png")

# Make prediction
prediction = model.predict(image)

# Print the digit that was predicted by the model
print(prediction.argmax())
