# Author: Sven Kortekaas

from model import Model
from keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define model and train it
model = Model(num_classes=10, model_name="model", model_dir="./")
model.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=32, verbose=1)
