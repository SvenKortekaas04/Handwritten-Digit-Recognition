# Handwriting-Recognition

Detect handwritten digits using a convolutional neural network in Python

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Firstly, ensure you have Python 3.6 or higher installed

### Installing

Clone this repository

```
git clone https://github.com/SvenKortekaas04/Handwriting-Recognition.git
```

Make sure you have the right dependencies installed

```
pip install -r requirements.txt
```

## How To Use

For all code examples see the examples folder

### Training a model

As shown in the training example we are training a model on the MNIST dataset of handwritten digits. This dataset consists of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

#### Example

```
from model import Model
from keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define model and train it
model = Model(num_classes=10, model_name="model", model_dir="./")
model.train(x_train, y_train, x_test, y_test, epochs=10, batch_size=32, verbose=1)
```

### Making predictions with the model

```
from model import Model
import cv2

# Define the model and load weights
model = Model(num_classes=10)
model.load_weights("model.h5")

# Load image
image = cv2.imread("assets/image.png")

# Make prediction
prediction = model.predict(image)

# Print the raw output of the model
print(prediction)
```

The above example returns the raw output of the model. To get the actual digit predicted by the model use `prediction.argmax()`.

## Built With

* [Keras](https://keras.io/)

## Authors

* **Sven Kortekaas** - [SvenKortekaas04](https://github.com/SvenKortekaas04)
