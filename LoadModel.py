from keras.models import load_model
import argparse
import pickle
import cv2

model = load_model("data/model_2c.h5")

# load the input image and resize it to the target spatial dimensions
image = cv2.imread("data/test.png")
output = image.copy()
image = cv2.resize(image, (32, 32)).flatten()

image = image.reshape((1, image.shape[0]))

# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0

preds = model.predict(image)

# 0: Start, 1: Stop
print(preds)


