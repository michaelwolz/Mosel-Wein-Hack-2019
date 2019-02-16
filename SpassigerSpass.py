# set the matplotlib backend so figures can be saved in the background
import matplotlib

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

#todo: auf richtiges DS
dataset = "data/srames"

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), flatten the image into 32x32x3=3072
    # pixel image into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = imagePath[12:13]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
dataX = np.array(data, dtype="float") / 255.0
labelsX = np.array(labels)

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for
# testing
(trainX, testX, trainY, testY) = train_test_split(dataX, labelsX, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary classification you should use Keras' to_categorical
# function instead as the scikit-learn's LabelBinarizer will not return a vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile the model using SGD as our optimizer and categorical cross-entropy loss (you'll want to use binary_
# crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(lb.classes_)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (CNN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("data/plot.jpg")

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save('data/model_2c.h5')

