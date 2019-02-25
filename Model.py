import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import sys


# TUTORIAL:
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
# builds keras model for binary labeled image data
def build_model():
    #  argv[1l = path to the labeled data
    #  argv[2l = path to the output

    data_set = sys.argv[1]
    output = sys.argv[2]

    # initialize the data and labels
    print('Images are being loaded and randomly shuffled')
    data = []
    labels = []

    # grab the image paths and randomly shuffle them so for train and test set
    image_paths = sorted(list(paths.list_images(data_set)))
    random.seed(42)
    random.shuffle(image_paths)

    # loop over the input images
    for image_path in image_paths:
        # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), flatten the image into
        # 32x32x3=3072 pixel image into a list, and store the image in the data list
        image = cv2.imread(image_path)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)

        # extract the class label from the image path and update the labels list
        # first character of an image name is its label
        # image_path = data_set + "/" + label (0 or 1 as string) + image name
        label = image_path[len(data_set) + 1:len(data_set) + 2]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of the data for training and the remaining 25% for
    # testing
    (trainX, testX, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    train_y = keras.utils.np_utils.to_categorical(train_y)
    test_y = keras.utils.np_utils.to_categorical(test_y)

    # define the 3072-1024-512-2 architecture using Keras
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(2, activation="softmax"))

    # initialize our initial learning rate and # of epochs to train for
    epochs = 30

    # TODO: tune hyper parameters
    # init_lr = 0.01 # needed for sgd, not for adam
    # opt = SGD(lr=INIT_LR) # old optimizer, dgs is not as good as adam
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

    print('Model is being build.')

    # train the neural network
    h = model.fit(trainX, train_y, validation_data=(testX, test_y), epochs=epochs, batch_size=32)

    # evaluate the network
    predictions = model.predict(testX, batch_size=32)
    classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=['0', '1'])

    # plot the training loss and accuracy
    n = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(n, h.history["loss"], label="train_loss")
    plt.plot(n, h.history["val_loss"], label="val_loss")
    plt.plot(n, h.history["acc"], label="train_acc")
    plt.plot(n, h.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (CNN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(output + 'plot_adam.jpg')
    model.save(output + 'model_adam.h5')

    print('Model is build to ' + output)


# method can be used to predict probabilities for running/stopping machines for input image
def load_model():
    # argv[1l = path to the model
    # argv[2] = path to test image

    model = load_model(sys.argv[1])

    # load the input image and resize it to the target spatial dimensions
    image = cv2.imread(sys.argv[2])
    image = cv2.resize(image, (32, 32)).flatten()
    image = image.reshape((1, image.shape[0]))

    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0

    prediction = model.predict(image)

    # 0: probability for running machine, 1: probability for stopping machine
    print(prediction)


# cuts video in frames which can be labelled afterwards
def generate_frames_from_video():
    #  argv[1l = location of the video which will be cut in frames
    #  argv[2l = location of the frames

    video_path = sys.argv[1]
    data_set = sys.argv[2]
    ct = 0
    print('Generating frames from video')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError('Error opening video file')

    while cap.isOpened():
        ret, orig_frame = cap.read()

        # Resize frame to 70% of orignal size
        frame = cv2.resize(orig_frame, (0, 0), fx=0.70, fy=0.70)

        if ret:
            ct += 1
            cv2.imwrite(data_set + str(ct) + '.png', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    build_model()


if __name__ == '__main__':
    main()
