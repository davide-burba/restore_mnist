"""
Code partially copied from https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

from restore_mnist.data_handling import build_split_images


def train_model(train_images, test_images, n_epochs=10):
    """
    Train a model to understand if the two halfs of
    an image match.
    """
    trainX, trainY, testX, testY = build_data_for_classifier(train_images, test_images)
    trainX = prep_pixels(trainX)
    testX = prep_pixels(testX)
    return _train_model(trainX, trainY, testX, testY, n_epochs)


def build_data_for_classifier(train_images, test_images):
    """
    Build data for binary classifiers. Ones correspond to images
    with matching left and right halfs, zero with un-matching ones.
    """
    (
        half_train_images,
        train_binary_labels,
        half_test_images,
        test_binary_labels,
    ) = _build_data_for_classifier(train_images, test_images)

    # format dataset
    trainX = np.concatenate(
        [np.expand_dims(v, 2).transpose(2, 0, 1) for v in half_train_images], 0
    )
    testX = np.concatenate(
        [np.expand_dims(v, 2).transpose(2, 0, 1) for v in half_test_images], 0
    )
    trainY = np.array(train_binary_labels, dtype=int)
    testY = np.array(test_binary_labels, dtype=int)
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def _build_wrong_images_labels(split_images, labels):
    """
    Build a list of images with un-matching left and right half,
    and the corresponding labels (all zeros except for random ones).
    """
    lefts = [i for i, v in enumerate(labels) if "left" in v]
    rights = [i for i, v in enumerate(labels) if "right" in v]

    left_idx = np.random.choice(lefts, size=len(split_images))
    right_idx = np.random.choice(rights, size=len(split_images))

    wrong_images = [
        np.concatenate([split_images[i], split_images[j]], axis=1)
        for i, j in zip(left_idx, right_idx)
    ]
    wrong_labels = np.array(left_idx == right_idx + 1, dtype=int)

    return wrong_images, wrong_labels


def _build_data_for_classifier(train_images, test_images):
    split_train_images, train_labels = build_split_images(train_images)
    split_test_images, test_labels = build_split_images(test_images)

    wrong_train_images, wrong_train_binary_labels = _build_wrong_images_labels(
        split_train_images, train_labels
    )
    wrong_test_images, wrong_test_binary_labels = _build_wrong_images_labels(
        split_test_images, test_labels
    )

    half_train_images = train_images + wrong_train_images
    half_test_images = test_images + wrong_test_images

    train_binary_labels = np.concatenate(
        [np.repeat(1, len(train_images)), wrong_train_binary_labels]
    )
    test_binary_labels = np.concatenate(
        [np.repeat(1, len(test_images)), wrong_test_binary_labels]
    )
    return half_train_images, train_binary_labels, half_test_images, test_binary_labels


def prep_pixels(X):
    # reshape dataset to have a single channel
    X = X.reshape((X.shape[0], 28, 28, 1))
    # convert from integers to floats
    X = X.astype("float32")
    # normalize to range 0-1
    return X / 255.0


def define_model():
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(2, activation="softmax"))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _train_model(trainX, trainY, testX, testY, n_epochs):
    # define model
    model = define_model()
    # fit model
    history = model.fit(
        trainX,
        trainY,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(testX, testY),
        verbose=1,
        shuffle=True,
    )
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print("> %.3f" % (acc * 100.0))
    return model
