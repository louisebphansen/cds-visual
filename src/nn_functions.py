# generic tools
import numpy as np
import cv2

# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import cifar10


def mnist_data():
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

    # normalise data
    data = data.astype("float")/255.0

    # (the data is already in black and white, so no need to greyscale)

    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=0.2)
    
    return X_train, X_test, y_train, y_test

def prep_data_cifar(X_train:np.ndarray, X_test:np.ndarray) -> np.ndarray:
    # do this smarter?
    '''
    This function preprocesses input training and testing image data to prepare it for scikit-learns classification methods.
    The preprocessing steps consists of converting the images to greyscale, scaling them and reshaping them.
    Arguments:
    - X_train: a numpy array containing the training data
    - X_test: a numpy array containing the testing data
    Returns:
    - Preprocessed numpy arrays X_train and X_test datasets ready to train a classifier.
    '''

    # convert each of the images to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # scale the images by dividing by 255
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0

    # reshape the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    return X_train_dataset, X_test_dataset

def cifar_data():
      # load X and y data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # manually define labels of the 10 classes
        labels = ["airplane", 
                "automobile", 
                "bird", 
                "cat", 
                "deer", 
                "dog", 
                "frog", 
                "horse", 
                "ship", 
                "truck"]
        
        X_train_data, X_test_data = prep_data_cifar(X_train, X_test)

        return X_train_data, X_test_data, y_train, y_test


def build_nn(X_train, X_test, y_train, y_test, input_size, first_layer, second_layer, output_size):
    
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    model = Sequential()
    model.add(Dense(first_layer, 
                input_shape=(input_size,), 
                activation="relu"))
    model.add(Dense(second_layer, activation="relu"))
    model.add(Dense(output_size, activation="softmax"))

    sgd = SGD(0.01)

    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=10, 
                    batch_size=32)

    print("[INFO] evaluating network...")
    predictions = model.predict(X_test, batch_size=32)


    report = (classification_report(y_test.argmax(axis=1), 
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_]))

    return report