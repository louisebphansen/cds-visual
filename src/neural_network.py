# generic tools
import numpy as np
import argparse

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

# matplotlib
import matplotlib.pyplot as plt

from nn_functions import mnist_data, cifar_data, build_nn

# create a script that can run on both the CIFAR and the MNIST data!

ap = argparse.ArgumentParser()
ap.add_argument("--data", help="data to classify; can be CIFAR or MNIST dataset")
ap.add_argument("--input_size", type=int, help="size of input layer")
ap.add_argument("--first_layer", type=int, help="size of first hidden layer")
ap.add_argument("--second_layer", type=int, help="size of second hidden layer")
ap.add_argument("--output_size", type=int, help="size of output")
args = vars(ap.parse_args())

if args['data'] == "mnist":
    X_train, X_test, y_train, y_test = mnist_data()
    report = build_nn(X_train, X_test, y_train, y_test, args['input_size'], args['first_layer'], args['second_layer'], args['output_size'])
    print(report)

if args['data'] == "cifar":
    X_train, X_test, y_train, y_test = cifar_data()
    report = build_nn(X_train, X_test, y_train, y_test, args['input_size'], args['first_layer'], args['second_layer'], args['output_size'])
    print(report)
