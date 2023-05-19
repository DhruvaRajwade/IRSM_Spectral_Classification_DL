#This is a dump of all my code for Tensorflow. I did not get time to modularize it
#Please mix and match all you need, I'll try my best to document the code
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow.keras as keras
import keras_tuner as kt
from operator import mod
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" disable gpu if you're poor like me :
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
#import csv
from sklearn.decomposition import PCA
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import keras
import tensorflow as tf
from keras.utils import to_categorical
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from tensorflow.keras.layers import (Concatenate, Conv1D, Dense, Flatten,
                                     Input, MaxPooling1D, Reshape)
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, MaxPool1D, Concatenate, Dropout, BatchNormalization, Softmax, InputLayer
from keras.models import Sequential, Model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# First, apply the followind data transformations to the data:
"""

y_train, y_test=to_categorical(y_train), to_categorical(y_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train_new = tf.keras.preprocessing.sequence.pad_sequences(
    y_train, padding="post")

"""


def evaluate_model(model, trainX, trainy, testX, testy):
    """
    Evaluates a Keras model on the test data and returns evaluation metrics.

    Args:
    - model (tensorflow.keras.models.Sequential): The Keras model to be evaluated.
    - trainX (numpy.ndarray): The training data.
    - trainy (numpy.ndarray): The training labels.
    - testX (numpy.ndarray): The test data.
    - testy (numpy.ndarray): The test labels.

    Returns:
    - metrics (dict): A dictionary containing evaluation metrics.
    """

    # Compile model
    model.compile(loss=BinaryCrossentropy(),
                  optimizer=RMSprop(), metrics=['accuracy'])

    # Train model
    model.fit(trainX, trainy, epochs=100)

    # Evaluate model on test data
    y_pred = np.round(model.predict(testX))
    accuracy = accuracy_score(testy, y_pred)
    precision = precision_score(testy, y_pred)
    recall = recall_score(testy, y_pred)
    f1 = f1_score(testy, y_pred)

    # Return evaluation metrics
    metrics = {'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1': f1}

    return metrics


def summarize_results(scores):
    """
    Prints out mean and standard deviation of evaluation metrics.

    Args:
    - scores (list): A list of evaluation metric scores.
    """

    mean_accuracy = np.mean([score['accuracy'] for score in scores])
    std_accuracy = np.std([score['accuracy'] for score in scores])

    mean_precision = np.mean([score['precision'] for score in scores])
    std_precision = np.std([score['precision'] for score in scores])

    mean_recall = np.mean([score['recall'] for score in scores])
    std_recall = np.std([score['recall'] for score in scores])

    mean_f1 = np.mean([score['f1'] for score in scores])
    std_f1 = np.std([score['f1'] for score in scores])

    print(f"Accuracy: {mean_accuracy:.3f} (+/-{std_accuracy:.3f})")
    print(f"Precision: {mean_precision:.3f} (+/-{std_precision:.3f})")
    print(f"Recall: {mean_recall:.3f} (+/-{std_recall:.3f})")
    print(f"F1-score: {mean_f1:.3f} (+/-{std_f1:.3f})")


def run_experiment(model, trainX, trainy,  testX, testy, repeats, epochs):
    """
    Runs an experiment to evaluate a Keras model on the test data.

    Args:
    - model (tensorflow.keras.models.Sequential): The Keras model to be evaluated.
    - trainX (numpy.ndarray): The training data.
    - trainy (numpy.ndarray): The training labels.
    - testX (numpy.ndarray): The test data.
    - testy (numpy.ndarray): The test labels.
    - repeats (int): The number of times to repeat the experiment.
    - epochs (int): The number of epochs to train the model.

    Returns:
    - scores (list): A list of evaluation metric scores.
    """

    # Repeat experiment
    scores = []
    for r in range(repeats):
        print(f"Running experiment #{r+1}")
        score = evaluate_model(model, trainX, trainy, testX, testy)
        scores.append(score)

    # Summarize results
    summarize_results(scores)

    return scores




