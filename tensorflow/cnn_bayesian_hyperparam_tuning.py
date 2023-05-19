import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten,MaxPooling1D, Reshape)
from tensorflow.keras.models import Sequential
import keras_tuner as kt
def build_model(X_train, y_train):
    """
    Build and compile the model using hyperparameters found through hyperparameter tuning.

    This function performs hyperparameter tuning using Keras Tuner to find the optimal
    hyperparameters for the model. It creates a model with a specified architecture and
    compiles it with the best hyperparameters. The hyperparameter search space includes
    the number of units in the first dense layer and the learning rate for the optimizer.

    Args:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.

    Returns:
        keras.models.Sequential: Compiled model with the optimal hyperparameters.
    """
    def model_builder(hp):
        model = keras.Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                         input_shape=(X_train.shape[1], 1)))

        # Tune the number of units in the first Dense layer
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))
        model.add(keras.layers.Dense(10))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(2, activation='sigmoid'))

        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    tuner = kt.Hyperband(model_builder,
                         objective='accuracy',
                         max_epochs=50,
                         factor=3,
                         overwrite=True)

    tuner.search(X_train, y_train, epochs=50)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    model = tuner.hypermodel.build(best_hps)
    model.summary()
    #Visualize the model
    keras.utils.plot_model(model, show_shapes=True,
                           show_layer_activations=True)

    return model


def plot_history(history):
    """
    Plot the accuracy and loss history of a trained model.

    This function takes the training history of a model and creates two plots: one for the
    accuracy and one for the loss. The plots show the training and validation performance
    over the epochs.

    Args:
        history (keras.callbacks.History): Training history of the model.
    """
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
   
#Refer to the 1d_cnn.py file for evaluation code