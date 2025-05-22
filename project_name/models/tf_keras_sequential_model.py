from models.model import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple

class KerasSequentialClassifier(Model):
    """
    KerasSequentialClassifier class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the Model class

    Attributes:
        _type (str): The type of model ("classification")
        _model (tf.keras.Model): A Keras Sequential model instance
        _parameters (dict): Dictionary holding hyperparameters and internal config
    """

    def __init__(self, input_shape: int, num_classes: int = 3, *args, **kwargs) -> None:
        """
        Initializes a Keras Sequential model for classification.

        Args:
            input_shape (int): Number of features in the input data.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self._type = "classification"

        self._model = tf.keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        self._model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self._parameters = {
            "input_shape": input_shape,
            "num_classes": num_classes,
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"]
        }

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data=None, epochs: int = 100, batch_size: int = 32) -> None:
        """
        Trains the Keras Sequential model.

        Args:
            X: Input features as a 2D numpy array
            y: Target labels as a 1D numpy array
            validation_data: Tuple (X_val, y_val) for validation
            epochs: Number of training epochs
            batch_size: Size of training batches
        """
        X = np.asarray(X)
        y = np.asarray(y)

        history = self._model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        self._parameters["training_history"] = history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels using the trained model.

        Args:
            X: A 2D array of input features

        Returns:
            A 1D array of predicted class labels
        """
        probs = self._model.predict(X)
        return np.argmax(probs, axis=1)
    

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
        """
        Evaluates the model on test data.

        Args:
            X: A 2D array of input features
            y: A 1D array of true labels

        Returns:
            A tuple (loss, accuracy)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        loss, accuracy = self._model.evaluate(X, y, verbose=0)
        return loss, accuracy