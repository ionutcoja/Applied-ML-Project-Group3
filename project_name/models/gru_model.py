from project_name.models.model import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple


class KerasBiGRUClassifier(Model):
    """
    KerasBiGRUClassifier: Wraps a Bidirectional GRU classifier using Keras Sequential API.

    Attributes:
        _type (str): Model type ("classification")
        _model (tf.keras.Model): The Keras model instance
        _parameters (dict): Stores model configuration and training history
    """

    def __init__(self, sequence_len: int, embedding_dim: int, num_classes: int = 3, *args, **kwargs):
        super().__init__()
        self._type = "classification"
        self._sequence_len = sequence_len
        self._embedding_dim = embedding_dim
        self._num_classes = num_classes

        self._parameters = {
            "sequence_len": sequence_len,
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"]
        }

        self._model = tf.keras.Sequential([
            layers.Reshape((sequence_len, embedding_dim)),
            layers.Bidirectional(layers.GRU(128, return_sequences=False)),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])

        self._model.compile(
            optimizer=self._parameters["optimizer"],
            loss=self._parameters["loss"],
            metrics=self._parameters["metrics"]
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Tuple[np.ndarray, np.ndarray] = None,
            epochs: int = 50, batch_size: int = 32) -> None:
        """
        Trains the BiGRU model.

        Args:
            X: Input features (shape: [samples, sequence_len * embedding_dim])
            y: Target labels
            validation_data: Optional (X_val, y_val) tuple
            epochs: Number of epochs
            batch_size: Batch size
        """
        X = np.asarray(X).reshape((-1, self._sequence_len, self._embedding_dim))
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
        Predicts class labels for input features.

        Args:
            X: Input features (shape: [samples, sequence_len * embedding_dim])

        Returns:
            A 1D array of predicted class labels
        """
        X = np.asarray(X).reshape((-1, self._sequence_len, self._embedding_dim))
        probs = self._model.predict(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluates the model on test data.

        Args:
            X: Input features (shape: [samples, sequence_len * embedding_dim])
            y: Ground truth labels

        Returns:
            Tuple of (loss, accuracy)
        """
        X = np.asarray(X).reshape((-1, self._sequence_len, self._embedding_dim))
        y = np.asarray(y)
        loss, accuracy = self._model.evaluate(X, y, verbose=0)
        return loss, accuracy
