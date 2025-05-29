from abc import abstractmethod
import numpy as np
from copy import deepcopy
from typing import Tuple, Union


class Model():
    """
    Model class: inherits from the ABC class

    Constructor Arguments:
        None

    Methods:
        fit
        predict
    """
    _model = None
    _parameters: dict = {}
    
    @property
    def parameters(self) -> dict:
        """Getter for _parameters

        Returns:
            str: deepcopy of _parameters
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray, validation_data = None) -> None:
        """
        Abstract method fit: fits the observations for a given model

        Arguments:
            observations
            ground_truth
            validation_data: optional validation data
        Returns:
                None
            """
        pass
    
    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Abstract method predict: predicts
        the value of the feature for each observation

        Arguments:
            observations: a 2D np.array with each row containing
            embeddings for each new sentence,
            with one column containing each list of embeddings

        Returns:
            a list of predictions of sentiment for each observation
        """
        pass

    @abstractmethod
    def evaluate(self, observations: np.ndarray, ground_truth: np.ndarray) -> Union[Tuple, str]:
        """
        Evaluates the model on test data.

        Args:
            observations (np.ndarray): Input features
            ground_truth (np.ndarray): True labels

        Returns:
            object: Evaluation metrics (format depends on model)
        """
        pass
