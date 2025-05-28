from project_name.models.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple


class LogisticRegressionClassifier(Model):
    """
    DecisionTreeClassifier class: inherits from the Model class

    Constructor Arguments:
        Inherits those of the model class: _parameters
        model: initialized with the Sklearn DecisionTreeClassifier
        model with its default arguments

    Methods:
        fit
        predict
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a DecisionTreeClassifier model instance.

        This constructor initializes a Sklearn DecisionTreeClassifier model
        and sets up the model's hyperparameters in the _parameters attribute.
        It is called with any additional arguments passed to the parent class
        initializer, allowing customization of the DecisionTreeClassifier's
        configuration.

        Args:
            *args: Positional arguments passed to the DecisionTreeClassifier
            initializer.
            **kwargs: Keyword arguments passed to the DecisionTreeClassifier
            initializer.

        Attributes:
            _type (str): The type of model, in this case, "classification".
            _model (DecTreeClass): The Sklearn DecisionTreeClassifier model
            instance, configured with the provided initialization arguments.
            _parameters (dict): A dictionary holding the hyperparameters
            of the model, initialized with the DecisionTreeClassifier's
            parameters.
        """
        super().__init__()
        self._model = LogReg(*args, **kwargs)
        self._parameters = {
            "hyperparameters": self._model.get_params()
        }

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data = None) -> None:
        """
        Fit method: fits the model
        to the provided observations and ground truth

        Arguments:
            X: a 2D array with each row containing
            features for each observation
            y: a 1D array containing the class labels
            for each observation

        Returns:
            None
        """
        X = np.asarray(X)

        self._model.fit(X, y)

        # Add model parameters to _parameters
        self._parameters.update({
            "strict parameters": {
            "classes": self._model.classes_,
            "coef": self._model.coef_,
            "intercept": self._model.intercept_,
            "n_features_in": self._model.n_features_in_,
            "n_iter": self._model.n_iter_,
            "feature_names_in": getattr(self._model, "feature_names_in_", None)}})

    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        Predict method: predicts the class labels for each observation

        Arguments:
            X: a 2D array with each row containing
            features for new observations

        Returns:
            a numpy array of predicted class labels
        
        return self._model.predict(X)
    """
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, str, np.ndarray]:
        """
        Evaluates the model and prints accuracy, classification report, and confusion matrix.

        Args:
            y: A 1D array of true labels
            y_pred: Optional precomputed predictions. If None, predictions will be computed.
        """
        y_pred = self._model.predict(X)
        
        y = np.asarray(y)

        accuracy = accuracy_score(y, y_pred)
        classification_rep =  classification_report(y, y_pred)
        confusion_mtrx =  confusion_matrix(y, y_pred)
        
        return accuracy, classification_rep, confusion_mtrx
