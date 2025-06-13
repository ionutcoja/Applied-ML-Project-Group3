from project_name.models.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score


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
        Initializes a Logistic Regression model instance.

        This constructor initializes a Sklearn LogisticRegressionClassifier model
        and sets up the model's hyperparameters in the _parameters attribute.
        It is called with any additional arguments passed to the parent class
        initializer, allowing customization of the DecisionTreeClassifier's
        configuration.

        Args:
            *args: Positional arguments passed to the LogisticRegressionClassifier
            initializer.
            **kwargs: Keyword arguments passed to the LogisticRegressionClassifier
            initializer.

        Attributes:
            _type (str): The type of model, in this case, "classification".
            _model (LogReg): The Sklearn DecisionTreeClassifier model
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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

        self._parameters.update({
            "strict parameters": {
            "classes": self._model.classes_,
            "coef": self._model.coef_,
            "intercept": self._model.intercept_,
            "n_features_in": self._model.n_features_in_,
            "n_iter": self._model.n_iter_,
            "feature_names_in": getattr(self._model, "feature_names_in_", None)}})


    def predict(self, X: np.ndarray) -> np.ndarray:
        
        return self._model.predict(X)

    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Evaluates the model and returns a formatted string of metrics:
        accuracy, classification report, confusion matrix, and F1 score.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.

        Returns:
            str: A formatted string with evaluation metrics.
        """
        y_pred = self._model.predict(X)
        y = np.asarray(y)

        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        report = classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)

        formatted_metrics = (
            f"Evaluation Metrics\n"
            f"{'='*40}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"F1 Score: {f1:.4f}\n\n"
            f"Classification Report:\n{report}\n"
            f"Confusion Matrix:\n{matrix}\n"
        )

        return formatted_metrics
