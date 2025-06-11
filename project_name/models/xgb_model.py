from project_name.models.model import Model
import numpy as np
import xgboost as xgb
from typing import Tuple
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, f1_score


class XGBoostClassifier(Model):
    """
    XGBoostClassifier class: wraps an XGBoost classifier

    Attributes:
        _type (str): The type of model ("classification")
        _model (xgb.XGBClassifier): An XGBoost classifier instance
        _parameters (dict): Dictionary holding hyperparameters and training history
    """

    def __init__(self, num_classes: int = 3, *args, **kwargs) -> None:
        super().__init__()
        self._type = "classification"
        self._num_classes = num_classes

        self._model = xgb.XGBClassifier(
            n_estimators=200, # equivalent for number of training epochs
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=num_classes
        )

        self._parameters = {
            "num_classes": num_classes,
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "eval_metric": "mlogloss"
        }


    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Tuple[np.ndarray, np.ndarray] = None) -> None:
        """
        Trains the XGBoost model.

        Args:
            X: Input features
            y: Target labels
            validation_data: Tuple (X_val, y_val) for early stopping
        """
        X = np.asarray(X)
        y = np.asarray(y)

        eval_set = [(X, y)]
        if validation_data:
            eval_set.append(validation_data)

        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=True
        )

        self._parameters["training_history"] = self._model.evals_result()


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input features.

        Args:
            X: A 2D array of lists of embeddings

        Returns:
            A 1D array of predicted sentiment labels
        """
        X = np.asarray(X)
        probs = self._model.predict_proba(X)
        return np.argmax(probs, axis=1)


    def evaluate(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Evaluates the model and returns a formatted string of metrics:
        accuracy, classification report, confusion matrix, log loss, and F1 score.

        Args:
            X: A 2D array of input features.
            y: A 1D array of true labels.

        Returns:
            str: A formatted string with evaluation metrics.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        probs = self._model.predict_proba(X)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")
        report = classification_report(y, preds)
        matrix = confusion_matrix(y, preds)

        formatted_metrics = (
            f"Evaluation Metrics\n"
            f"{'='*40}\n"
            f"Accuracy: {acc:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
            f"Classification Report:\n{report}\n"
            f"Confusion Matrix:\n{matrix}\n"
        )

        return formatted_metrics

    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, average="weighted")
        }