from project_name.models.model import Model
import numpy as np
import xgboost as xgb
from typing import Tuple
from sklearn.metrics import accuracy_score, log_loss


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


    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[float, float]:
        """
        Evaluates the model using accuracy and log loss.

        Args:
            X: A 2D array of lists of embeddings
            y: A 1D array of true labels

        Returns:
            Tuple of (log loss, accuracy)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        probs = self._model.predict_proba(X)
        preds = np.argmax(probs, axis=1)

        loss = log_loss(y, probs)
        acc = accuracy_score(y, preds)

        return {"loss": loss, "accuracy": acc}
