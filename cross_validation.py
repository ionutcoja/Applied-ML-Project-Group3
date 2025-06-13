from sklearn.model_selection import KFold
import numpy as np
from project_code.models.logistic_regression_model import LogisticRegressionClassifier
from project_code.models.DNN import DNNClassifier
import pandas as pd
from sklearn.metrics import accuracy_score 
from train_model import preprocess_features


def cross_validate_model(model_class, X, y, k=5, **kwargs):
    """
    Perform K-Fold Cross-Validation and print training and validation accuracy for each fold.

    Args:
        model_class: The class of the model to instantiate.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        k (int): Number of folds.
        kwargs: Any keyword arguments to pass to the model constructor.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_accuracies = []
    val_accuracies = []

    print(f"\nPerforming {k}-Fold Cross-Validation for {model_class.__name__}...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        model = model_class(**kwargs)
        model.fit(X[train_idx], y[train_idx])

        # Training accuracy
        y_train_pred = model.predict(X[train_idx])
        train_acc = accuracy_score(y[train_idx], y_train_pred)
        train_accuracies.append(train_acc)

        # Validation accuracy
        y_val_pred = model.predict(X[val_idx])
        val_acc = accuracy_score(y[val_idx], y_val_pred)
        val_accuracies.append(val_acc)

        print(f"Fold {fold}: Train Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")

    print("\nSummary:")
    print(f"Average Train Accuracy: {np.mean(train_accuracies):.4f}")
    print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f}\n")
    
def main():
    # the training dataset will be split into k folds
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')

    X_train, y_train = preprocess_features(dataset_train)
    # input dimension for DNN
    input_dim = X_train.shape[1]
    
    cross_validate_model(LogisticRegressionClassifier, X_train, y_train, k=5, max_iter=1000)
    cross_validate_model(DNNClassifier, X_train, y_train, k=5, input_dim=input_dim)
