import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE

from project_name.features.text_cleaning import parse_words_dataset
from project_name.features.text_embeddings import embedding_words
from project_name.models.logistic_regression_model import LogisticRegressionClassifier
from project_name.models.DNN import DNNClassifier


def preprocess_features(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses and embeds dataset, then encodes labels.

    Returns:
        Tuple of feature matrix X and label vector y.
    """
    parse_words_dataset(dataset)
    X = embedding_words(dataset)

    label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    y = dataset['sa'].map(label_mapping).astype(np.int32).to_numpy()

    return X, y


def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE to balance class distribution in training data.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def train(model, X_train, y_train) -> None:
    model.fit(X_train, y_train)


def main():
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')

    X_train, y_train = preprocess_features(dataset_train)

    # Apply SMOTE to handle class imbalance
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Initialize models
    baseline_model = LogisticRegressionClassifier()
    input_dim = X_train.shape[1]
    advanced_model = DNNClassifier(input_dim=input_dim)

    # Train models on resampled (balanced) data
    train(baseline_model, X_train_balanced, y_train_balanced)
    train(advanced_model, X_train_balanced, y_train_balanced)

    # Save models
    joblib.dump(baseline_model, "logreg_model.joblib")
    joblib.dump(advanced_model, "advanced_model.joblib")


if __name__ == "__main__":
    main()
