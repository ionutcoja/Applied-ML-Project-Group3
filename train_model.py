import pandas as pd
import numpy as np
import joblib
from project_code.features.text_cleaning import parse_words_dataset
from project_code.features.text_embeddings import embedding_words
from project_code.models.logistic_regression_model import LogisticRegressionClassifier
from project_code.models.DNN import DNNClassifier


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


def train(model, X_train, y_train) -> None:
    """
    Trains the model on the provided input data

    Returns:
        None - we just fit the model
    """
    model.fit(X_train, y_train)


def main():
    dataset_train = pd.read_csv('project_name/data/sa_spaeng_train.csv')

    X_train, y_train = preprocess_features(dataset_train)
    # input dimension for DNN
    input_dim = X_train.shape[1]

    # Initialize models
    baseline_model = LogisticRegressionClassifier()  
    advanced_model = DNNClassifier(input_dim=input_dim)

    # Train models on data
    train(baseline_model, X_train, y_train)
    train(advanced_model, X_train, y_train)

    # Save models
    joblib.dump(baseline_model, "logreg_model.joblib")
    joblib.dump(advanced_model, "advanced_model.joblib")


if __name__ == "__main__":
    main()
